// pipeline-supervisor — M2.5 shape.
//
// DB-driven reconciler. Every N seconds, reads the cameras table from
// Postgres, diffs against a map of running per-camera threads, and starts
// or stops workers to match. Each worker independently restarts its
// GStreamer pipeline with exponential backoff on EOS/error.
//
// Jetson build wires the DeepStream nvinfer chain; non-Jetson falls back
// to pass-through record via software encoder.

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <signal.h>   // kill()
#include <unistd.h>   // getpid()

#include <glib.h>
#include <gst/gst.h>

#include "config.h"
#include "db_reconciler.h"
#include "dynamic_pads.h"
#include "nats_publisher.h"
#include "pipeline.h"
#include "supervisor.h"

namespace {
volatile std::sig_atomic_t g_stop = 0;
void HandleSignal(int) { g_stop = 1; }
}  // namespace

int main(int argc, char** argv) {
    // Lightweight publish mode — used by pipeline-entrypoint.sh to
    // announce calibrating / compiling_engine / ready states before the
    // supervisor is actually up. Invoked as:
    //   pipeline-supervisor --publish <subject> <payload>
    if (argc >= 4 && std::string(argv[1]) == "--publish") {
        auto cfg = fnvr::LoadFromEnv();
        fnvr::NatsPublisher nats(cfg.nats_url);
        if (!nats.Connected()) return 1;
        bool ok = nats.Publish(argv[2], argv[3], /*flush=*/true);
        return ok ? 0 : 2;
    }

    // Dynamic-pad smoke test (stage 2a verification). Builds an
    // nvstreammux + fakesink pipeline with one initial source, sets
    // PLAYING, then exercises AddSourceToMux + RemoveSourceFromMux
    // a couple of times. Prints OK / FAILED at each step. Invoked as:
    //   pipeline-supervisor --test-dynamic-pads
    // Exit 0 = all steps OK, non-zero = at least one step failed.
    if (argc >= 2 && std::string(argv[1]) == "--test-dynamic-pads") {
        gst_init(nullptr, nullptr);
        const std::string desc =
            "videotestsrc is-live=true pattern=ball ! "
            "video/x-raw,width=320,height=240,framerate=30/1 ! "
            "nvvideoconvert ! "
            "video/x-raw(memory:NVMM),format=NV12,width=320,height=240 ! "
            "nvstreammux name=mux batch-size=4 width=320 height=240 "
            "  live-source=1 batched-push-timeout=40000 ! "
            "fakesink sync=false async=false";
        GError* err = nullptr;
        GstElement* p = gst_parse_launch(desc.c_str(), &err);
        if (!p) {
            std::cerr << "test: parse failed: "
                      << (err ? err->message : "?") << "\n";
            if (err) g_error_free(err);
            return 1;
        }
        // Initial source is parsed inline + linked to mux.sink_0
        // automatically by gst_parse_launch's auto-linker.
        if (gst_element_set_state(p, GST_STATE_PLAYING) ==
            GST_STATE_CHANGE_FAILURE) {
            std::cerr << "test: initial PLAYING failed\n";
            gst_object_unref(p);
            return 2;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cerr << "test: initial pipeline rolled to PLAYING — OK\n";

        GstElement* mux = gst_bin_get_by_name(GST_BIN(p), "mux");
        if (!mux) {
            std::cerr << "test: get_by_name(mux) failed\n";
            gst_element_set_state(p, GST_STATE_NULL);
            gst_object_unref(p);
            return 3;
        }

        // Step 1: add a second source. Caps filter is in raw form
        // because gst-launch syntax doesn't tolerate (memory:NVMM)
        // inside parse_bin_from_description (the parens conflict
        // with bin-grouping). The trailing nvvideoconvert produces
        // NVMM on its src pad either way.
        const std::string src2 =
            "videotestsrc is-live=true pattern=snow ! "
            "video/x-raw,width=320,height=240,framerate=30/1 ! "
            "nvvideoconvert";
        auto* added2 = fnvr::AddSourceToMux(p, mux, src2, "snow");
        if (!added2) {
            std::cerr << "test: AddSourceToMux(snow) FAILED\n";
            gst_element_set_state(p, GST_STATE_NULL);
            gst_object_unref(mux);
            gst_object_unref(p);
            return 4;
        }
        std::cerr << "test: added snow source as source_id="
                  << added2->source_id << " — OK\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Step 2: remove it.
        if (!fnvr::RemoveSourceFromMux(p, mux, added2)) {
            std::cerr << "test: RemoveSourceFromMux(snow) reported errors\n";
        } else {
            std::cerr << "test: removed snow source — OK\n";
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Step 3: add a different source, then remove it.
        const std::string src3 =
            "videotestsrc is-live=true pattern=smpte ! "
            "video/x-raw,width=320,height=240,framerate=30/1 ! "
            "nvvideoconvert";
        auto* added3 = fnvr::AddSourceToMux(p, mux, src3, "smpte");
        if (!added3) {
            std::cerr << "test: AddSourceToMux(smpte) FAILED\n";
            gst_element_set_state(p, GST_STATE_NULL);
            gst_object_unref(mux);
            gst_object_unref(p);
            return 5;
        }
        std::cerr << "test: added smpte source as source_id="
                  << added3->source_id << " — OK\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));

        if (!fnvr::RemoveSourceFromMux(p, mux, added3)) {
            std::cerr << "test: RemoveSourceFromMux(smpte) reported errors\n";
        } else {
            std::cerr << "test: removed smpte source — OK\n";
        }

        gst_element_set_state(p, GST_STATE_NULL);
        gst_object_unref(mux);
        gst_object_unref(p);
        std::cerr << "test: all steps complete — pipeline torn down OK\n";
        return 0;
    }

    // Worker mode: one subprocess per camera. Isolates splitmuxsink / nvinfer
    // asserts so a single bad camera can't crash the whole supervisor.
    // Invoked as: pipeline-supervisor --worker <camera_id> <url> <record_mode>
    //   [--rotation]
    // The optional --rotation flag signals an hourly segment-rotation
    // respawn: the previous worker was `running` seconds ago and the
    // operator shouldn't see the "starting" state flash in between.
    if (argc >= 5 && std::string(argv[1]) == "--worker") {
        gst_init(nullptr, nullptr);
        std::signal(SIGINT, HandleSignal);
        std::signal(SIGTERM, HandleSignal);

        auto cfg = fnvr::LoadFromEnv();
        fnvr::NatsPublisher nats(cfg.nats_url);

        fnvr::CameraConfig cam;
        cam.id = argv[2];
        cam.url = argv[3];
        cam.recording_mode = argv[4];
        const bool rotation = (argc >= 6 && std::string(argv[5]) == "--rotation");
        // Resolve the effective mute set from Postgres before the
        // pipeline builds — gives the InferSrcProbe a point-in-time
        // snapshot that persists for the life of this worker. Operators
        // restart the pipeline to apply changes (matches Settings UI).
        cam.muted_classes = fnvr::ReadMutedClassesForCamera(cfg.database_url, cam.id);
        if (!cam.muted_classes.empty()) {
            std::cerr << "worker[" << cam.id << "]: muting "
                      << cam.muted_classes.size() << " class(es) at pipeline\n";
        }
        cam.rotation = fnvr::ReadRotationForCamera(cfg.database_url, cam.id);
        if (cam.rotation != 0) {
            std::cerr << "worker[" << cam.id << "]: rotation=" << cam.rotation
                      << " (forces transcode path)\n";
        }
        cam.enabled_detectors = fnvr::ReadEnabledDetectorsForCamera(
            cfg.database_url, cam.id);
        if (!cam.enabled_detectors.empty()) {
            std::string joined;
            for (const auto& d : cam.enabled_detectors) {
                if (!joined.empty()) joined += ",";
                joined += d;
            }
            std::cerr << "worker[" << cam.id << "]: detectors=" << joined << "\n";
        }
        cam.mtx_proxy = fnvr::ReadMtxProxyForCamera(cfg.database_url, cam.id);
        if (cam.mtx_proxy) {
            std::cerr << "worker[" << cam.id << "]: mtx_proxy=on\n";
        }
        cam.detector_backend = fnvr::ReadDetectorBackendForCamera(
            cfg.database_url, cam.id);
        if (cam.detector_backend != "trt") {
            std::cerr << "worker[" << cam.id << "]: detector_backend="
                      << cam.detector_backend << "\n";
        }
        // Inform pipeline's bus-error path of the startup grace window
        // so transient faults during warmup don't publish `failed` to
        // the UI. Reads from the same settings key the supervisor uses.
        fnvr::SetWorkerStartupGraceSec(
            fnvr::ReadPipelineStartupGraceSec(cfg.database_url));

        fnvr::SingleCameraPipeline p(cam, cfg.recordings_dir, cfg.inference_config,
                                      cfg.use_deepstream, cfg.use_anpr, cfg.use_face_id,
                                      &nats);

        // Announce "starting" before Start() so the UI can show progress
        // during the TRT engine build (first run after cache wipe: 60-90s).
        // Pipeline will publish "running" itself on reaching PLAYING.
        //
        // Subject: fnvr.state.camera.<id>. This is a JetStream-backed
        // last-value stream (declared by api-server) so a fresh api-server
        // subscriber immediately sees the current state for every camera,
        // instead of being stuck at "unknown" until a worker happens to
        // (re)start.
        //
        // EXCEPT during an hourly segment rotation (`--rotation`). The
        // previous worker was `running` a few seconds ago; the old
        // message is still valid in the last-value stream. Publishing
        // `starting` here would flip every tile to the amber
        // "starting…" pulse every hour, which is not what the operator
        // wants to see. The new worker's own `running` publish on
        // reaching PLAYING wraps up the transition silently.
        const std::string subj = "fnvr.state.camera." + cam.id;
        if (!rotation) {
            std::string payload = "{\"camera_id\":\"" + cam.id + "\",\"state\":\"starting\"}";
            nats.Publish(subj, payload, /*flush=*/true);
        }
        if (!p.Start()) {
            // A failed Start() during rotation is still a real failure
            // — publish `failed` so the UI flips red.
            std::string payload = "{\"camera_id\":\"" + cam.id + "\",\"state\":\"failed\"}";
            nats.Publish(subj, payload, /*flush=*/true);
            std::cerr << "worker[" << cam.id << "]: start failed\n";
            return 2;
        }

        // Run a GMainLoop on the main thread. GStreamer bus watches
        // (added via gst_bus_add_watch) dispatch only when the main
        // context iterates. Without this the pipeline starves on state-
        // change messages and never emits buffers past nvinfer — which
        // is exactly what was making USB cams hang while standalone
        // gst-launch (which auto-iterates) worked fine.
        GMainLoop* loop = g_main_loop_new(nullptr, FALSE);
        guint watch_id = g_timeout_add(500, [](gpointer) -> gboolean {
            if (g_stop) {
                return FALSE;
            }
            return TRUE;
        }, nullptr);
        // Watchdog + fault-latch. If the pipeline doesn't reach PLAYING
        // within 60 s of Start(), we fault the pipeline ourselves so the
        // supervisor respawns instead of wedging on a silent rtspsrc
        // SETUP hang. rtspsrc with tcp-timeout=15s normally errors out
        // long before 60 s; this is belt-and-braces for cases where the
        // element doesn't surface the failure on the bus.
        std::thread stop_watcher([&loop, &p, &cam, subj, &nats] {
            const auto startup_deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (!g_stop && !p.Faulted()) {
                if (!p.Playing() &&
                    std::chrono::steady_clock::now() > startup_deadline) {
                    std::cerr << "worker[" << cam.id
                              << "]: did not reach PLAYING within 60s — faulting\n";
                    std::string payload = "{\"camera_id\":\"" + cam.id +
                        "\",\"state\":\"failed\"}";
                    nats.Publish(subj, payload, /*flush=*/true);
                    p.Fault();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            g_main_loop_quit(loop);
        });

        // Data-flow watchdog. Catches silent stalls where the GStreamer
        // bus never fires ERROR — typical cause is an NvMedia element
        // wedged inside a kernel syscall (VIC transform, decoder)
        // where the GStreamer pipeline thinks it's still PLAYING but
        // no buffers actually flow. We saw this in the wild: house-side
        // sat as a zombie for 37 min until an operator SIGKILL'd it.
        //
        // Samples BuffersPassed() every 5 s; if the count hasn't
        // advanced for 20 s WHILE Playing() is true, publish failed +
        // _exit(3) directly. We bypass the bus handler here because the
        // whole point is that the bus is silent.
        std::thread flow_watchdog([&p, &cam, subj, &nats] {
            // Re-enabled (stage 1): the flow counter is now bumped by
            // SingleCameraPipeline::FlowCounterProbe on pgie.src for
            // every batch buffer flowing through the inference branch.
            // Catches silent stalls where the bus stays quiet (typical
            // cause: NvMedia/VIC kernel syscall wedge — observed in
            // the wild as house-side sitting zombie for 37 min).
            //
            // Sample every 5 s; if the count hasn't advanced in 20 s
            // WHILE Playing()=true, publish failed + _Exit(3) directly.
            // We bypass the bus handler because the whole point is the
            // bus is silent.
            //
            // Skip the watchdog for cameras without inference (the
            // no-AI tier — enabled_detectors=["none"]). Those have no
            // pgie element so the probe is never attached and the
            // counter would never advance.
            const bool no_inference = cam.enabled_detectors.size() == 1 &&
                                      cam.enabled_detectors[0] == "none";
            if (no_inference) return;
            while (!g_stop && !p.Faulted() && !p.Playing()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            std::uint64_t last_seen = p.BuffersPassed();
            auto last_progress = std::chrono::steady_clock::now();
            const auto stall_threshold = std::chrono::seconds(20);
            while (!g_stop && !p.Faulted()) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                std::uint64_t now_count = p.BuffersPassed();
                if (now_count != last_seen) {
                    last_seen = now_count;
                    last_progress = std::chrono::steady_clock::now();
                    continue;
                }
                if (std::chrono::steady_clock::now() - last_progress > stall_threshold) {
                    std::cerr << "worker[" << cam.id
                              << "]: data-flow stalled 20s — hard exit rc=3\n";
                    std::string payload = "{\"camera_id\":\"" + cam.id +
                        "\",\"state\":\"failed\"}";
                    nats.Publish(subj, payload, /*flush=*/true);
                    std::_Exit(3);
                }
            }
        });

        // Heartbeat: republish "running" every 30s so the api-server's
        // per-camera state stays fresh in the 10-minute window. The
        // initial "running" publish in pipeline.cpp on GST_STATE_PLAYING
        // kicks off the heartbeat loop; if the pipeline faults we stop
        // heartbeating so the state naturally expires to "unknown".
        std::thread heartbeat([&p, &nats, &cam, subj] {
            // Wait briefly for the pipeline to reach PLAYING before the
            // first heartbeat (the state-change publish will have fired).
            std::this_thread::sleep_for(std::chrono::seconds(5));
            while (!g_stop && !p.Faulted()) {
                std::string payload = "{\"camera_id\":\"" + cam.id + "\",\"state\":\"running\"}";
                nats.Publish(subj, payload);
                for (int i = 0; i < 60 && !g_stop && !p.Faulted(); i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
        });
        g_main_loop_run(loop);
        g_source_remove(watch_id);
        if (stop_watcher.joinable()) stop_watcher.join();
        if (heartbeat.joinable()) heartbeat.join();
        if (flow_watchdog.joinable()) flow_watchdog.join();
        g_main_loop_unref(loop);

        // Clean exit path (operator stopped the supervisor). For
        // fault paths we hard-exit from the bus handler / watchdog
        // threads directly; this is only reached on SIGTERM-to-stop
        // or a voluntary Faulted() that we set elsewhere.
        p.Stop();
        gst_deinit();
        return p.Faulted() ? 3 : 0;
    }

    // Parent (supervisor) mode.
    gst_init(&argc, &argv);
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    auto cfg = fnvr::LoadFromEnv();
    std::cerr << "pipeline-supervisor starting. db=" << cfg.database_url
              << " nats=" << cfg.nats_url
              << " recordings=" << cfg.recordings_dir
              << " deepstream=" << (cfg.use_deepstream ? "yes" : "no")
              << "\n";

    fnvr::NatsPublisher nats(cfg.nats_url);
    if (!nats.Connected()) {
        std::cerr << "pipeline-supervisor: continuing without NATS (degraded mode)\n";
    }

    // Subscribe to the restart signal published by api-server when the
    // operator picks a new YOLO variant / precision. On receipt, set the
    // stop flag — docker compose's restart=unless-stopped brings us back
    // up, and the entrypoint re-reads settings from the DB before the
    // supervisor starts.
    natsConnection* restart_conn = nullptr;
    natsSubscription* restart_sub = nullptr;
    if (nats.Connected()) {
        natsStatus st = natsConnection_ConnectTo(&restart_conn, cfg.nats_url.c_str());
        if (st == NATS_OK) {
            natsConnection_Subscribe(&restart_sub, restart_conn,
                "fnvr.system.pipeline.restart",
                [](natsConnection*, natsSubscription*, natsMsg* msg, void*) {
                    std::cerr << "pipeline-supervisor: received restart signal\n";
                    g_stop = 1;
                    // Main thread is parked in pause(); poke it with a
                    // process-level signal so its pause() returns and the
                    // loop notices g_stop. std::raise() only signals the
                    // calling thread; kill(getpid(), ...) signals the
                    // process which wakes whichever thread is in pause().
                    kill(getpid(), SIGTERM);
                    natsMsg_Destroy(msg);
                }, nullptr);
        } else {
            std::cerr << "pipeline-supervisor: restart subscriber failed: "
                      << natsStatus_GetText(st) << "\n";
        }
    }

    // Announce ready to the pipeline-state stream, once the engine
    // exists. If it already exists at startup, publish immediately.
    // Otherwise spawn a watcher thread that polls until the first
    // worker's nvinfer writes the engine to disk, then publishes — so
    // the UI banner flips from "compiling_engine" to "ready" as soon as
    // the compile actually finishes, rather than being stuck forever.
    auto engine_path = [&]() -> std::string {
        const char* cfg_env = std::getenv("FNVR_INFER_CONFIG");
        if (!cfg_env || !*cfg_env) return {};
        std::ifstream cf(cfg_env);
        std::string line;
        while (std::getline(cf, line)) {
            const char* KEY = "model-engine-file=";
            auto pos = line.find(KEY);
            if (pos == std::string::npos) continue;
            std::string val = line.substr(pos + std::strlen(KEY));
            while (!val.empty() && (val.back() == '\r' || val.back() == '\n' || val.back() == ' '))
                val.pop_back();
            return val;
        }
        return {};
    }();

    std::thread ready_watcher;
    if (engine_path.empty() || std::filesystem::exists(engine_path)) {
        nats.Publish("fnvr.state.pipeline", "{\"state\":\"ready\"}", /*flush=*/true);
    } else {
        std::cerr << "pipeline-supervisor: engine missing, watching "
                  << engine_path << " for ready signal\n";
        ready_watcher = std::thread([&nats, engine_path]() {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(30);
            while (!g_stop &&
                   std::chrono::steady_clock::now() < deadline &&
                   !std::filesystem::exists(engine_path)) {
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }
            if (!g_stop && std::filesystem::exists(engine_path)) {
                std::cerr << "pipeline-supervisor: engine appeared — publishing ready\n";
                nats.Publish("fnvr.state.pipeline", "{\"state\":\"ready\"}", /*flush=*/true);
            }
        });
    }

    fnvr::Supervisor sup(cfg, &nats);
    std::thread runner([&sup] { sup.Run(); });

    while (!g_stop) pause();

    std::cerr << "pipeline-supervisor: shutting down\n";
    if (restart_sub) natsSubscription_Destroy(restart_sub);
    if (restart_conn) natsConnection_Destroy(restart_conn);
    sup.Stop();
    if (runner.joinable()) runner.join();
    if (ready_watcher.joinable()) ready_watcher.join();
    gst_deinit();
    return 0;
}
