#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <gst/gst.h>

#include "config.h"
#include "nats_publisher.h"

namespace fnvr {

// Set the grace window (seconds) used by the bus-error hard-exit path
// to suppress its "failed" NATS publish. Called by the worker child
// process at startup with the value from settings.pipeline.startup_grace_sec.
// Without this, a slow-to-connect source (MediaMTX-proxied, cold-boot
// TLS, etc.) fires `failed` in the UI on its first warmup exit even
// though the supervisor is about to respawn and succeed.
void SetWorkerStartupGraceSec(int sec);

// SingleCameraPipeline is the M2 shape: one camera in, push to MediaMTX
// + (if DeepStream is available) detect. MediaMTX handles recording
// (its built-in fMP4 recorder) and live WebRTC. One pipeline per camera
// keeps the M2 blast-radius tiny; M3 replaces this with batched
// nvstreammux across a group of cameras.
class SingleCameraPipeline {
public:
    SingleCameraPipeline(CameraConfig cam, std::string recordings_dir,
                         std::string infer_config, bool use_deepstream,
                         bool use_anpr, bool use_face_id,
                         NatsPublisher* nats);
    ~SingleCameraPipeline();

    bool Start();
    void Stop();

private:
    GstElement* BuildPipeline();
    static gboolean BusHandler(GstBus*, GstMessage*, gpointer user_data);
    // Pad-probe callback that bumps buffersPassed_ on every batch
    // buffer flowing past pgie.src. user_data points at the
    // SingleCameraPipeline so the probe can update the atomic.
    // The data-flow watchdog samples this counter every 5s.
    static GstPadProbeReturn FlowCounterProbe(GstPad*, GstPadProbeInfo*, gpointer);

    CameraConfig     cam_;
    std::string      recordings_dir_;
    std::string      infer_config_;
    bool             use_deepstream_;
    bool             use_anpr_ = false;
    bool             use_face_id_ = false;
    NatsPublisher*   nats_ = nullptr;
    GstElement*      pipeline_ = nullptr;
    guint            bus_watch_id_ = 0;
    std::atomic<bool> faulted_{false};
    // Flips true the first time the bus reports GST_STATE_PLAYING.
    // Used by main.cpp's watchdog to detect "pipeline never rolled"
    // — e.g. rtspsrc hung between TCP connect and SETUP reply.
    std::atomic<bool> playing_{false};
    // Reserved for a re-introduced flow watchdog (the previous one
    // monitored a recparse element that no longer exists; tracked
    // for revival on a still-flowing pad such as tracker.src or the
    // rtspclientsink request pad).
    std::atomic<std::uint64_t> buffersPassed_{0};

    // Target dimensions for the inference branch. Zero until
    // BuildPipeline has probed the source; caps for nvstreammux key
    // off these so aspect is preserved end-to-end.
    int rec_width_ = 0;
    int rec_height_ = 0;
    // Codec carried on the gst tee — "h264" or "h265". Drives the
    // rtspclientsink branch's parser choice so MediaMTX receives a
    // clean elementary stream.
    std::string pipeline_codec_ = "h264";

public:
    bool Faulted() const { return faulted_.load(); }
    bool Playing() const { return playing_.load(); }
    void Fault() { faulted_.store(true); }
    // Sampled by the (currently disabled) flow watchdog.
    std::uint64_t BuffersPassed() const { return buffersPassed_.load(); }
};

}  // namespace fnvr
