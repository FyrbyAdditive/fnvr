#include "dynamic_pads.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

namespace fnvr {

namespace {

// State change wait helper. nvstreammux + decoders may take a few
// hundred ms to roll up; cap the wait so a wedged element can't
// hang the caller forever.
constexpr int kStateChangeTimeoutMs = 5000;

bool waitForState(GstElement* el, GstState target) {
    GstState cur = GST_STATE_VOID_PENDING;
    GstState pending = GST_STATE_VOID_PENDING;
    GstStateChangeReturn ret = gst_element_get_state(
        el, &cur, &pending,
        guint64(kStateChangeTimeoutMs) * GST_MSECOND);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "dynamic_pads: state change to "
                  << gst_element_state_get_name(target) << " failed\n";
        return false;
    }
    if (cur != target) {
        std::cerr << "dynamic_pads: state change to "
                  << gst_element_state_get_name(target)
                  << " timed out (current="
                  << gst_element_state_get_name(cur) << ")\n";
        return false;
    }
    return true;
}

// Sync a freshly-added element to its parent's state. Required
// after gst_bin_add — without this the new element sits in NULL
// while the parent is PLAYING and no buffers flow through it.
bool syncToParent(GstElement* el) {
    if (!gst_element_sync_state_with_parent(el)) {
        std::cerr << "dynamic_pads: sync_state_with_parent failed\n";
        return false;
    }
    return waitForState(el, GST_STATE_PLAYING);
}

// Per-removal probe state. Probes fire on a streaming thread (or a
// GStreamer internal thread for IDLE probes); the main thread waits
// on the condvar before doing the synchronous unlink + bin_remove
// sequence. The probe itself just signals — putting bin_remove in
// the probe sometimes raced with GStreamer's internal state
// propagation and produced GST_IS_ELEMENT CRITICALs on later adds.
struct RemovalCtx {
    std::mutex                 mu;
    std::condition_variable    cv;
    bool                       fired = false;
};

GstPadProbeReturn signalAndStayProbe(GstPad* /*pad*/,
                                     GstPadProbeInfo* /*info*/,
                                     gpointer user_data) {
    auto* ctx = static_cast<RemovalCtx*>(user_data);
    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        ctx->fired = true;
    }
    ctx->cv.notify_one();
    // Stay in place — the probe blocks the pad until removed by
    // the caller. The IDLE variant is one-shot anyway.
    return GST_PAD_PROBE_OK;
}

// Try to send EOS to the mux's sink pad and wait briefly for the
// mux to acknowledge by releasing the slot. nvstreammux logs
// "Successfully handled EOS for source_id=N" when it does. We
// don't directly observe that log line here — instead we just
// give the mux 1 s to process the EOS event before proceeding
// to unlink. If the mux is wedged, unlink will still fail, but
// at least we tried the canonical path first.
bool sendEosToMux(GstPad* mux_sink) {
    if (!mux_sink) return false;
    gboolean sent = gst_pad_send_event(mux_sink, gst_event_new_eos());
    if (!sent) {
        std::cerr << "dynamic_pads: gst_pad_send_event(EOS) returned FALSE\n";
        return false;
    }
    // Give the mux a beat to process the event before unlink.
    // 200ms is enough at typical batched-push-timeout=40ms cycles.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return true;
}

// Try to flush the link by sending FLUSH_START + FLUSH_STOP to the
// mux sink pad. This forces the mux to release any stream lock it's
// holding without acknowledging pending data. Works even when EOS
// can't get through because the aggregator is wedged inside its
// poll cycle.
bool sendFlushToMux(GstPad* mux_sink) {
    if (!mux_sink) return false;
    bool start_ok = gst_pad_send_event(mux_sink,
        gst_event_new_flush_start()) == TRUE;
    bool stop_ok  = gst_pad_send_event(mux_sink,
        gst_event_new_flush_stop(TRUE)) == TRUE;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return start_ok && stop_ok;
}

}  // namespace

AddedSource* AddSourceToMux(GstElement* parent,
                            GstElement* mux,
                            const std::string& source_desc,
                            const std::string& tag) {
    if (!parent || !mux) {
        std::cerr << "dynamic_pads: AddSourceToMux: null parent or mux\n";
        return nullptr;
    }

    // Wrap the description in a bin so we have a single handle to
    // remove later. gst_parse_bin_from_description handles the
    // launch parsing + creates an outer GstBin around it whose
    // ghost src pad is the description's last element's src.
    // gst-launch syntax interprets "(...)" after a caps as a bin
    // grouping, which conflicts with caps modifiers like
    // "video/x-raw(memory:NVMM)". The caller must escape the parens
    // as "\(memory:NVMM\)" if they need NVMM caps explicitly. In
    // practice nvv4l2decoder etc. already advertise NVMM on their
    // src pad without an explicit caps filter.
    GError* err = nullptr;
    GstElement* source_bin = gst_parse_bin_from_description(
        source_desc.c_str(), TRUE, &err);
    if (!source_bin) {
        std::cerr << "dynamic_pads: parse_bin_from_description failed: "
                  << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        return nullptr;
    }

    // Name it for log readability.
    std::string name = "src_" + tag;
    gst_element_set_name(source_bin, name.c_str());

    // Add to parent. After this call the bin's refcount belongs to
    // the parent; we don't unref it ourselves.
    if (!gst_bin_add(GST_BIN(parent), source_bin)) {
        std::cerr << "dynamic_pads: gst_bin_add failed\n";
        gst_object_unref(source_bin);
        return nullptr;
    }

    // Request a new sink pad on the mux. Two complications:
    //
    // 1. nvstreammux requires an explicit sink_N name (its
    //    template is sink_%u but its request handler doesn't
    //    auto-pick the next available index — DeepStream expects
    //    the caller to pick).
    // 2. RemoveSourceFromMux deliberately leaks request pads (see
    //    the comment in that function — release_request_pad on
    //    nvstreammux deadlocks). So sink_0 may already exist as a
    //    leaked slot from a previous removal. Calling
    //    request_pad_simple on an existing-but-leaked pad would
    //    trip "Element mux already has a pad named sink_N" and
    //    pthread_assert.
    //
    // We probe each sink_N: first ask whether a pad with that
    // name already exists via gst_element_get_static_pad; if it
    // does, that index is taken (active or leaked) and we move
    // on. Only when the name is fresh do we call request_pad.
    GstPad* mux_sink = nullptr;
    int chosen_source_id = -1;
    for (unsigned i = 0; i < 256 && !mux_sink; i++) {
        std::string n = "sink_" + std::to_string(i);
        GstPad* existing = gst_element_get_static_pad(mux, n.c_str());
        if (existing) {
            gst_object_unref(existing);
            continue;  // taken (either active or leaked)
        }
        mux_sink = gst_element_request_pad_simple(mux, n.c_str());
        if (mux_sink) chosen_source_id = static_cast<int>(i);
    }
    if (!mux_sink) {
        std::cerr << "dynamic_pads: could not request a sink_N pad on mux "
                     "(searched up to sink_255 — too many leaked slots? "
                     "consider restarting the group)\n";
        gst_bin_remove(GST_BIN(parent), source_bin);
        return nullptr;
    }

    // Get the bin's ghost src pad and link.
    GstPad* source_src = gst_element_get_static_pad(source_bin, "src");
    if (!source_src) {
        std::cerr << "dynamic_pads: source bin has no src pad\n";
        gst_element_release_request_pad(mux, mux_sink);
        gst_object_unref(mux_sink);
        gst_bin_remove(GST_BIN(parent), source_bin);
        return nullptr;
    }

    GstPadLinkReturn lr = gst_pad_link(source_src, mux_sink);
    if (lr != GST_PAD_LINK_OK) {
        std::cerr << "dynamic_pads: gst_pad_link returned " << lr << "\n";
        gst_object_unref(source_src);
        gst_element_release_request_pad(mux, mux_sink);
        gst_object_unref(mux_sink);
        gst_bin_remove(GST_BIN(parent), source_bin);
        return nullptr;
    }

    // Sync the new source's state to the parent's PLAYING. Without
    // this the source sits in NULL and produces no buffers.
    if (!syncToParent(source_bin)) {
        std::cerr << "dynamic_pads: source_bin failed to reach PLAYING\n";
        // Caller will see this as no buffer flow and the per-source
        // watchdog will eventually kick the source out. Don't try
        // to unwind now — the PLAYING transition can be slow on
        // RTSP cold-connect; we want the source to keep trying.
    }

    auto* added = new AddedSource;
    added->source_bin     = source_bin;
    added->source_src_pad = source_src;
    added->mux_sink_pad   = mux_sink;
    added->source_id      = chosen_source_id;
    return added;
}

bool RemoveSourceFromMux(GstElement* parent,
                         GstElement* mux,
                         AddedSource* s) {
    if (!s) return true;
    if (!parent || !mux || !s->source_src_pad || !s->mux_sink_pad ||
        !s->source_bin) {
        // Bad inputs — clean up what we can and bail.
        if (s->source_src_pad) {
            gst_object_unref(s->source_src_pad);
            s->source_src_pad = nullptr;
        }
        delete s;
        return false;
    }

    // Layered fallback. Each phase tries to make the mux release
    // its hold on this source so the synchronous unlink (run after
    // the loop) is free of the STREAM_LOCK deadlock. Whichever
    // phase succeeds wins; later ones never run.
    enum class Phase { A_block, B_idle, C_flush, D_giveup };
    Phase reached = Phase::D_giveup;
    gulong probe_id = 0;
    GstPad* probe_pad = nullptr;

    // ---- Phase A: BLOCK probe on source's src pad ----
    // Fences in-flight buffers (if any). If the source was
    // actively flowing, the probe fires within ms and we send
    // EOS to the mux sink pad. The mux logs "Successfully
    // handled EOS for source_id=N" and releases its hold.
    {
        RemovalCtx ctx;
        gulong id = gst_pad_add_probe(
            s->source_src_pad,
            GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
            &signalAndStayProbe, &ctx, nullptr);
        if (id != 0) {
            std::unique_lock<std::mutex> lock(ctx.mu);
            if (ctx.cv.wait_for(lock, std::chrono::seconds(1),
                                [&]{ return ctx.fired; })) {
                std::cerr << "dynamic_pads: phase A — BLOCK probe fired\n";
                lock.unlock();
                // Probe is blocking. Send EOS to the mux sink (UPSTREAM
                // direction relative to the mux — gst_pad_send_event
                // delivers events to the pad's element).
                if (sendEosToMux(s->mux_sink_pad)) {
                    reached = Phase::A_block;
                }
                // Remove the BLOCK probe so subsequent phases or the
                // synchronous cleanup don't fight it.
                gst_pad_remove_probe(s->source_src_pad, id);
                id = 0;
            } else {
                std::cerr << "dynamic_pads: phase A — BLOCK probe did not fire "
                             "in 1s (source idle), removing probe\n";
                gst_pad_remove_probe(s->source_src_pad, id);
                id = 0;
            }
        } else {
            std::cerr << "dynamic_pads: phase A — failed to install BLOCK probe\n";
        }
        probe_id = id;  // 0 if removed
    }

    // ---- Phase B: IDLE probe on the mux's sink pad ----
    // Fires when the aggregator visits this slot, even if no
    // buffer is present. nvstreammux polls every cycle so this
    // typically fires within batched-push-timeout (40ms).
    if (reached == Phase::D_giveup) {
        RemovalCtx ctx;
        gulong id = gst_pad_add_probe(
            s->mux_sink_pad,
            GstPadProbeType(GST_PAD_PROBE_TYPE_IDLE),
            &signalAndStayProbe, &ctx, nullptr);
        if (id != 0) {
            std::unique_lock<std::mutex> lock(ctx.mu);
            if (ctx.cv.wait_for(lock, std::chrono::seconds(2),
                                [&]{ return ctx.fired; })) {
                std::cerr << "dynamic_pads: phase B — IDLE probe fired on mux sink\n";
                lock.unlock();
                if (sendEosToMux(s->mux_sink_pad)) {
                    reached = Phase::B_idle;
                }
                gst_pad_remove_probe(s->mux_sink_pad, id);
            } else {
                std::cerr << "dynamic_pads: phase B — IDLE probe did not fire "
                             "in 2s (mux aggregator wedged?), removing\n";
                gst_pad_remove_probe(s->mux_sink_pad, id);
            }
        } else {
            std::cerr << "dynamic_pads: phase B — failed to install IDLE probe\n";
        }
    }

    // ---- Phase C: explicit flush event ----
    // Forces the mux to release any stream lock it's holding,
    // even if the aggregator is wedged. This is a reset that
    // discards pending data, but no other data is flowing through
    // this pad anyway.
    if (reached == Phase::D_giveup) {
        std::cerr << "dynamic_pads: phase C — flushing mux sink\n";
        if (sendFlushToMux(s->mux_sink_pad)) {
            reached = Phase::C_flush;
        } else {
            std::cerr << "dynamic_pads: phase C — flush failed\n";
        }
    }

    // ---- Phase D: give up gracefully ----
    if (reached == Phase::D_giveup) {
        std::cerr << "dynamic_pads: phase D — could not coax mux to release; "
                     "best-effort cleanup follows\n";
    }

    // Synchronous cleanup. Even if every phase failed, we still
    // try unlink + bin_remove; phases A-C are about giving us the
    // best chance of unlink succeeding without deadlock.
    bool ok = (reached != Phase::D_giveup);
    (void)probe_pad;
    (void)probe_id;

    if (!gst_pad_unlink(s->source_src_pad, s->mux_sink_pad)) {
        std::cerr << "dynamic_pads: gst_pad_unlink failed\n";
        ok = false;
    }

    // Set the source bin to NULL before further cleanup. Some
    // elements emit one last buffer on state change; want that
    // dropped against an unlinked pair, not active.
    gst_element_set_state(s->source_bin, GST_STATE_NULL);

    // We deliberately do NOT call gst_element_release_request_pad
    // on nvstreammux. The bundled DeepStream 7.1 nvstreammux
    // implementation blocks indefinitely on release for any slot
    // (even slots whose source is fully drained — verified with
    // EOS-on-sink-pad acknowledged via "Successfully handled EOS
    // for source_id=N" before the release call). And calling
    // release with a timeout + thread-detach corrupts internal
    // state (next request returns the same numeric pad name and
    // crashes with "Element mux already has a pad named sink_N").
    //
    // Instead: leak the request pad. AddSourceToMux's pad-search
    // loop probes upward starting at sink_0, so it'll find the
    // next genuinely-free slot regardless. Over the life of one
    // pipeline process this leaks a few KB per source removal.
    // For a Group running indefinitely with occasional source
    // restarts, this is the price of avoiding the release_request_pad
    // landmine. When the parent pipeline goes to NULL on group
    // teardown, all internal state cleans up.
    //
    // Drop our ref to the mux_sink_pad — the mux still holds its
    // own internal ref so the pad object survives.
    gst_object_unref(s->mux_sink_pad);
    s->mux_sink_pad = nullptr;

    if (!gst_bin_remove(GST_BIN(parent), s->source_bin)) {
        std::cerr << "dynamic_pads: gst_bin_remove failed\n";
        ok = false;
    }

    if (s->source_src_pad) {
        gst_object_unref(s->source_src_pad);
        s->source_src_pad = nullptr;
    }
    delete s;
    return ok;
}

}  // namespace fnvr
