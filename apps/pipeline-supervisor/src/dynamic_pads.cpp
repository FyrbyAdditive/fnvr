#include "dynamic_pads.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <condition_variable>

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

// Per-removal state for the BLOCK-DOWNSTREAM probe. The probe fires
// on the streaming thread; we use a condvar to wake the caller on
// the main thread when the block is established and we've finished
// unlink + release. Non-RAII because GStreamer probes have specific
// removal semantics.
struct RemovalCtx {
    std::mutex                 mu;
    std::condition_variable    cv;
    bool                       done = false;
    GstElement*                source_bin   = nullptr;
    GstPad*                    source_src   = nullptr;
    GstPad*                    mux_sink     = nullptr;
    GstElement*                mux          = nullptr;
    GstElement*                parent       = nullptr;
    bool                       ok           = false;
};

GstPadProbeReturn removeBlockProbe(GstPad* pad,
                                   GstPadProbeInfo* /*info*/,
                                   gpointer user_data) {
    auto* ctx = static_cast<RemovalCtx*>(user_data);

    // We're blocking the source's src pad — no more buffers will
    // reach the mux from this source. The probe's job is minimal:
    // just signal the main thread to do the actual unlink + remove
    // safely under its own thread context. Trying to do bin_remove
    // from a streaming thread (the probe's context) sometimes
    // triggers the "GST_IS_ELEMENT" CRITICAL because GStreamer is
    // mid-flight on internal state.
    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        ctx->ok = true;
        ctx->done = true;
    }
    ctx->cv.notify_one();
    // Don't return REMOVE here — leave the BLOCK probe in place
    // until the main thread has finished its synchronous unlink +
    // remove sequence. The pad stays blocked the whole time.
    return GST_PAD_PROBE_OK;
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

    // Request a new sink pad on the mux. nvstreammux requires an
    // explicit sink_N name (its template is sink_%u but its
    // request handler doesn't auto-pick the next available index
    // the way the generic GstElement handler does — DeepStream
    // expects the caller to pick the index). We probe upward
    // starting at 0 until one succeeds; numbers below the first
    // success are already in use.
    GstPad* mux_sink = nullptr;
    for (unsigned i = 0; i < 64 && !mux_sink; i++) {
        std::string n = "sink_" + std::to_string(i);
        mux_sink = gst_element_request_pad_simple(mux, n.c_str());
    }
    if (!mux_sink) {
        std::cerr << "dynamic_pads: could not request a sink_N pad on mux\n";
        gst_bin_remove(GST_BIN(parent), source_bin);
        return nullptr;
    }
    // nvstreammux may ignore the explicit "sink_<i>" name and return
    // the next-free slot; read the actual pad name to learn the
    // assigned source_id.
    int chosen_source_id = -1;
    {
        gchar* pn = gst_pad_get_name(mux_sink);
        if (pn) {
            sscanf(pn, "sink_%d", &chosen_source_id);
            g_free(pn);
        }
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
    bool ok = true;

    // The remove path uses a BLOCK-DOWNSTREAM probe to fence in-
    // flight buffers, then does the actual unlink + bin_remove on
    // THIS thread (the caller's thread, typically the main loop).
    // Doing bin_remove from inside the probe (streaming thread)
    // sometimes raced with GStreamer's internal state propagation
    // and produced GST_IS_ELEMENT CRITICALs on the next add.
    bool used_block_probe = false;
    if (parent && mux && s->source_src_pad && s->mux_sink_pad &&
        s->source_bin) {
        RemovalCtx ctx;
        ctx.parent      = parent;
        ctx.mux         = mux;
        ctx.source_bin  = s->source_bin;
        ctx.source_src  = s->source_src_pad;
        ctx.mux_sink    = s->mux_sink_pad;

        gulong probe_id = gst_pad_add_probe(
            s->source_src_pad,
            GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
            &removeBlockProbe, &ctx, nullptr);

        if (probe_id == 0) {
            std::cerr << "dynamic_pads: gst_pad_add_probe(BLOCK) failed\n";
            ok = false;
        } else {
            // Wait briefly for the probe to fire (it just signals
            // and stays in place). 2s is plenty under load; if it
            // doesn't fire (source already silent), we proceed
            // anyway — there's no buffer in flight to race with.
            std::unique_lock<std::mutex> lock(ctx.mu);
            if (ctx.cv.wait_for(lock, std::chrono::seconds(2),
                                [&]{ return ctx.done; })) {
                used_block_probe = true;
            } else {
                std::cerr << "dynamic_pads: BLOCK probe didn't fire in 2s; "
                             "proceeding without (source likely idle)\n";
            }
        }

        // Synchronous cleanup. Probe (if any) is still blocking
        // the src pad — remove it FIRST so EOS can travel
        // downstream. Pushing EOS through a blocked pad would
        // deadlock the caller.
        if (probe_id != 0) {
            gst_pad_remove_probe(s->source_src_pad, probe_id);
        }
        // EOS only when the source was actually flowing (probe
        // fired). On an idle source nvstreammux gets confused by
        // EOS-without-prior-buffers and the subsequent
        // release_request_pad hangs. The unlink + bin_remove path
        // below handles the idle case cleanly without EOS.
        if (used_block_probe) {
            gst_pad_push_event(s->source_src_pad, gst_event_new_eos());
        }
        if (!gst_pad_unlink(s->source_src_pad, s->mux_sink_pad)) {
            std::cerr << "dynamic_pads: gst_pad_unlink failed\n";
            ok = false;
        }
        // Roll the source bin to NULL BEFORE releasing the mux
        // request pad — it gives the bin a clean shutdown without
        // nvstreammux waiting on the dead source.
        gst_element_set_state(s->source_bin, GST_STATE_NULL);
        gst_element_release_request_pad(mux, s->mux_sink_pad);
        gst_object_unref(s->mux_sink_pad);
        s->mux_sink_pad = nullptr;
        if (!gst_bin_remove(GST_BIN(parent), s->source_bin)) {
            std::cerr << "dynamic_pads: gst_bin_remove failed\n";
            ok = false;
        }
    }

    // gst_element_get_static_pad gave us a separate ref in
    // AddSourceToMux; release it now. Safe even after the
    // bin_remove inside the probe — pad object stays alive while
    // anyone holds a ref to it.
    if (s->source_src_pad) {
        gst_object_unref(s->source_src_pad);
        s->source_src_pad = nullptr;
    }
    delete s;
    return ok;
}

}  // namespace fnvr
