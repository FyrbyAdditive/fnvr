#pragma once

#include <gst/gst.h>
#include <string>

namespace fnvr {

// Dynamic pad manipulation helpers — runtime add/remove of source
// sub-bins from a PLAYING pipeline whose downstream end is an
// nvstreammux. fnvr today uses gst_parse_launch for static graphs
// only; these helpers are the foundation for the batched-mux
// project's per-source watchdog (a stalled source is unlinked +
// respawned without disturbing siblings) and for runtime camera
// add/remove later.
//
// All functions are blocking — they wait for the GStreamer state
// changes they trigger to complete (or fail) before returning.
// Bounded waits are used so a wedged downstream element doesn't
// hang the caller indefinitely.

// AddedSource tracks the bits we created for a source so the caller
// can pass them back to RemoveSourceFromMux later. Holding raw
// GstElement*/GstPad* — caller is responsible for not letting the
// parent pipeline outlive these.
struct AddedSource {
    GstElement* source_bin;     // the parsed sub-bin we added
    GstPad*     source_src_pad; // its src pad
    GstPad*     mux_sink_pad;   // request pad on the mux
    int         source_id;      // index assigned by mux (sink_N)
};

// AddSourceToMux parses a gst-launch description for a single
// source sub-pipeline, wraps it in a bin (if not already one),
// adds it to the parent pipeline, requests a new sink pad on the
// mux, links them, and rolls the new sub-bin to PLAYING.
//
//   parent        the running pipeline (must already be PLAYING)
//   mux           the nvstreammux element (must be in `parent`)
//   source_desc   gst-launch description, e.g.
//                   "rtspsrc ... ! rtph264depay ! h264parse ! nvv4l2decoder"
//                 The last element's src pad is what gets linked
//                 to the mux. Sub-bin must be self-contained (no
//                 external links).
//   tag           short tag included in the bin's name for log
//                 readability ("camera_id" works fine).
//
// Returns nullptr on any failure (with a log line indicating
// where). Returns a heap-allocated AddedSource on success — caller
// owns it and must pass it to RemoveSourceFromMux to clean up.
AddedSource* AddSourceToMux(GstElement* parent,
                            GstElement* mux,
                            const std::string& source_desc,
                            const std::string& tag);

// RemoveSourceFromMux blocks the source's src pad, sends EOS so
// the mux drains it, unlinks, releases the mux request pad,
// removes the sub-bin from the parent, and rolls the sub-bin to
// NULL.
//
// **Known limitation (stage 2a, 2026-05-03)**: works cleanly for an
// actively-flowing source. For a source that has gone idle (no
// recent buffers — exactly the stall case the watchdog fires on)
// the BLOCK probe doesn't fire and the synchronous-fallback path
// hangs in gst_pad_unlink or gst_element_release_request_pad
// against nvstreammux's stale slot. The DeepStream-bundled
// nvstreammux is unhappy releasing a slot that never produced
// data. Workarounds for stage 2c: either (a) try
// USE_NEW_NVSTREAMMUX=yes which has cleaner runtime semantics,
// or (b) accept group-level respawn on stall instead of in-place
// source removal.
//
// `s` is consumed; caller must not use it after this returns.
// Returns true if every step succeeded; false (with logs) if any
// step failed but the function recovered. False does not mean
// "leaked" — best-effort cleanup runs in all cases.
bool RemoveSourceFromMux(GstElement* parent,
                         GstElement* mux,
                         AddedSource* s);

}  // namespace fnvr
