#include "preview_probe.h"
#include "face_crop_jpeg.h"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <sys/stat.h>

#if __has_include(<nvbufsurface.h>)
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#define FNVR_HAS_DEEPSTREAM 1
#else
#define FNVR_HAS_DEEPSTREAM 0
#endif

namespace fnvr {

namespace {
// Output thumbnail dimensions. 480×270 is 16:9; non-16:9 sources are
// scaled-fit (no letterbox) since this image is only used as a tile
// fallback / snapshot — no bbox alignment needs preserving.
constexpr int kOutW = 480;
constexpr int kOutH = 270;
constexpr int kQuality = 75;

// Ring depth — matches the historical multifilesink max-files=4
// contract. The api-server snapshot reader globs <id>.*.jpg and picks
// the second-newest mtime (snapshot.go:99-141), so the ring needs at
// least 2 entries to never serve a half-written file. We keep 4 for
// belt-and-braces; with our atomic tmp+rename writes any of the 4 is
// always a complete JPEG.
constexpr int kRingSize = 4;

// 1 fps cadence. The first eligible frame fires immediately so the
// snapshot endpoint has something to serve as soon as the pipeline
// reaches PLAYING.
constexpr auto kInterval = std::chrono::milliseconds(1000);
}  // namespace

struct PreviewProbeCtx {
    // Indexed by NvDsFrameMeta.source_id (the nvstreammux input pad
    // index). Single entry today — one camera per worker — but the
    // map exists so a future batched-mux group can mount N cameras
    // behind one probe and emit one JPEG ring per source.
    std::vector<std::string> sources;
    std::string live_dir;
    int         source_width  = 0;
    int         source_height = 0;

    std::mutex                            ctx_mu;        // serialises the slow path
    std::chrono::steady_clock::time_point last_emit{};   // 0 = never
    int                                   ring_idx = 0;  // 0..kRingSize-1

#if FNVR_HAS_DEEPSTREAM
    NvBufSurface* dst_surf = nullptr;
#endif

    const std::string& cameraIdFor(unsigned source_id) const {
        if (source_id < sources.size()) return sources[source_id];
        static const std::string empty;
        return sources.empty() ? empty : sources[0];
    }
    const std::string& primaryCameraId() const {
        static const std::string empty;
        return sources.empty() ? empty : sources[0];
    }
};

PreviewProbeCtx* preview_probe_ctx_new(std::string camera_id,
                                       std::string live_dir,
                                       int src_w, int src_h) {
    auto* ctx = new PreviewProbeCtx;
    ctx->sources       = {std::move(camera_id)};
    ctx->live_dir      = std::move(live_dir);
    ctx->source_width  = src_w  > 0 ? src_w  : 1920;
    ctx->source_height = src_h > 0 ? src_h : 1080;
    return ctx;
}

void preview_probe_ctx_free(PreviewProbeCtx* ctx) {
    if (!ctx) return;
#if FNVR_HAS_DEEPSTREAM
    if (ctx->dst_surf) {
        NvBufSurfaceDestroy(ctx->dst_surf);
        ctx->dst_surf = nullptr;
    }
#endif
    delete ctx;
}

#if FNVR_HAS_DEEPSTREAM

namespace {

// Lazily allocate the reusable RGBA destination at the gpuId of the
// first observed input surface. Same pattern as hailo_probe's
// ensureDstSurface — the surface is reused across every encode, so the
// cost is paid once per worker.
bool ensureDstSurface(PreviewProbeCtx* ctx, NvBufSurface* in_surf) {
    if (ctx->dst_surf) return true;

    NvBufSurfaceAllocateParams ap{};
    ap.params.gpuId        = in_surf->gpuId;
    ap.params.width        = static_cast<unsigned>(kOutW);
    ap.params.height       = static_cast<unsigned>(kOutH);
    ap.params.size         = 0;
    ap.params.isContiguous = true;
    ap.params.colorFormat  = NVBUF_COLOR_FORMAT_RGBA;
    ap.params.layout       = NVBUF_LAYOUT_PITCH;
    ap.params.memType      = NVBUF_MEM_SURFACE_ARRAY;
    ap.memtag              = NvBufSurfaceTag_VIDEO_CONVERT;
    if (NvBufSurfaceAllocate(&ctx->dst_surf, 1, &ap) != 0 || !ctx->dst_surf) {
        std::cerr << "preview_probe[" << ctx->primaryCameraId()
                  << "]: NvBufSurfaceAllocate failed\n";
        ctx->dst_surf = nullptr;
        return false;
    }
    ctx->dst_surf->numFilled = 1;
    return true;
}

// Encode the dst_surf to the next ring slot using tmp + rename so the
// api-server reader (which sorts by mtime) never sees a partial file.
bool writeRingEntry(PreviewProbeCtx* ctx) {
    NvBufSurfaceParams& dp = ctx->dst_surf->surfaceList[0];
    if (NvBufSurfaceMap(ctx->dst_surf, 0, 0, NVBUF_MAP_READ) != 0) {
        std::cerr << "preview_probe[" << ctx->primaryCameraId()
                  << "]: NvBufSurfaceMap failed\n";
        return false;
    }
    NvBufSurfaceSyncForCpu(ctx->dst_surf, 0, 0);

    const std::uint8_t* rgba =
        static_cast<const std::uint8_t*>(dp.mappedAddr.addr[0]);
    const int stride = static_cast<int>(dp.pitch);

    // Build paths: live_dir/<id>.<n>.jpg.tmp → live_dir/<id>.<n>.jpg
    char tmp_path[512];
    char out_path[512];
    std::snprintf(tmp_path, sizeof(tmp_path), "%s/%s.%d.jpg.tmp",
                  ctx->live_dir.c_str(), ctx->primaryCameraId().c_str(), ctx->ring_idx);
    std::snprintf(out_path, sizeof(out_path), "%s/%s.%d.jpg",
                  ctx->live_dir.c_str(), ctx->primaryCameraId().c_str(), ctx->ring_idx);

    bool ok = encodeJpegRGBA(rgba, stride, 0, 0, kOutW, kOutH, kQuality,
                             tmp_path);
    NvBufSurfaceUnMap(ctx->dst_surf, 0, 0);

    if (!ok) {
        std::cerr << "preview_probe[" << ctx->primaryCameraId()
                  << "]: encodeJpegRGBA failed\n";
        std::remove(tmp_path);
        return false;
    }

    if (std::rename(tmp_path, out_path) != 0) {
        std::cerr << "preview_probe[" << ctx->primaryCameraId()
                  << "]: rename(" << tmp_path << " -> " << out_path
                  << ") failed: " << std::strerror(errno) << "\n";
        std::remove(tmp_path);
        return false;
    }

    ctx->ring_idx = (ctx->ring_idx + 1) % kRingSize;
    return true;
}

}  // namespace

#endif  // FNVR_HAS_DEEPSTREAM

GstPadProbeReturn PreviewSnapshotProbe(GstPad*, GstPadProbeInfo* info,
                                       gpointer user) {
    auto* ctx = static_cast<PreviewProbeCtx*>(user);
    if (!ctx) return GST_PAD_PROBE_OK;

    // Rate-limit BEFORE touching the buffer so the hot path is two
    // atomic loads and a compare. Branch is unpredictable at startup
    // (last_emit == 0) but cheap at steady state.
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(ctx->ctx_mu);
        if (ctx->last_emit.time_since_epoch().count() != 0 &&
            now - ctx->last_emit < kInterval) {
            return GST_PAD_PROBE_OK;
        }
        // Optimistically advance last_emit BEFORE doing the work — if
        // the work fails, we still wait the full interval before the
        // next attempt rather than retrying every frame.
        ctx->last_emit = now;
    }

#if !FNVR_HAS_DEEPSTREAM
    return GST_PAD_PROBE_OK;
#else
    GstBuffer* buf = gst_pad_probe_info_get_buffer(info);
    if (!buf) return GST_PAD_PROBE_OK;

    GstMapInfo map{};
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) return GST_PAD_PROBE_OK;
    auto* in_surf = reinterpret_cast<NvBufSurface*>(map.data);
    if (!in_surf) {
        gst_buffer_unmap(buf, &map);
        return GST_PAD_PROBE_OK;
    }

    std::lock_guard<std::mutex> lock(ctx->ctx_mu);

    if (!ensureDstSurface(ctx, in_surf)) {
        gst_buffer_unmap(buf, &map);
        return GST_PAD_PROBE_OK;
    }

    // First populated frame in the batch is the camera's frame.
    // nvstreammux with batch-size=1 always sets numFilled to 1.
    if (in_surf->numFilled == 0) {
        gst_buffer_unmap(buf, &map);
        return GST_PAD_PROBE_OK;
    }
    NvBufSurfaceParams& in_p = in_surf->surfaceList[0];

    NvBufSurfTransformRect src_rect {
        0, 0, guint(in_p.width), guint(in_p.height)
    };
    NvBufSurfTransformRect dst_rect {
        0, 0, guint(kOutW), guint(kOutH)
    };
    NvBufSurfTransformParams tp{};
    tp.src_rect = &src_rect;
    tp.dst_rect = &dst_rect;
    tp.transform_flag =
        NVBUFSURF_TRANSFORM_FILTER |
        NVBUFSURF_TRANSFORM_CROP_SRC |
        NVBUFSURF_TRANSFORM_CROP_DST;
    tp.transform_filter = NvBufSurfTransformInter_Default;

    NvBufSurface tmp_in = *in_surf;
    tmp_in.surfaceList = &in_surf->surfaceList[0];
    tmp_in.numFilled   = 1;
    tmp_in.batchSize   = 1;

    if (NvBufSurfTransform(&tmp_in, ctx->dst_surf, &tp)
        != NvBufSurfTransformError_Success) {
        std::cerr << "preview_probe[" << ctx->primaryCameraId()
                  << "]: NvBufSurfTransform failed\n";
        gst_buffer_unmap(buf, &map);
        return GST_PAD_PROBE_OK;
    }

    bool ok = writeRingEntry(ctx);
    gst_buffer_unmap(buf, &map);

    // One-shot liveness log on first successful write so the operator
    // can confirm the probe is engaged; silent thereafter.
    static std::atomic<int> first_ok{0};
    if (ok && first_ok.exchange(1) == 0) {
        std::cerr << "preview_probe[" << ctx->primaryCameraId()
                  << "]: first JPEG ring entry written\n";
    }

    return GST_PAD_PROBE_OK;
#endif
}

}  // namespace fnvr
