#include "pipeline.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "face_crop_jpeg.h"
#include "hailo_probe.h"
#include "preview_probe.h"
#include "object_phash.h"
#include "rtsp_probe.h"

// DeepStream metadata. Only include when building for Jetson — these headers
// come from the deepstream-l4t base image.
#if __has_include(<gstnvdsmeta.h>)
#  define FNVR_HAS_DEEPSTREAM 1
#  include <gstnvdsmeta.h>
#  include <nvdsmeta.h>
// NvDsInferTensorMeta + NVDSINFER_TENSOR_OUTPUT_META for ArcFace
// output-tensor-meta extraction.
#  include <gstnvdsinfer.h>
// NvBufSurface + NvBufSurfTransform for in-probe face cropping:
// we VIC-convert the batched NVMM buffer into a small pitch-linear
// RGBA surface that libjpeg can read synchronously.
#  include <nvbufsurface.h>
#  include <nvbufsurftransform.h>
#endif

namespace fnvr {

namespace fs = std::filesystem;

SingleCameraPipeline::SingleCameraPipeline(CameraConfig cam, std::string recordings_dir,
                                           std::string infer_config, bool use_deepstream,
                                           bool use_anpr, bool use_face_id,
                                           NatsPublisher* nats)
    : cam_(std::move(cam)),
      recordings_dir_(std::move(recordings_dir)),
      infer_config_(std::move(infer_config)),
      use_deepstream_(use_deepstream),
      use_anpr_(use_anpr),
      use_face_id_(use_face_id),
      nats_(nats) {}

SingleCameraPipeline::~SingleCameraPipeline() { Stop(); }

#if FNVR_HAS_DEEPSTREAM
namespace {

// Generate a short random ID (no UUID lib dependency).
std::string short_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream os;
    os << std::hex << rng();
    return os.str();
}

struct ProbeCtx {
    // Indexed by NvDsFrameMeta.source_id (the nvstreammux input pad
    // index). Single entry today — one camera per worker — but the
    // map exists so a future batched-mux group can mount N cameras
    // behind one probe without re-plumbing every call site.
    // primaryCameraId() returns sources[0] for the existing log /
    // bus-publish paths that summarise across the worker.
    std::vector<std::string> sources;
    NatsPublisher*           nats;
    // Snapshot of the effective mute set, resolved at worker startup.
    // The probe short-circuits on empty so unmuted cameras pay zero.
    std::set<std::string> muted_classes;
    // Output directory for per-detection face JPEGs. Written as
    //   {thumbs_dir}/{event_id}.jpg
    // where event_id is the detection's short hex id (same one that
    // goes into the NATS payload and becomes the detection's event_id
    // in PG, so event-processor can rename to {pg_id}.jpg on INSERT).
    // Empty when face_id is off — saveFaceCrop short-circuits.
    std::string           thumbs_dir;
    // Output directory for per-detection OBJECT crop JPEGs (smaller;
    // 128x128 @ quality 75). Used by the Flags page to show what was
    // flagged. Empty = don't write (probe still computes phash).
    std::string           thumbs_dir_objects;
    // Heartbeat counters. Logged every ~250 buffers so a silent probe
    // (attached but seeing no objects, or no batch meta) is visible
    // in the pipeline log without needing a debugger attached.
    std::uint64_t         hb_buffers     = 0;
    std::uint64_t         hb_batch_null  = 0;
    std::uint64_t         hb_frames_obj  = 0;
    std::uint64_t         hb_objects     = 0;
    std::uint64_t         hb_published   = 0;

    // Per-frame source_id → camera_id lookup. Falls back to
    // sources[0] if the index is out of range (defensive — if
    // nvstreammux ever sends an unexpected source_id we'd rather
    // mislabel than crash).
    const std::string& cameraIdFor(unsigned source_id) const {
        if (source_id < sources.size()) return sources[source_id];
        static const std::string empty;
        return sources.empty() ? empty : sources[0];
    }
    // For log lines + bus publishes that summarise across the
    // worker (heartbeats, etc.). Today there's only ever one source.
    const std::string& primaryCameraId() const {
        static const std::string empty;
        return sources.empty() ? empty : sources[0];
    }
};

// JSON-escape minimal — only the fields we emit. Labels are small ASCII, IDs
// are hex. Good enough for M2; swap for a real encoder when we move to binary
// protobuf on the bus.
std::string json_escape(std::string_view s) {
    std::string out; out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;
        }
    }
    return out;
}

// LPDNet is attached as gie-unique-id=2 in lpdnet.txt. Any obj_meta
// with this component id is a plate crop, not a primary-detector
// object. Pgie (YOLO26) = 1; LPRNet (classifier) only updates
// classifier_meta on the plate's obj_meta — it doesn't add new objs.
constexpr unsigned LPDNET_GIE_ID  = 2;
// SCRFD detector is gie-unique-id=4 in scrfd.txt (arcface is 5).
// Face obj_meta carry unique_component_id = SCRFD_GIE_ID.
constexpr unsigned SCRFD_GIE_ID   = 4;
// ArcFace's 512-d output lands on the face obj_meta's user meta as
// NVDSINFER_TENSOR_OUTPUT_META.
constexpr int      ARCFACE_DIM    = 512;
// Minimum face bbox in pixels — below this, ArcFace output is noise.
constexpr int      MIN_FACE_PX    = 30;

// base64_encode is a tiny stdlib-free encoder for the 2048-byte
// embedding blob (512 × float32). No padding variant because consumers
// always receive a fixed length.
std::string base64_encode(const void* data, size_t n) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const uint8_t* b = static_cast<const uint8_t*>(data);
    std::string out;
    out.reserve(((n + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        uint32_t v = (uint32_t(b[i]) << 16) | (uint32_t(b[i + 1]) << 8) | b[i + 2];
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        out.push_back(tbl[(v >> 6) & 0x3F]);
        out.push_back(tbl[v & 0x3F]);
    }
    if (i < n) {
        uint32_t v = uint32_t(b[i]) << 16;
        if (i + 1 < n) v |= uint32_t(b[i + 1]) << 8;
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        out.push_back(i + 1 < n ? tbl[(v >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}

// extractFaceEmbedding finds the ArcFace 512-d output on the face's
// user-meta list and base64-encodes it. Returns empty string if
// ArcFace hasn't run on this object (e.g., face_id disabled or the
// bbox was too small for the SGIE to bother). The output tensor has
// shape [1,512] — we copy the 512 float32s directly as little-endian
// IEEE754 bytes, matching the decoder in api-server/server/faces.go.
std::string extractFaceEmbedding(NvDsObjectMeta* obj) {
    for (NvDsMetaList* ul = obj->obj_user_meta_list; ul; ul = ul->next) {
        auto* um = static_cast<NvDsUserMeta*>(ul->data);
        if (!um) continue;
        if (um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;
        auto* tm = static_cast<NvDsInferTensorMeta*>(um->user_meta_data);
        if (!tm || tm->num_output_layers == 0) continue;
        // For output-tensor-meta=1 SGIEs, the host copy lives on the
        // meta's own out_buf_ptrs_host array — layers_info[i].buffer
        // stays null in this mode. Ref: deepstream-infer-tensor-meta-test.
        void* host_buf = tm->out_buf_ptrs_host ? tm->out_buf_ptrs_host[0] : nullptr;
        if (!host_buf) continue;
        auto& li = tm->output_layers_info[0];
        unsigned total = 1;
        for (unsigned d = 0; d < li.inferDims.numDims; d++) {
            total *= li.inferDims.d[d];
        }
        if (total != ARCFACE_DIM) continue;
        return base64_encode(host_buf, size_t(ARCFACE_DIM) * sizeof(float));
    }
    return {};
}

// extractPlateText pulls the plate string from the obj_meta's
// classifier_meta_list, as populated by the LPRNet CTC parser
// (NvDsInferParseCustomNVPlate in libnvds_infercustomparser_tao.so).
// Returns empty string if no classifier meta is attached (low-
// confidence OCR skip / chain not run).
std::string extractPlateText(NvDsObjectMeta* obj) {
    for (NvDsMetaList* cl = obj->classifier_meta_list; cl; cl = cl->next) {
        auto* cmeta = static_cast<NvDsClassifierMeta*>(cl->data);
        if (!cmeta) continue;
        for (NvDsMetaList* ll = cmeta->label_info_list; ll; ll = ll->next) {
            auto* label = static_cast<NvDsLabelInfo*>(ll->data);
            if (label && label->result_label[0]) {
                return std::string(label->result_label);
            }
        }
    }
    return {};
}

// parentVehicleClass reads the upstream vehicle's label (car, truck,
// bus, motorcycle) off a plate's parent obj_meta. Useful context in
// the published detection so the UI / rules engine can show "car
// AB12CDE" without a follow-up lookup.
std::string parentVehicleClass(NvDsObjectMeta* obj) {
    if (obj->parent && obj->parent->obj_label[0]) {
        return std::string(obj->parent->obj_label);
    }
    return {};
}

// saveFaceCrop extracts a crop of the current probe buffer's frame at
// the given normalised bbox, converts it to a small RGBA host buffer
// via VIC-accelerated NvBufSurfTransform (handles NV12 block-linear
// → RGBA pitch-linear + resize in one pass), JPEG-encodes and writes.
// Zero PTS drift — the buffer pixels are literally the ones the face
// detection was produced from. short_id becomes {thumbs_dir}/{id}.jpg;
// event-processor renames to {pg_id}.jpg after INSERT.
//
// 256x256 output — a cheap, face-sized canvas. Bbox aspect usually
// fits a square crop with the 1.4x padding we apply below; the tile
// UI is square anyway.
constexpr int CROP_OUT_W = 256;
constexpr int CROP_OUT_H = 256;

void saveFaceCrop(const ProbeCtx& ctx, GstBuffer* gst_buf,
                  NvDsFrameMeta* frame,
                  float nx, float ny, float nw, float nh,
                  const std::string& short_id) {
    if (ctx.thumbs_dir.empty() || !gst_buf || !frame) return;

    // Expand bbox 1.4x around centre, clip to [0,1].
    float pad = 1.4f;
    float cx = nx + nw / 2.0f;
    float cy = ny + nh / 2.0f;
    float pw = nw * pad;
    float ph = nh * pad;
    float x0 = cx - pw / 2.0f;
    float y0 = cy - ph / 2.0f;
    float x1 = cx + pw / 2.0f;
    float y1 = cy + ph / 2.0f;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > 1) x1 = 1;
    if (y1 > 1) y1 = 1;

    // Map the input surface. The NvBufSurface* is embedded in the
    // gst buffer's map.data.
    GstMapInfo map{};
    if (!gst_buffer_map(gst_buf, &map, GST_MAP_READ)) return;
    auto* in_surf = reinterpret_cast<NvBufSurface*>(map.data);
    if (!in_surf || frame->batch_id >= in_surf->numFilled) {
        gst_buffer_unmap(gst_buf, &map);
        return;
    }
    NvBufSurfaceParams& in_p = in_surf->surfaceList[frame->batch_id];
    const int W = int(in_p.width);
    const int H = int(in_p.height);

    int px = int(x0 * W);
    int py = int(y0 * H);
    int pw_px = int((x1 - x0) * W);
    int ph_px = int((y1 - y0) * H);
    if (pw_px <= 1 || ph_px <= 1) {
        gst_buffer_unmap(gst_buf, &map);
        return;
    }

    // Create a 1-frame pitch-linear RGBA destination we CAN CPU-map.
    // Jetson-correct: memType=NVBUF_MEM_SURFACE_ARRAY + PITCH layout.
    NvBufSurfaceAllocateParams ap{};
    ap.params.gpuId        = in_surf->gpuId;
    ap.params.width        = guint(CROP_OUT_W);
    ap.params.height       = guint(CROP_OUT_H);
    ap.params.size         = 0;
    ap.params.isContiguous = true;
    ap.params.colorFormat  = NVBUF_COLOR_FORMAT_RGBA;
    ap.params.layout       = NVBUF_LAYOUT_PITCH;
    ap.params.memType      = NVBUF_MEM_SURFACE_ARRAY;
    ap.memtag              = NvBufSurfaceTag_VIDEO_CONVERT;
    NvBufSurface* dst = nullptr;
    if (NvBufSurfaceAllocate(&dst, 1, &ap) != 0 || !dst) {
        gst_buffer_unmap(gst_buf, &map);
        return;
    }
    dst->numFilled = 1;

    // VIC-accelerated source crop + colour-format + resize in one pass.
    NvBufSurfTransformRect src_rect {
        guint(py), guint(px), guint(pw_px), guint(ph_px)  // top, left, width, height
    };
    NvBufSurfTransformRect dst_rect {
        0, 0, guint(CROP_OUT_W), guint(CROP_OUT_H)
    };
    NvBufSurfTransformParams tp{};
    tp.src_rect = &src_rect;
    tp.dst_rect = &dst_rect;
    tp.transform_flag =
        NVBUFSURF_TRANSFORM_FILTER |
        NVBUFSURF_TRANSFORM_CROP_SRC |
        NVBUFSURF_TRANSFORM_CROP_DST;
    tp.transform_filter = NvBufSurfTransformInter_Default;

    // Narrow the source surface to just our batch slot (batch-size=1
    // per worker, but this keeps us robust against future batching).
    NvBufSurface tmp_in = *in_surf;
    tmp_in.surfaceList = &in_surf->surfaceList[frame->batch_id];
    tmp_in.numFilled   = 1;
    tmp_in.batchSize   = 1;

    if (NvBufSurfTransform(&tmp_in, dst, &tp) != NvBufSurfTransformError_Success) {
        NvBufSurfaceDestroy(dst);
        gst_buffer_unmap(gst_buf, &map);
        return;
    }

    // CPU-map the destination, sync, read pixels.
    if (NvBufSurfaceMap(dst, 0, 0, NVBUF_MAP_READ) != 0) {
        NvBufSurfaceDestroy(dst);
        gst_buffer_unmap(gst_buf, &map);
        return;
    }
    NvBufSurfaceSyncForCpu(dst, 0, 0);

    NvBufSurfaceParams& dp = dst->surfaceList[0];
    const auto* rgba = static_cast<const uint8_t*>(dp.mappedAddr.addr[0]);
    const int pitch = int(dp.planeParams.pitch[0]);

    std::string out_path = ctx.thumbs_dir + "/" + short_id + ".jpg";
    (void)encodeJpegRGBA(rgba, pitch, 0, 0, CROP_OUT_W, CROP_OUT_H, 85, out_path);

    NvBufSurfaceUnMap(dst, 0, 0);
    NvBufSurfaceDestroy(dst);
    gst_buffer_unmap(gst_buf, &map);
}

// Object-detection thumbnail dims. Smaller than face (objects matter
// less as thumbnails and we may write a lot more of them); 128×128
// JPEGs at quality 75 are ~2.5 KB.
constexpr int OBJ_CROP_OUT_W = 128;
constexpr int OBJ_CROP_OUT_H = 128;

// saveObjectCropAndHash mirrors saveFaceCrop but for non-face / non-
// plate detections. Returns the 64-bit average-hash of the bbox
// crop so the probe can attach it to the NATS payload. Writes a
// 128×128 JPEG thumbnail to {thumbs_dir_objects}/{short_id}.jpg.
//
// Returns 0 on any failure (missing thumbs dir, bbox too small, VIC
// transform error). Caller can still emit the detection without a
// phash — suppression just doesn't apply.
//
// No 1.4x padding (unlike faces): the bbox shape for objects is
// already larger + less aspect-sensitive, and padding would dilute
// the hash's signal from the object itself.
std::uint64_t saveObjectCropAndHash(const ProbeCtx& ctx, GstBuffer* gst_buf,
                                    NvDsFrameMeta* frame,
                                    float nx, float ny, float nw, float nh,
                                    const std::string& short_id) {
    if (!gst_buf || !frame) return 0;

    float x0 = nx, y0 = ny, x1 = nx + nw, y1 = ny + nh;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > 1) x1 = 1;
    if (y1 > 1) y1 = 1;

    GstMapInfo map{};
    if (!gst_buffer_map(gst_buf, &map, GST_MAP_READ)) return 0;
    auto* in_surf = reinterpret_cast<NvBufSurface*>(map.data);
    if (!in_surf || frame->batch_id >= in_surf->numFilled) {
        gst_buffer_unmap(gst_buf, &map);
        return 0;
    }
    NvBufSurfaceParams& in_p = in_surf->surfaceList[frame->batch_id];
    const int W = int(in_p.width);
    const int H = int(in_p.height);

    int px = int(x0 * W);
    int py = int(y0 * H);
    int pw_px = int((x1 - x0) * W);
    int ph_px = int((y1 - y0) * H);
    if (pw_px <= 8 || ph_px <= 8) {
        // Tiny detections — pHash on an 8x8 downsample of a sub-8x8
        // source is nonsense. Skip.
        gst_buffer_unmap(gst_buf, &map);
        return 0;
    }

    NvBufSurfaceAllocateParams ap{};
    ap.params.gpuId        = in_surf->gpuId;
    ap.params.width        = guint(OBJ_CROP_OUT_W);
    ap.params.height       = guint(OBJ_CROP_OUT_H);
    ap.params.size         = 0;
    ap.params.isContiguous = true;
    ap.params.colorFormat  = NVBUF_COLOR_FORMAT_RGBA;
    ap.params.layout       = NVBUF_LAYOUT_PITCH;
    ap.params.memType      = NVBUF_MEM_SURFACE_ARRAY;
    ap.memtag              = NvBufSurfaceTag_VIDEO_CONVERT;
    NvBufSurface* dst = nullptr;
    if (NvBufSurfaceAllocate(&dst, 1, &ap) != 0 || !dst) {
        gst_buffer_unmap(gst_buf, &map);
        return 0;
    }
    dst->numFilled = 1;

    NvBufSurfTransformRect src_rect {
        guint(py), guint(px), guint(pw_px), guint(ph_px)
    };
    NvBufSurfTransformRect dst_rect {
        0, 0, guint(OBJ_CROP_OUT_W), guint(OBJ_CROP_OUT_H)
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
    tmp_in.surfaceList = &in_surf->surfaceList[frame->batch_id];
    tmp_in.numFilled   = 1;
    tmp_in.batchSize   = 1;

    if (NvBufSurfTransform(&tmp_in, dst, &tp) != NvBufSurfTransformError_Success) {
        NvBufSurfaceDestroy(dst);
        gst_buffer_unmap(gst_buf, &map);
        return 0;
    }

    if (NvBufSurfaceMap(dst, 0, 0, NVBUF_MAP_READ) != 0) {
        NvBufSurfaceDestroy(dst);
        gst_buffer_unmap(gst_buf, &map);
        return 0;
    }
    NvBufSurfaceSyncForCpu(dst, 0, 0);

    NvBufSurfaceParams& dp = dst->surfaceList[0];
    const auto* rgba = static_cast<const std::uint8_t*>(dp.mappedAddr.addr[0]);
    const int pitch = int(dp.planeParams.pitch[0]);

    // Compute pHash from an 8×8 luma downsample of the 128×128 crop.
    std::uint8_t luma[64];
    downsampleToLuma8x8(rgba, OBJ_CROP_OUT_W, OBJ_CROP_OUT_H, pitch, luma);
    std::uint64_t hash = computeAverageHash64(luma);

    // Best-effort JPEG thumbnail. A failed write is fine — the flag
    // path can fall back to the live-preview ring. Gate on the
    // thumbs_dir_objects being configured; empty means "don't write".
    if (!ctx.thumbs_dir_objects.empty()) {
        std::string out_path = ctx.thumbs_dir_objects + "/" + short_id + ".jpg";
        (void)encodeJpegRGBA(rgba, pitch, 0, 0, OBJ_CROP_OUT_W, OBJ_CROP_OUT_H, 75, out_path);
    }

    NvBufSurfaceUnMap(dst, 0, 0);
    NvBufSurfaceDestroy(dst);
    gst_buffer_unmap(gst_buf, &map);
    return hash;
}

// Called for every batched frame leaving the last nvinfer (LPRNet
// when ANPR is enabled, tracker otherwise). Emits two payload
// shapes — kind="object" for pgie detections, kind="anpr" for
// plates with decoded text.
GstPadProbeReturn InferSrcProbe(GstPad*, GstPadProbeInfo* info, gpointer user) {
    auto* ctx = static_cast<ProbeCtx*>(user);
    GstBuffer* buf = gst_pad_probe_info_get_buffer(info);
    if (!buf) return GST_PAD_PROBE_OK;

    ctx->hb_buffers++;
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch) {
        ctx->hb_batch_null++;
        if ((ctx->hb_buffers % 250) == 0) {
            std::cerr << "probe[" << ctx->primaryCameraId() << "]: buffers="
                      << ctx->hb_buffers << " batch_null=" << ctx->hb_batch_null
                      << " (no NvDsBatchMeta on pad)\n";
        }
        return GST_PAD_PROBE_OK;
    }

    gint64 ts_ns = g_get_real_time() * 1000;  // µs → ns
    // Millisecond-resolution ISO 8601. Second-resolution was too
    // coarse for the Live overlay's age-out logic — multiple events
    // emitted in the same wall-clock second shared one `ts`, and
    // when the source clock drifted vs the client filter (or the
    // broker introduced any latency) bboxes could vanish despite
    // the object still being detected each frame.
    auto iso = [ts_ns]{
        std::time_t t = ts_ns / 1'000'000'000;
        long ms = (ts_ns / 1'000'000) % 1000;
        std::tm tm{}; gmtime_r(&t, &tm);
        char b[40];
        std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", &tm);
        char out[40];
        std::snprintf(out, sizeof out, "%s.%03ldZ", b, ms);
        return std::string(out);
    }();

    for (NvDsMetaList* fl = batch->frame_meta_list; fl; fl = fl->next) {
        auto* frame = static_cast<NvDsFrameMeta*>(fl->data);
        if (!frame) continue;
        const int W = frame->source_frame_width  ? frame->source_frame_width  : 1920;
        const int H = frame->source_frame_height ? frame->source_frame_height : 1080;
        // Per-frame camera identity. Today there is one source per
        // worker so this resolves to ctx->sources[0]; once batched-mux
        // groups land it picks the right id from the per-frame
        // source_id without touching the rest of the loop.
        const std::string& cam_id = ctx->cameraIdFor(frame->source_id);
        bool frame_had_obj = false;

        for (NvDsMetaList* ol = frame->obj_meta_list; ol; ol = ol->next) {
            auto* obj = static_cast<NvDsObjectMeta*>(ol->data);
            if (!obj) continue;
            ctx->hb_objects++;
            frame_had_obj = true;

            float x = obj->rect_params.left   / float(W);
            float y = obj->rect_params.top    / float(H);
            float w = obj->rect_params.width  / float(W);
            float h = obj->rect_params.height / float(H);

            const bool is_plate = (obj->unique_component_id == LPDNET_GIE_ID);
            const bool is_face  = (obj->unique_component_id == SCRFD_GIE_ID);
            const char* label = is_plate
                ? "plate"
                : is_face
                    ? "face"
                    : (obj->obj_label[0] ? obj->obj_label : "object");

            // Class-mute gate at source. Drops before NATS publish so
            // muted classes don't reach Live bboxes, SSE, or event-
            // processor. The Go rules engine runs an identical gate as
            // defence-in-depth; both staying in sync is enforced by the
            // resolution formula living in both languages.
            if (!ctx->muted_classes.empty() &&
                ctx->muted_classes.count(label) > 0) {
                continue;
            }

            // ANPR branch: only publish when we have a decoded plate
            // string — a plate crop with no OCR output is noise.
            std::string plate, parent;
            if (is_plate) {
                plate = extractPlateText(obj);
                if (plate.empty()) continue;
                parent = parentVehicleClass(obj);
            }
            // Face branch: drop tiny faces (below MIN_FACE_PX on any
            // axis — ArcFace output on 10×10 px crops is noise). Pull
            // the 512-d embedding from tensor-meta; if ArcFace didn't
            // run (SGIE off, or the stage errored), skip the publish.
            std::string embedding_b64;
            if (is_face) {
                if (obj->rect_params.width < MIN_FACE_PX ||
                    obj->rect_params.height < MIN_FACE_PX) {
                    continue;
                }
                embedding_b64 = extractFaceEmbedding(obj);
                if (embedding_b64.empty()) continue;
            }
            const char* kind = is_plate ? "anpr" : is_face ? "face" : "object";
            // Plate inherits its vehicle's track_id so the rules
            // engine can correlate plate ↔ car without extra state.
            // Faces get their own SCRFD track_id (no parent).
            const uint64_t track_id = (is_plate && obj->parent)
                ? obj->parent->object_id
                : obj->object_id;

            // Compute the detection id once — it doubles as the face
            // thumbnail filename so the event-processor can rename
            // from {event_id}.jpg to {pg_id}.jpg after INSERT.
            const std::string det_id = short_id();
            if (is_face) {
                saveFaceCrop(*ctx, buf, frame, x, y, w, h, det_id);
            }

            // Object pHash + thumbnail. Only for plain object
            // detections (not face / plate — those have their own
            // matching paths). Zero cost when the crop fails.
            std::uint64_t obj_phash = 0;
            if (!is_face && !is_plate) {
                obj_phash = saveObjectCropAndHash(*ctx, buf, frame, x, y, w, h, det_id);
            }

            std::ostringstream js;
            js << "{"
               << "\"id\":\""         << det_id               << "\","
               << "\"camera_id\":\""  << json_escape(cam_id) << "\","
               << "\"ts\":\""         << iso                  << "\","
               << "\"class_name\":\"" << json_escape(label)   << "\","
               << "\"kind\":\""       << kind                 << "\","
               << "\"confidence\":"   << obj->confidence      << ","
               << "\"bbox\":{\"x\":"  << x << ",\"y\":" << y
               <<          ",\"w\":"  << w << ",\"h\":" << h << "},"
               << "\"track_id\":\""   << track_id             << "\"";
            if (is_plate) {
                js << ",\"attributes\":{"
                   << "\"plate\":\""         << json_escape(plate)  << "\"";
                if (!parent.empty()) {
                    js << ",\"parent_class\":\"" << json_escape(parent) << "\"";
                }
                js << "}";
            } else if (is_face) {
                js << ",\"attributes\":{"
                   << "\"embedding\":\""     << embedding_b64       << "\""
                   << "}";
            }
            // pHash for object detections rides as a TOP-LEVEL field on
            // the NATS payload, not inside attributes — event-processor
            // unmarshals it directly into a typed column. Plain-object
            // rows that aren't matched against a flag therefore land with
            // attributes = NULL in postgres, saving the JSONB row +
            // TOAST pointer.
            if (!is_plate && !is_face && obj_phash != 0) {
                js << ",\"phash\":\"" << uint64ToHex16(obj_phash) << "\"";
            }
            js << "}";
            std::string payload = js.str();
            std::string subj = std::string("fnvr.events.detection.") + cam_id;
            if (ctx->nats && ctx->nats->Publish(subj, payload)) {
                ctx->hb_published++;
            }
        }
        if (frame_had_obj) ctx->hb_frames_obj++;
    }
    if ((ctx->hb_buffers % 250) == 0) {
        std::cerr << "probe[" << ctx->primaryCameraId() << "]: buffers=" << ctx->hb_buffers
                  << " batch_null=" << ctx->hb_batch_null
                  << " frames_with_obj=" << ctx->hb_frames_obj
                  << " objects=" << ctx->hb_objects
                  << " published=" << ctx->hb_published << "\n";
    }
    return GST_PAD_PROBE_OK;
}

}  // namespace
#endif  // FNVR_HAS_DEEPSTREAM

namespace {

// KeyframeGate drops non-keyframe buffers until the first keyframe arrives,
// at which point it removes itself from the pad. This is the reliable way
// to satisfy splitmuxsink's check_completed_gop g_assert, which fires if
// the very first buffer at its input isn't on a GOP boundary. Identity
// element's drop-buffer-flags didn't cut it for NVENC H.265 on v4l2 sources.
struct KeyframeGate {
    std::atomic<bool> open{false};
};

GstPadProbeReturn KeyframeGateProbe(GstPad*, GstPadProbeInfo* info, gpointer user) {
    auto* gate = static_cast<KeyframeGate*>(user);
    if (gate->open.load()) return GST_PAD_PROBE_OK;
    GstBuffer* buf = gst_pad_probe_info_get_buffer(info);
    if (!buf) return GST_PAD_PROBE_OK;
    // No DELTA_UNIT flag → this is a keyframe (sync point).
    if (!GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT)) {
        gate->open.store(true);
        std::cerr << "keyframe gate: opened\n";
        return GST_PAD_PROBE_OK;
    }
    return GST_PAD_PROBE_DROP;
}

void AttachKeyframeGate(GstElement* pipeline, const char* element_name) {
    GstElement* el = gst_bin_get_by_name(GST_BIN(pipeline), element_name);
    if (!el) return;
    GstPad* src = gst_element_get_static_pad(el, "src");
    if (src) {
        auto* gate = new KeyframeGate();  // leaked intentionally — pipeline-scoped
        gst_pad_add_probe(src, GST_PAD_PROBE_TYPE_BUFFER, &KeyframeGateProbe, gate, nullptr);
        gst_object_unref(src);
    }
    gst_object_unref(el);
}

// FlowCounter probe. User_data is a pointer to the
// std::atomic<uint64_t> on the owning pipeline instance; the probe
// runs on every buffer that flows through the src pad and bumps
// the count. Sampled by main.cpp's flow_watchdog thread.
GstPadProbeReturn FlowCounterProbe(GstPad*, GstPadProbeInfo*, gpointer user) {
    auto* counter = static_cast<std::atomic<std::uint64_t>*>(user);
    counter->fetch_add(1, std::memory_order_relaxed);
    return GST_PAD_PROBE_OK;
}

void AttachFlowCounter(GstElement* pipeline, const char* element_name,
                       std::atomic<std::uint64_t>* counter) {
    GstElement* el = gst_bin_get_by_name(GST_BIN(pipeline), element_name);
    if (!el) return;
    GstPad* src = gst_element_get_static_pad(el, "src");
    if (src) {
        gst_pad_add_probe(src, GST_PAD_PROBE_TYPE_BUFFER,
                          &FlowCounterProbe, counter, nullptr);
        gst_object_unref(src);
    }
    gst_object_unref(el);
}

}  // namespace

GstElement* SingleCameraPipeline::BuildPipeline() {
    auto now_tm = [] {
        auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm{};
        gmtime_r(&tt, &tm);
        return tm;
    }();
    char datebuf[64];
    std::strftime(datebuf, sizeof(datebuf), "%Y/%m/%d/%H", &now_tm);
    fs::path dir = fs::path(recordings_dir_) / datebuf / cam_.id;
    std::error_code ec;
    fs::create_directories(dir, ec);

    // Source selection: URL scheme picks the element. v4l2:// → v4l2src,
    // everything else → rtspsrc (good default for RTSP, and rtspsrc also
    // tolerates SRT/RTMP when paired with protocols=). Additional schemes
    // (rtmpsrc, srtsrc) land with the upstream source-factory rework in M3.
    // Per-camera MediaMTX re-muxer: when enabled, swap the source URL
    // for the local MediaMTX path that api-server has already wired
    // up to proxy the real upstream. The source looks like a normal
    // RTSP server to us — rtspsrc, h264parse, everything downstream
    // — but MediaMTX has re-normalised the H.264 framing so qtmux
    // and NVDEC see a clean bitstream.
    std::string url = cam_.url;
    if (cam_.mtx_proxy) {
        url = "rtsp://mediamtx:8554/proxy_" + cam_.id;
        std::cerr << "pipeline[" << cam_.id << "]: via mediamtx proxy ("
                  << url << ")\n";
    }
    const bool is_v4l2 = url.rfind("v4l2://", 0) == 0;
    const std::string v4l2_dev = is_v4l2 ? url.substr(7) : "";

    // Detector-shape flag computed early because the transcode branch
    // above the tee needs to know it (the pre-tee h264parse is only
    // needed when an inference branch exists downstream). The full
    // per-camera detector resolution (wants_anpr_ / wants_face_) is
    // done later where the SGIE chain strings get built.
    const bool skip_inference_early =
        cam_.enabled_detectors.size() == 1 &&
        cam_.enabled_detectors[0] == "none";

    // Live-thumbnail sidecar. A tee branch downsamples to 1 fps and writes
    // a single JPEG file that gets rewritten each second. The
    // /cameras/<id>/snapshot.jpg endpoint prefers this over segment
    // extraction for near-real-time preview.
    const std::string live_jpg_done = "/var/lib/fnvr/live/" + cam_.id + ".jpg";
    fs::create_directories("/var/lib/fnvr/live", ec);

    std::ostringstream p;

    // Shared source → H.264 elementary stream with a `rawtee.` named tee
    // the live-JPEG branch taps. Common structure for both DeepStream and
    // fallback paths.
    if (is_v4l2) {
        // Direct v4l2 pipelines (v4l2src → nvv4l2 → tee) SIGSEGV after
        // NVENC init on Jetson in our container. Users should instead run
        // USB cams through MediaMTX (rtsp://mediamtx:8554/fnvr-usb0 — see
        // deploy/docker/docker-compose.yml). We still recognise v4l2://
        // here to fail-fast with a clear message rather than hang.
        std::cerr << "pipeline[" << cam_.id
                  << "]: v4l2:// URLs aren't supported — point the camera at "
                     "rtsp://mediamtx:8554/usb0 instead (usb-bridge service).\n";
        return nullptr;
    } else {
        // RTSP source: probe codec + dimensions. Reolink cams in particular
        // name their paths "h264Preview_…" but deliver HEVC on newer
        // firmware, and panorama cams (e.g. Duo 2) come in at 4608x1728.
        // We keep the source's aspect ratio by deriving target dims from
        // the probed size, capped to 1080 lines.
        auto probe = ProbeRtsp(url);
        if (probe.codec.empty()) probe.codec = "h264";
        std::cerr << "pipeline[" << cam_.id << "]: probed codec=" << probe.codec
                  << " size=" << probe.width << "x" << probe.height << "\n";

        // Take the transcode path only when we have to: rotation
        // requires nvvideoconvert flip-method which only works on
        // decoded NVMM surfaces; mtx_proxy cameras have broken H.264
        // bitstreams (Bambu H2D, cheap Rockchip RTSP servers, etc.)
        // with mid-stream SPS/PPS changes that qtmux rejects with
        // "Could not multiplex stream" 10-20s into a recording, so
        // we route them through NVDEC+NVENC to launder the bitstream.
        //
        // HEVC sources used to force transcode too (the rest of the
        // gst graph was hard-wired to H.264). After the H.265
        // passthrough refactor that's no longer required: parsers,
        // muxers, RTP payloader and the WHEP webrtcbin caps all
        // branch on `pipeline_codec` below. So an HEVC source with
        // no rotation and no mtx_proxy now stays HEVC end-to-end —
        // skips one NVENC session per camera and halves recording
        // disk usage on those streams.
        const bool needs_transcode =
            (cam_.rotation != 0) || cam_.mtx_proxy;
        // pipeline_codec is the codec the gst tee carries. After
        // transcode it's always H.264 (NVENC encoder above is fixed
        // to nvv4l2h264enc). After passthrough it equals the source
        // codec. Drives the rtspclientsink branch's parser choice
        // so MediaMTX receives a clean elementary stream.
        const std::string pipeline_codec =
            needs_transcode ? std::string("h264") : probe.codec;
        pipeline_codec_ = pipeline_codec;
        // GStreamer nvvideoconvert flip-method mapping:
        //   0 = none, 1 = CCW 90°, 2 = 180°, 3 = CW 90°, 4/5 = flips
        // Our camera rotation is clockwise degrees; translate.
        int flip_method = 0;
        switch (cam_.rotation) {
            case 90:  flip_method = 3; break;
            case 180: flip_method = 2; break;
            case 270: flip_method = 1; break;
            default:  flip_method = 0; break;
        }
        const bool rotate_swaps_axes = (cam_.rotation == 90 || cam_.rotation == 270);

        if (needs_transcode) {
            // Compute aspect-preserving target. If we know source size,
            // fit to max(1080) height; otherwise fall back to 1920x1080.
            int src_w = probe.width;
            int src_h = probe.height;
            // After rotation the visible dims swap, so the encoder caps
            // must describe the post-rotation frame.
            if (rotate_swaps_axes) std::swap(src_w, src_h);
            int tw = 1920, th = 1080;
            if (src_w > 0 && src_h > 0) {
                th = std::min(1080, src_h);
                // width scaled, rounded to even (H.264 4:2:0 needs even).
                tw = static_cast<int>(
                    static_cast<double>(th) * src_w / src_h + 0.5);
                if (tw & 1) tw += 1;
            }
            // Clamp width to something sane (avoid ultra-wide 4K → 3520x1080
            // for 21:9 panoramas is ok, but cap at 2880 to keep encoder
            // within NVENC session budget).
            if (tw > 2880) {
                th = static_cast<int>(
                    static_cast<double>(2880) * th / tw + 0.5);
                if (th & 1) tw = (tw / 2880) * th; else tw = 2880;
                tw = 2880;
            }
            std::cerr << "pipeline[" << cam_.id
                      << "]: recording target=" << tw << "x" << th
                      << " rotation=" << cam_.rotation << "\n";

            // tcp-timeout=15s + retry=3 so a hung RTSP SETUP bails instead of
// blocking a worker forever (observed: Reolink firmware hang where
// TCP/554 accepts connects but SETUP never completes, starving the
// whole worker). On timeout, rtspsrc posts an error message — our
// bus watch flips faulted_ and main.cpp exits, letting the
// supervisor respawn with its backoff.
p << "rtspsrc location=" << url
  << " latency=200 protocols=tcp tcp-timeout=15000000 retry=3"
  << " name=src ! ";
            if (probe.codec == "h265") {
                p << "rtph265depay ! h265parse config-interval=1 ! "
                     "video/x-h265,stream-format=byte-stream,alignment=au ! ";
            } else {
                p << "rtph264depay ! h264parse config-interval=1 ! ";
            }
            // Re-mux source → H.264 via NVDEC+NVENC so the rest of the
            // pipeline only has to handle one codec path. Rotation (if
            // any) is applied on the decoded NVMM surface before caps
            // negotiation so the encoder sees the post-rotation dims.
            p << "nvv4l2decoder ! "
                 "nvvideoconvert flip-method=" << flip_method << " ! "
                 "video/x-raw(memory:NVMM),format=NV12,width=" << tw
                 << ",height=" << th << " ! "
                 "nvv4l2h264enc insert-sps-pps=1 idrinterval=30 iframeinterval=30 ! ";
            // The pre-tee h264parse is only needed when the
            // inference branch does a second decode+encode (the tee
            // output is already parsed once by then). For
            // skip_inference+rotation the record branch is a direct
            // passthrough to qtmux — a second parser through the tee
            // makes qtmux fail at caps negotiation with
            // "Could not multiplex stream". Skip it in that case.
            if (!skip_inference_early) {
                p << "h264parse config-interval=1 ! ";
            }
            rec_width_ = tw;
            rec_height_ = th;
        } else {
            // Source passes through unchanged — no rotation, no
            // mtx_proxy. Branch the depay+parse on the source codec
            // so HEVC and H.264 both reach the tee in their native
            // codec. Downstream branches (recording, WHEP, JPEG)
            // also branch on `pipeline_codec` to avoid forcing a
            // re-encode.
            // tcp-timeout=15s + retry=3 so a hung RTSP SETUP bails instead of
// blocking a worker forever (observed: Reolink firmware hang where
// TCP/554 accepts connects but SETUP never completes, starving the
// whole worker). On timeout, rtspsrc posts an error message — our
// bus watch flips faulted_ and main.cpp exits, letting the
// supervisor respawn with its backoff.
p << "rtspsrc location=" << url
  << " latency=200 protocols=tcp tcp-timeout=15000000 retry=3"
  << " name=src ! ";
            if (pipeline_codec_ == "h265") {
                p << "rtph265depay ! ";
            } else {
                p << "rtph264depay ! ";
            }
            // Pre-tee parser: needed on the inference path (our
            // nvstreammux branch consumes parsed AU-aligned frames).
            // Skipped for skip_inference because each tee branch has
            // its own parser and two parsers around a tee trip
            // qtmux with "Could not multiplex stream" — same fix we
            // applied earlier for the transcode+skip_inference case.
            if (!skip_inference_early) {
                if (pipeline_codec_ == "h265") {
                    p << "h265parse config-interval=1 ! ";
                } else {
                    p << "h264parse config-interval=1 ! ";
                }
            }
            if (probe.width > 0 && probe.height > 0) {
                rec_width_ = probe.width;
                rec_height_ = probe.height;
            }
        }
    }

    p << "tee name=t ";

    // Per-camera detector resolution.
    //   enabled_detectors == []         → all SGIEs permitted
    //     (subject to pipeline-level kill switches below)
    //   enabled_detectors == ["none"]   → no inference at all; build a
    //     passthrough record branch and skip attaching the probe
    //   otherwise                       → whitelist membership check
    // Pipeline-level use_anpr_ / use_face_id_ remain the upper bound
    // (a global kill switch). The per-camera flags only *narrow*.
    const auto& det = cam_.enabled_detectors;
    const bool skip_inference_ = det.size() == 1 && det[0] == "none";
    auto detectorListed = [&](const char* kind) {
        if (det.empty()) return true;  // "all" convention
        for (const auto& v : det) if (v == kind) return true;
        return false;
    };
    const bool wants_anpr_ = use_anpr_    && !skip_inference_ && detectorListed("anpr");
    const bool wants_face_ = use_face_id_ && !skip_inference_ && detectorListed("face");

    // --- Recording branch ---
    // Three shapes depending on per-camera detector config:
    //   1. skip_inference_ + rotation==0 → pure H.264 passthrough.
    //      No decode, no NVENC, no inference — cheapest possible, and
    //      the only viable shape for sources we don't want to spend
    //      GPU on (e.g. corrupt-stream cameras recorded for evidence).
    //   2. skip_inference_ + rotation!=0 → decode → rotate → encode,
    //      but no PGIE/tracker/SGIEs on this branch. Happens when an
    //      operator explicitly disabled detection on a rotated cam.
    //   3. full inference branch (default) — pgie + tracker + any
    //      enabled SGIE chains, then nvvideoconvert + NVENC + qtmux.
    p << "t. ! queue max-size-buffers=200 leaky=downstream ! ";

    // Use the probed recording dims if we have them, falling back to 1080p.
    // enable-padding=1 tells nvstreammux to letterbox rather than stretch,
    // preserving the source aspect — important for panorama cams.
    int mux_w = rec_width_ > 0 ? rec_width_ : 1920;
    int mux_h = rec_height_ > 0 ? rec_height_ : 1080;

    // ANPR SGIE chain — LPDNet (plate detector) + LPRNet (OCR) run
    // after the tracker so they see vehicles with stable track_ids.
    // gie-unique-id values (2 + 3) are wired into lpdnet.txt /
    // lprnet.txt so the probe can distinguish plate obj_meta from
    // pgie obj_meta via unique_component_id. Empty string when ANPR
    // is off — the primary chain is unchanged.
    std::string anpr_chain;
    if (wants_anpr_) {
        anpr_chain =
            "nvinfer name=lpdnet config-file-path=/etc/fnvr/nvinfer/lpdnet.txt ! "
            "nvinfer name=lprnet config-file-path=/etc/fnvr/nvinfer/lprnet.txt ! ";
    }
    // SCRFD (face detect, gie-unique-id=4) + ArcFace (embed, id=5).
    // Runs after ANPR so a single frame's nvinfer chain is:
    //   pgie(1) → tracker → lpdnet(2) → lprnet(3) → scrfd(4) → arcface(5)
    // ArcFace surfaces its 512-d output via tensor-meta; the probe
    // base64-encodes it into attributes.embedding.
    std::string face_chain;
    if (wants_face_) {
        face_chain =
            "nvinfer name=scrfd  config-file-path=/etc/fnvr/nvinfer/scrfd.txt ! "
            "nvinfer name=arcface config-file-path=/etc/fnvr/nvinfer/arcface.txt ! ";
    }

    // Primary-detector element. "trt" = DeepStream nvinfer running on
    // the Orin GPU; "hailo" = in-process libhailort inference attached
    // as a src-pad probe on a no-op `queue name=pgie` (see
    // hailo_probe.cpp). The probe walks the batch, downscales each
    // frame's NVMM surface to 640x640 RGB via NvBufSurfTransform, runs
    // yolov11l on the Hailo-8, decodes the NMS output, and attaches
    // NvDsObjectMeta directly to the frame meta. Downstream nvtracker
    // + SGIEs + InferSrcProbe see the same shape regardless of backend
    // — they all bind to the element named "pgie" and read
    // NvDsObjectMeta from the buffer.
    //
    // We deliberately avoid the `hailonet` + `hailofilter` GStreamer
    // plugins: hailonet re-emits a fresh GstBuffer rather than
    // forwarding the input, which drops NvDsBatchMeta and breaks
    // DeepStream's metadata flow. Going direct to libhailort keeps
    // the DeepStream graph single-pipeline.
    std::string pgie_element;
    if (cam_.detector_backend == "hailo") {
        pgie_element = "queue name=pgie max-size-buffers=4 leaky=no ! ";
    } else {
        pgie_element = std::string(
            "nvinfer name=pgie config-file-path=") + infer_config_ + " ! ";
    }

    // Recording is now MediaMTX's job — it subscribes to the same
    // RTSP stream the WHEP/WebRTC live tile uses, writes fragmented
    // MP4 (moof+mdat), serves both live and playback HTTP clips out
    // of the same files. Pipeline-supervisor used to record here via
    // qtmux+filesink, which produced progressive MP4 with a giant
    // pre-allocated moov rewritten every second. Browsers opening
    // an in-progress file saw whatever moov was last flushed and
    // failed to seek past it — most visibly with HEVC, where the
    // mvhd-driven seek table is stricter than H.264.
    //
    // The graph below stays for the inference probes (pgie + tracker
    // + ANPR/face SGIEs publish detections to NATS via a pad probe
    // installed in Start()). It terminates at fakesink because none
    // of the post-tracker frames need to reach disk anymore.
    if (skip_inference_) {
        // No-AI tier: nothing to do — the WHEP/recording branches
        // are off the source tee `t.` independently. Just sink this
        // arm of the tee to keep the pipeline preroll happy.
        p << "fakesink sync=false ";
    } else if (use_deepstream_) {
        // Common to is_v4l2 + RTSP — they used to differ on whether
        // the inference branch fed an encoder; with recording gone
        // both terminate at fakesink. nvv4l2decoder is required even
        // for non-v4l2 sources because the inference branch consumes
        // decoded NVMM frames; nvstreammux + nvinfer + nvtracker
        // operate on those.
        p << "nvv4l2decoder ! "
             "mux.sink_0 nvstreammux name=mux batch-size=1 "
             "  width=" << mux_w << " height=" << mux_h
             << " live-source=1 batched-push-timeout=40000 enable-padding=1 ! "
          << pgie_element
          << "nvtracker name=tracker "
             "  ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
             "  ll-config-file=/etc/fnvr/nvinfer/tracker_NvDCF.yml "
             "  tracker-width=960 tracker-height=544 ! "
          << anpr_chain
          << face_chain
          << "fakesink sync=false ";
    } else {
        // No DeepStream tier — pipeline only exists to get RTSP into
        // MediaMTX via the WHEP branch elsewhere; this arm just
        // sinks the tee.
        p << "fakesink sync=false ";
    }

    // --- Live-thumbnail branch removed ---
    // The 480×270 1-fps preview JPEG ring at /var/lib/fnvr/live/<cam>.<n>.jpg
    // used to be a separate decode + jpegenc tee branch. That doubled NVDEC
    // load per camera — the inference branch above already decodes the
    // exact same frames microseconds earlier. We now generate the ring via
    // a pad probe on pgie.src that taps the in-flight NVMM surface, scales
    // to 480×270 via VIC, and encodes a JPEG via the existing
    // face_crop_jpeg helper. See preview_probe.{h,cpp} and the probe
    // attachment block below ensureDstSurface() / Start().
    //
    // (Face-crop branch removed — thumbnails now come from the probe
    // directly via NvBufSurfTransform on the inference buffer.)

    // --- WebRTC live-view branch ---
    // Publish the source bitstream to MediaMTX over RTSP at
    // rtsp://mediamtx:8554/live_<camera_id>. Browsers fetch live
    // video via MediaMTX's built-in WHEP server (port 8889 on the
    // host LAN IP) — that path handles ICE/STUN/NAT-1to1 properly.
    //
    // Earlier shape used an in-process webrtcbin per viewer fed from
    // an rtph264pay/rtp_tee branch; that never worked end-to-end from
    // a LAN browser because GStreamer's webrtcbin doesn't expose a
    // nat-1to1-ip override and only ever advertised host candidates
    // on the docker bridge (172.18.x.x), unreachable from the LAN.
    // MediaMTX has webrtcAdditionalHosts and a single-port UDP mux
    // for ICE; both are wired in deploy/docker/docker-compose.yml.
    //
    // rtspclientsink takes parsed elementary-stream input and lets
    // its built-in payloader handle RTP framing. protocols=tcp keeps
    // the publish on the same TCP path the rest of the docker network
    // uses; UDP would need extra port-forwarding inside the bridge.
    p << "t. ! queue max-size-buffers=200 leaky=downstream ! ";
    if (pipeline_codec_ == "h265") {
        p << "h265parse config-interval=-1 ! ";
    } else {
        p << "h264parse config-interval=-1 ! ";
    }
    p << "rtspclientsink name=mtxpush "
         "  location=rtsp://mediamtx:8554/live_" << cam_.id
      << " protocols=tcp";

    std::string desc = p.str();
    std::cerr << "pipeline[" << cam_.id << "]: " << desc << "\n";

    GError* err = nullptr;
    GstElement* pipeline = gst_parse_launch(desc.c_str(), &err);
    if (!pipeline) {
        std::cerr << "gst_parse_launch: " << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        return nullptr;
    }

#if FNVR_HAS_DEEPSTREAM
    // Per-camera skip_inference tier has no nvinfer elements at all —
    // nothing to probe, nothing to publish. The recording + WHEP +
    // thumbnail branches still work without the probe.
    const bool skip_inference_probe = cam_.enabled_detectors.size() == 1 &&
                                       cam_.enabled_detectors[0] == "none";

    // For the hailo backend the `pgie` element is a no-op queue. Install
    // the Hailo inference probe on its src pad so libhailort runs between
    // nvstreammux and nvtracker, injecting NvDsObjectMeta into the frame
    // meta exactly as nvinfer would have. This keeps the rest of the
    // DeepStream chain (tracker + SGIEs + InferSrcProbe) unchanged.
    if (use_deepstream_ && !skip_inference_probe &&
        cam_.detector_backend == "hailo") {
        GstElement* pgie_elem = gst_bin_get_by_name(GST_BIN(pipeline), "pgie");
        if (pgie_elem) {
            GstPad* src = gst_element_get_static_pad(pgie_elem, "src");
            if (src) {
                // Source dimensions come from the probed recording dims —
                // same values we set on nvstreammux width/height.
                int hsrc_w = rec_width_  > 0 ? rec_width_  : 1920;
                int hsrc_h = rec_height_ > 0 ? rec_height_ : 1080;
                auto* hctx = fnvr::hailo_probe_ctx_new(
                    "/var/lib/fnvr/models/hailo/yolov11l.hef",
                    hsrc_w, hsrc_h);
                if (hctx) {
                    gst_pad_add_probe(src, GST_PAD_PROBE_TYPE_BUFFER,
                                      &fnvr::HailoInferProbe, hctx, nullptr);
                    std::cerr << "pipeline[" << cam_.id
                              << "]: hailo-8 inference probe attached on pgie.src\n";
                } else {
                    std::cerr << "pipeline[" << cam_.id
                              << "]: hailo-8 inference setup failed; "
                              << "pipeline runs without detections on this cam\n";
                }
                gst_object_unref(src);
            }
            gst_object_unref(pgie_elem);
        }
    }

    // Live-thumbnail probe — taps the decoded NVMM frame already in
    // flight on the inference branch and writes a 1 fps 480×270 JPEG
    // ring at /var/lib/fnvr/live/<cam>.<n>.jpg. Replaces the earlier
    // parallel decode branch that doubled NVDEC load per camera. See
    // preview_probe.cpp for the rate-limit + transform + encode flow.
    if (use_deepstream_ && !skip_inference_probe) {
        GstElement* pgie_elem = gst_bin_get_by_name(GST_BIN(pipeline), "pgie");
        if (pgie_elem) {
            GstPad* src = gst_element_get_static_pad(pgie_elem, "src");
            if (src) {
                int sw = rec_width_  > 0 ? rec_width_  : 1920;
                int sh = rec_height_ > 0 ? rec_height_ : 1080;
                auto* pctx = fnvr::preview_probe_ctx_new(
                    cam_.id, "/var/lib/fnvr/live", sw, sh);
                if (pctx) {
                    gst_pad_add_probe(src, GST_PAD_PROBE_TYPE_BUFFER,
                                      &fnvr::PreviewSnapshotProbe, pctx, nullptr);
                    std::cerr << "pipeline[" << cam_.id
                              << "]: preview-thumbnail probe attached on pgie.src\n";
                }
                gst_object_unref(src);
            }
            gst_object_unref(pgie_elem);
        }
    }

    if (use_deepstream_ && nats_ && !skip_inference_probe) {
        // Attach the detection probe to the LAST nvinfer in the
        // active chain. SGIE-produced obj_meta + user_meta only
        // appears on buffers downstream of the SGIE, so a probe on
        // e.g. tracker.src never sees face or plate objects.
        //
        // For the Hailo backend (detector_backend == "hailo") the detector
        // probe injects NvDsObjectMeta on the pgie (queue) src pad, BEFORE
        // nvtracker. nvtracker apparently drops NvDsObjectMeta it didn't
        // produce (even with detector_bbox_info populated), so for hailo
        // cameras we prefer attaching the InferSrcProbe on the same pad as
        // the hailo injector (pgie.src) — objects observed there survive
        // the tee that feeds them both. Tradeoff: no track_ids on hailo
        // cameras until we figure out how to make nvtracker accept our
        // injected objects.
        GstElement* attach = nullptr;
        const char* attach_name = nullptr;
        if (cam_.detector_backend == "hailo") {
            attach = gst_bin_get_by_name(GST_BIN(pipeline), "pgie");
            attach_name = "pgie";
        } else {
            attach = gst_bin_get_by_name(GST_BIN(pipeline), "arcface");
            attach_name = "arcface";
            if (!attach) { attach = gst_bin_get_by_name(GST_BIN(pipeline), "lprnet"); attach_name = "lprnet"; }
            if (!attach) { attach = gst_bin_get_by_name(GST_BIN(pipeline), "tracker"); attach_name = "tracker"; }
            if (!attach) { attach = gst_bin_get_by_name(GST_BIN(pipeline), "pgie"); attach_name = "pgie"; }
        }
        if (attach) {
            GstPad* src = gst_element_get_static_pad(attach, "src");
            if (src) {
                // Leaked on purpose: lifetime matches the pipeline, cleaned up
                // when the process exits. Fine for M2, tighten when we have
                // multi-pipeline lifecycle.
                auto* ctx = new ProbeCtx;
                ctx->sources       = {cam_.id};
                ctx->nats          = nats_;
                ctx->muted_classes = cam_.muted_classes;
                // When face_id is on the probe also writes a JPEG
                // crop of each face using NvBufSurfTransform on the
                // same NVMM buffer the detection came from — zero
                // temporal drift. Empty thumbs_dir disables crop
                // writes without branching the probe.
                if (use_face_id_) {
                    ctx->thumbs_dir = "/var/lib/fnvr/thumbs/faces";
                    std::error_code _ec;
                    std::filesystem::create_directories(ctx->thumbs_dir, _ec);
                }
                // Object detections always get a small thumbnail +
                // pHash, independent of face_id. The Flags page
                // needs the thumbnail so the operator can see what
                // they're flagging.
                {
                    ctx->thumbs_dir_objects = "/var/lib/fnvr/thumbs/objects";
                    std::error_code _ec;
                    std::filesystem::create_directories(ctx->thumbs_dir_objects, _ec);
                }
                gst_pad_add_probe(src, GST_PAD_PROBE_TYPE_BUFFER, &InferSrcProbe, ctx, nullptr);
                std::cerr << "pipeline[" << cam_.id
                          << "]: detection probe attached on " << attach_name << ".src\n";
                gst_object_unref(src);
            }
            gst_object_unref(attach);
        } else {
            std::cerr << "pipeline[" << cam_.id
                      << "]: FAILED to attach detection probe — no element matched"
                      << " (looked for arcface/lprnet/tracker/pgie)\n";
        }
    }
#endif

    // Recording is now MediaMTX's job. The keyframe-gate + flow
    // watchdog used to sit on `recparse` (the post-encoder parser
    // before qtmux) — both are gone now. If we ever need flow
    // monitoring back, attach the counter to a still-flowing pad
    // (e.g. tracker.src or the WHEP rtspclientsink request pad).

    return pipeline;
}

// Time this child-worker process started. Used by abortWorkerAfterFault
// to decide whether an early fault should publish `failed` to the UI
// (it shouldn't — the supervisor will respawn and the new child may
// reach PLAYING). Only useful in worker child processes; the parent
// supervisor has its own grace logic.
static const auto g_worker_process_start = std::chrono::steady_clock::now();
// The grace window within which abortWorkerAfterFault suppresses its
// `failed` publish. Zero = publish immediately like before. Set by
// main.cpp at worker startup from settings.pipeline.startup_grace_sec.
static int g_worker_startup_grace_sec = 0;

void SetWorkerStartupGraceSec(int sec) { g_worker_startup_grace_sec = sec; }

// abortWorkerAfterFault publishes {"state":"failed"} with a bounded
// flush, then _exits the process so the parent supervisor respawns
// with a clean slate. Used for bus ERROR/EOS and (from main.cpp) the
// data-flow watchdog. We don't try to stop the pipeline gracefully —
// today's incident was a 37-minute zombie worker caused by
// gst_element_set_state(NULL) blocking forever on a wedged NvMedia
// element, which defeats the whole point of having a supervisor.
[[noreturn]] static void abortWorkerAfterFault(const std::string& cam_id,
                                               NatsPublisher* nats,
                                               const char* reason) {
    std::cerr << "worker[" << cam_id << "]: hard-exit rc=3 ("
              << reason << ")\n";
    // Startup grace: if this child has been up for less than the grace
    // window, don't publish `failed` — the parent supervisor will
    // respawn shortly and the next child may reach PLAYING. Without
    // this, a slow-source camera (MediaMTX-proxied, cold-boot RTSPS,
    // etc.) flashes `failed` in the UI on its first warmup exit even
    // though the supervisor is about to retry successfully.
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - g_worker_process_start).count();
    const bool within_grace =
        g_worker_startup_grace_sec > 0 &&
        age < g_worker_startup_grace_sec;
    if (nats && !within_grace) {
        const std::string subj = "fnvr.state.camera." + cam_id;
        const std::string payload =
            "{\"camera_id\":\"" + cam_id + "\",\"state\":\"failed\"}";
        // flush=true so the message reaches the broker before we
        // _exit. 2 s is plenty on a localhost bridge; if NATS itself
        // is the problem the publish returns quickly with an error
        // and we exit anyway.
        nats->Publish(subj, payload, /*flush=*/true);
    } else if (within_grace) {
        std::cerr << "worker[" << cam_id << "]: within "
                  << g_worker_startup_grace_sec
                  << "s startup grace (age=" << age
                  << "s), not publishing failed\n";
    }
    std::_Exit(3);
}

gboolean SingleCameraPipeline::BusHandler(GstBus*, GstMessage* msg, gpointer user_data) {
    auto* self = static_cast<SingleCameraPipeline*>(user_data);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cerr << "pipeline[" << self->cam_.id << "]: EOS\n";
            self->faulted_.store(true);
            abortWorkerAfterFault(self->cam_.id, self->nats_, "EOS");
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            std::string msg_str = err ? err->message : "?";
            std::cerr << "pipeline[" << self->cam_.id << "] error: "
                      << msg_str << "\n";
            if (err) g_error_free(err);
            g_free(dbg);
            self->faulted_.store(true);
            // Do NOT gst_element_set_state(NULL) here — that blocks
            // indefinitely when a downstream element is wedged inside
            // NvMedia (the failure mode we're recovering from).
            // Hard-exit and let the supervisor respawn.
            abortWorkerAfterFault(self->cam_.id, self->nats_, "bus error");
        }
        case GST_MESSAGE_STATE_CHANGED: {
            if (GST_MESSAGE_SRC(msg) == GST_OBJECT(self->pipeline_)) {
                GstState oldS, newS;
                gst_message_parse_state_changed(msg, &oldS, &newS, nullptr);
                if (newS == GST_STATE_PLAYING) {
                    self->playing_.store(true);
                    if (self->nats_) {
                        // Last-value stream on api-server side — see state.go.
                        std::string subj = "fnvr.state.camera." + self->cam_.id;
                        std::string payload = "{\"camera_id\":\"" + self->cam_.id + "\",\"state\":\"running\"}";
                        self->nats_->Publish(subj, payload, /*flush=*/true);
                    }
                }
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

bool SingleCameraPipeline::Start() {
    pipeline_ = BuildPipeline();
    if (!pipeline_) return false;
    GstBus* bus = gst_element_get_bus(pipeline_);
    bus_watch_id_ = gst_bus_add_watch(bus, &SingleCameraPipeline::BusHandler, this);
    gst_object_unref(bus);

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "pipeline[" << cam_.id << "]: failed to set PLAYING\n";
        Stop();
        return false;
    }

    return true;
}

void SingleCameraPipeline::Stop() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    if (bus_watch_id_) {
        g_source_remove(bus_watch_id_);
        bus_watch_id_ = 0;
    }
}

}  // namespace fnvr
