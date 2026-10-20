// src/core/pinned.cu - the pinned host arena and the parallel expert load.
#include "basalt/core/load_progress.hpp"
#include "basalt/core/pinned.hpp"
#include "basalt/core/device.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/system.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

namespace basalt::core {

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
bool clear_error() { (void) cudaGetLastError(); return true; }
}  // namespace


namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                         uint64_t max_pinned_bytes, const std::string& shared_file,
                         uint64_t shared_hash) : capacity(bytes) {
    if (bytes == 0) return;
    basalt::platform::RegionBacking region_backing = basalt::platform::RegionBacking::NormalPages;
    base = basalt::platform::alloc_region(bytes, region_backing, note, shared_file, shared_hash, mapping_base,
                                          mapping_bytes);
    backing = region_backing == basalt::platform::RegionBacking::LargePages ? PageBacking::LargePages
                                                                             : PageBacking::NormalPages;

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base) {
        uint64_t cap = max_pinned_bytes;
        const std::string cap_why = "by the engine (remote experts)";
        const bool capped = cap > 0 && cap < bytes && bounds.size() >= 2;
        const cudaError_t e = capped ? cudaSuccess :
            cudaHostRegister(base, (size_t) bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (!capped && e == cudaSuccess) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = bytes;
        } else if (bounds.size() >= 2 && (capped || clear_error())) {
            // the whole range is refused, so pin it slice by slice from the start.  The rest stays resident through the
            // working-set lock below.  (Slices may differ in size, one per layer.)
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            bool limited = capped;
            uint64_t limit = cap;
            std::string limit_why;
            // Stop below the shared-memory budget rather than at the driver's refusal, and only when the budget is
            // actually smaller than the arena: an arena the segment can hold is pinned whole, as before.
            if (!capped) {
                // The budget belongs to the adapter this arena is pinned for, so name that adapter by its id.
                uint64_t seg = 0;
                std::string seg_why;
                if (basalt::platform::pin_budget(seg, seg_why, basalt::core::device_luid()) && seg < bytes) {
                    limited = true;
                    limit = seg;
                    limit_why = std::move(seg_why);
                }
            }
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
                if (limited && (off > limit || n > limit - off)) break;
                if (cudaHostRegister((uint8_t*) base + off, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                    (void) cudaGetLastError();
                    break;
                }
                slice_starts.push_back(off);
                registered_bytes = off + n;
                ++registered_slices;
            }
            char gib[32];
            std::snprintf(gib, sizeof gib, "%.1f", (double) limit / (double) (1ull << 30));
            note = (capped ? "cudaHostRegister limited to " + std::to_string(cap >> 30) + " GiB " + cap_why + "; " :
                             "cudaHostRegister of the whole arena FAILED (" + std::string(cudaGetErrorString(e)) + "); " +
                             (limited ? "slices capped at " + std::string(gib) + " GiB (" + limit_why +
                                        "); " : std::string())) +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB); " + note;
            if (registered_bytes < bytes) {
                const basalt::platform::LockResult lr =
                    basalt::platform::lock_resident((uint8_t*) base + registered_bytes, bytes - registered_bytes);
                locked_bytes = lr.locked_bytes;
                note = lr.note + "; " + note;
            }
        } else {
            note = std::string("cudaHostRegister FAILED (") + cudaGetErrorString(e) +
                   ") - the arena is NOT pinned, so copies will be slow; " + note;
            // **CONSUME THE ERROR, OR IT LIES ABOUT SOMETHING ELSE LATER.**
            //
            // `cudaGetLastError()` returns the last error and CLEARS it; until something reads it, the error
            // state is sticky.  This failure is caught and handled right here - the arena is simply not pinned -
            // but leaving it set meant the next `cudaGetLastError()` in the engine, which is `gr_read`'s launch
            // check, reported "out of memory" for kernels that allocate nothing.  That cost a round: the arena
            // was written off as not fitting the machine when in fact the only thing wrong was a stale error
            // from this line.
            //
            // It is the same trap `gr.cu` warns about for ASYNC faults, in the other direction: a synchronous
            // failure is sticky too, and it lies about where it happened just as convincingly.
            (void) cudaGetLastError();
            // Keep it resident instead of pinned, so the OS does not page the arena out.
            const basalt::platform::LockResult lr = basalt::platform::lock_resident(base, bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        }
    }
}

PinnedArena::~PinnedArena() {
    if (base) {
        if (locked_bytes) basalt::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            for (uint64_t off : slice_starts) cudaHostUnregister((uint8_t*) base + off);
        } else {
            cudaHostUnregister(base);
        }
        basalt::platform::free_region(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L) off[(size_t) L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk, uint64_t base) {
    LoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    LoadProgress progress("ram", st.bytes);
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::atomic<uint64_t> read_ns_sum{0};   // summed over the threads: see LoadStats::read_seconds
    std::atomic<uint64_t> copy_ns_sum{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        // One handle per thread, seeked once per layer: a shared handle would need a lock around the seek and would
        // serialise the very thing the threads are here to parallelise.  `fread` on a `FILE*` rather than
        // `std::ifstream`: `fread` sees a request bigger than the stream buffer and passes it to the read call
        // unchanged, so the chunk size reaches the disk.  Buffered: the cache should still hold what it can.
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::lock_guard<std::mutex> g(err_mu);
            err = "cannot open " + path;
            return;
        }
        // A `FILE*` has no destructor that closes it, and this function has early returns below (open, seek and
        // short-read failures), so the guard is what keeps the closing correct on every path.
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        uint64_t read_ns = 0, copy_ns = 0;
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // 64-bit seek: the file is far larger than 4 GiB, so a 32-bit `fseek` would wrap
            if (basalt::platform::fseek64(f, (int64_t) (base + off), SEEK_SET) != 0) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "seek to " + std::to_string(off) + " B failed in layer " + std::to_string(L);
                return;
            }
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                const auto t_read = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t) n, f);
                read_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_read).count();
                // A short read is EOF or an I/O error, never a silent zero fill: say WHERE and HOW SHORT, and
                // keep going no further - the caller turns this into a refused load, not a wrong answer.
                if (got != (size_t) n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L) + ": got " + std::to_string(got) + " of "
                          + std::to_string(n) + " B at offset " + std::to_string(off + pos)
                          + (std::ferror(f) != 0 ? " (ferror set)" : "");
                    return;
                }
                const auto t_copy = std::chrono::steady_clock::now();
                std::memcpy(dst + off + pos, buf.data(), (size_t) n);
                h = fnv1a64(buf.data(), n, h);
                copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_copy).count();
                pos += n;
                remaining -= n;
                progress.add(n);
            }
            layer_hash[(size_t) L] = h;
        }
        read_ns_sum.fetch_add(read_ns);
        copy_ns_sum.fetch_add(copy_ns);
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        st.ok = false;
        st.error = err;
        return st;
    }
    progress.finish();
    st.read_seconds = (double) read_ns_sum.load() / 1e9;
    st.copy_seconds = (double) copy_ns_sum.load() / 1e9;
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        std::fprintf(stderr, "stream_bandwidth: cudaMalloc failed for %llu B\n", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

namespace {
PinnedArena* g_report_arena = nullptr;

/// Register ordinary host memory in `slice`-sized pieces until the driver refuses one, then release every piece that
/// was accepted. The pages are touched before each registration, because cudaHostRegister pins what is resident and a
/// region that has not been faulted in yet is a different (and cheaper) thing to pin than the arena's filled pages.
HostRegisterProbe probe_host_register_impl(uint64_t slice_bytes, uint64_t max_bytes) {
    HostRegisterProbe p;
    p.slice_bytes = slice_bytes;
    if (slice_bytes == 0 || max_bytes < slice_bytes) {
        p.error = "the probe's slice or limit is too small";
        return p;
    }
    std::vector<void*> pinned;   // one entry per accepted slice, released before the return
    std::vector<void*> blocks;
    const uint64_t slices = max_bytes / slice_bytes;
    for (uint64_t i = 0; i < slices; ++i) {
        // An ordinary host block, page-aligned and released as one piece. A request this size comes from the OS
        // either way on both platforms; what the probe is measuring is the registration, not the mapping.
        void* block = basalt::platform::alloc_aligned((size_t) slice_bytes, 4096);
        if (block == nullptr) {
            p.error = "cannot allocate another " + std::to_string(slice_bytes >> 20) + " MiB of host memory";
            break;
        }
        blocks.push_back(block);
        std::memset(block, 1, (size_t) slice_bytes);   // resident before it is pinned, like the arena's pages
        if (cudaHostRegister(block, (size_t) slice_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
            const cudaError_t e = cudaGetLastError();
            p.refused_at = (uint64_t) pinned.size() * slice_bytes + slice_bytes;
            p.error = std::string("cudaHostRegister refused at ") +
                      std::to_string((p.refused_at) >> 20) + " MiB: " + cudaGetErrorString(e);
            break;
        }
        pinned.push_back(block);
    }
    // Release everything, whatever happened above: a probe must not leave a registration alive for later measurements.
    for (void* b : pinned) (void) cudaHostUnregister(b);
    for (void* b : blocks) basalt::platform::free_aligned(b);
    p.granted = (uint64_t) pinned.size() * slice_bytes;
    p.slices = (uint32_t) pinned.size();
    p.ok = true;   // a ceiling was found (or the limit was reached) even when the driver refused part way
    if (p.refused_at == 0) p.error = "the probe's limit was reached before the driver refused";
    return p;
}

/// The two one-time probes. Each runs on the first report and is cached, so a reprint does not touch the device or
/// the drive again.
struct H2dProbe {
    bool ok = false;
    uint64_t bytes = 0;
    uint64_t chunk = 0;
    double gib_per_s = 0;
};

H2dProbe h2d_probe_once() {
    static const H2dProbe p = [] {
        H2dProbe r;
        if (g_report_arena == nullptr || g_report_arena->registered_bytes == 0) return r;
        const uint64_t chunk = 8ull << 20;
        uint64_t bytes = std::min<uint64_t>(g_report_arena->registered_bytes, 512ull << 20);
        bytes = bytes / chunk * chunk;
        if (bytes < chunk) return r;
        const StreamStats st = stream_bandwidth(g_report_arena->data(), bytes, chunk, 2);
        if (st.seconds <= 0.0) return r;
        r.ok = true;
        r.bytes = st.bytes;
        r.chunk = chunk;
        r.gib_per_s = st.gib_per_second();
        return r;
    }();
    return p;
}

basalt::platform::LargePageProbe large_page_probe_once() {
    static const basalt::platform::LargePageProbe p = [] {
        const uint64_t want = g_report_arena != nullptr && g_report_arena->capacity > 0
                                  ? g_report_arena->capacity
                                  : (1ull << 30);
        return basalt::platform::probe_large_pages(want);
    }();
    return p;
}
}  // namespace

void set_report_arena(PinnedArena* arena) { g_report_arena = arena; }

HostRegisterProbe probe_host_register(uint64_t slice_bytes, uint64_t max_bytes) {
    HostRegisterProbe p = probe_host_register_impl(slice_bytes, max_bytes);
    // The two figures the engine compares the measurement against: the formula the sliced pin falls back to, and the
    // DXGI shared budget it prefers. Both are what the sliced pin would use, so the probe says whether the answer
    // makes either conservative.
    const uint64_t ram = basalt::platform::total_physical_memory();
    if (ram > 0) p.formula_bytes = ram / 2 > (8ull << 30) ? ram / 2 - (8ull << 30) : 0;
    p.budget_bytes = basalt::platform::dxgi_non_local_budget(basalt::core::device_luid());   // 0 where nothing budgets it
    return p;
}

void print_memory_report(const ReportFacts& facts) {
    std::fprintf(stderr, "basalt memory-report:\n");
    // Every figure comes from the platform probe, so one line reads out the same way on both platforms: what the OS
    // has, what it can still hand out, and what it can still commit - the pair a page-locked arena must fit inside.
    // A probe that did not answer says so rather than printing zeros.
    const uint64_t phys_total = basalt::platform::total_physical_memory();
    if (phys_total == 0) {
        std::fprintf(stderr, "  host: unknown (the OS gave no memory reading)\n");
    } else {
        std::fprintf(stderr, "  host: total phys %llu MiB, avail phys %llu MiB, commit limit %llu MiB, commit headroom %llu MiB\n",
                     (unsigned long long) (phys_total >> 20),
                     (unsigned long long) (basalt::platform::available_memory() >> 20),
                     (unsigned long long) (basalt::platform::commit_limit() >> 20),
                     (unsigned long long) (basalt::platform::commit_headroom() >> 20));
    }
    const basalt::platform::ProcessMemory pmem = basalt::platform::process_memory();
    std::fprintf(stderr, "  process: resident %llu MiB, peak resident %llu MiB, private %llu MiB\n",
                 pmem.rss_mib, pmem.peak_rss_mib, pmem.private_mib);
    uint64_t ws_min = 0, ws_max = 0;
    if (basalt::platform::working_set_quota(ws_min, ws_max))
        std::fprintf(stderr, "  working set quota: min %llu MiB, max %llu MiB\n",
                     (unsigned long long) (ws_min >> 20), (unsigned long long) (ws_max >> 20));
    if (g_report_arena) {
        const char* backing = "unknown";
        switch (g_report_arena->backing) {
            case PageBacking::LargePages: backing = "large pages"; break;
            case PageBacking::NormalPages: backing = "normal pages"; break;
            case PageBacking::PinnedByCuda: backing = "pinned by cuda"; break;
        }
        if (g_report_arena->slice_bytes == 0) {
            std::fprintf(stderr, "  arena: capacity %llu MiB, backing %s, registered %llu MiB (whole), locked %llu MiB\n",
                         (unsigned long long) (g_report_arena->capacity >> 20), backing,
                         (unsigned long long) (g_report_arena->registered_bytes >> 20),
                         (unsigned long long) (g_report_arena->locked_bytes >> 20));
        } else {
            std::fprintf(stderr, "  arena: capacity %llu MiB, backing %s, registered %llu MiB (%d slices of %llu MiB), locked %llu MiB\n",
                         (unsigned long long) (g_report_arena->capacity >> 20), backing,
                         (unsigned long long) (g_report_arena->registered_bytes >> 20), g_report_arena->registered_slices,
                         (unsigned long long) (g_report_arena->slice_bytes >> 20),
                         (unsigned long long) (g_report_arena->locked_bytes >> 20));
        }
        std::fprintf(stderr, "  arena note: %s\n", g_report_arena->note.empty() ? "(none)" : g_report_arena->note.c_str());
    } else {
        std::fprintf(stderr, "  arena: none recorded\n");
    }
    // What the large-page pool would grant, once: a pool that declines an arena can still fund part of one. A
    // privilege the token does not hold is the session's ceiling rather than the pool's, so it is named as such.
    {
        const basalt::platform::LargePageProbe lp = large_page_probe_once();
        if (lp.minimum == 0) {
            std::fprintf(stderr, "  large pages: none on this system\n");
        } else if (lp.granted > 0) {
            std::fprintf(stderr, "  large pages: minimum %llu MiB, the pool grants %llu MiB of the %llu MiB requested"
                                 " (%llu of %llu addresses of the grant read back as large pages)\n",
                         (unsigned long long) (lp.minimum >> 20), (unsigned long long) (lp.granted >> 20),
                         (unsigned long long) (lp.requested >> 20), (unsigned long long) lp.large_pages,
                         (unsigned long long) lp.sampled);
        } else if (lp.error == 1314) {
            std::fprintf(stderr, "  large pages: minimum %llu MiB, the token does not hold the privilege (1314), so "
                                 "none of the %llu MiB requested was granted - the session's ceiling, not the pool's\n",
                         (unsigned long long) (lp.minimum >> 20), (unsigned long long) (lp.requested >> 20));
        } else {
            std::fprintf(stderr, "  large pages: minimum %llu MiB, the pool grants none of the %llu MiB requested "
                                 "(error %llu)\n", (unsigned long long) (lp.minimum >> 20),
                         (unsigned long long) (lp.requested >> 20), (unsigned long long) lp.error);
        }
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
        std::fprintf(stderr, "  vram: free %llu MiB / total %llu MiB (cudaMemGetInfo)\n",
                     (unsigned long long) (free_b >> 20), (unsigned long long) (total_b >> 20));
    else
        std::fprintf(stderr, "  vram: unknown (cudaMemGetInfo failed)\n");
    // The budget that matters is the one the process is charged against, so the query is matched to the device
    // being reported on. When the driver gives no id there is nothing to match, and the figures are the first
    // adapter's - said on the line rather than passed off as the device's. Where the OS budgets no video memory
    // at all, the query reports not-ok and every comparison below says so instead of guessing.
    const void* luid = basalt::core::device_luid();
    const basalt::platform::DxgiMemory dx = basalt::platform::print_dxgi_memory_report(luid, luid != nullptr);
    // Device memory is charged to the LOCAL segment and page-locked memory the GPU maps to the shared (NON_LOCAL)
    // one, so each plan is compared against its own budget and every verdict names the segment it is about.
    if (!facts.tier_known) {
        std::fprintf(stderr, "  tier: unknown (no expert-tier facts recorded)\n");
    } else if (facts.tier_bytes == 0) {
        std::fprintf(stderr, "  tier: none (no expert cache)\n");
    } else if (!dx.ok) {
        std::fprintf(stderr, "  tier: %llu MiB in %llu slots (full %llu MiB), charged to the LOCAL segment; the local "
                             "budget is unknown, so it cannot be compared\n",
                     (unsigned long long) (facts.tier_bytes >> 20), (unsigned long long) facts.tier_slots,
                     (unsigned long long) (facts.tier_full_bytes >> 20));
    } else if (dx.local_usage > dx.local_budget) {
        std::fprintf(stderr, "  tier: %llu MiB in %llu slots (full %llu MiB), charged to the LOCAL segment: usage is "
                             "over the local budget by %llu MiB, so the driver can place that much in system memory\n",
                     (unsigned long long) (facts.tier_bytes >> 20), (unsigned long long) facts.tier_slots,
                     (unsigned long long) (facts.tier_full_bytes >> 20),
                     (unsigned long long) ((dx.local_usage - dx.local_budget) >> 20));
    } else {
        std::fprintf(stderr, "  tier: %llu MiB in %llu slots (full %llu MiB), charged to the LOCAL segment: within the "
                             "local budget, %llu MiB of headroom left\n",
                     (unsigned long long) (facts.tier_bytes >> 20), (unsigned long long) facts.tier_slots,
                     (unsigned long long) (facts.tier_full_bytes >> 20),
                     (unsigned long long) ((dx.local_budget - dx.local_usage) >> 20));
    }
    if (g_report_arena != nullptr && g_report_arena->registered_bytes > 0) {
        if (!dx.ok) {
            std::fprintf(stderr, "  arena charge: %llu MiB registered, charged to the shared (NON_LOCAL) segment; that "
                                 "budget is unknown\n",
                         (unsigned long long) (g_report_arena->registered_bytes >> 20));
        } else if (dx.non_local_usage > dx.non_local_budget) {
            std::fprintf(stderr, "  arena charge: %llu MiB registered, charged to the shared (NON_LOCAL) segment: usage "
                                 "is over the non-local budget by %llu MiB\n",
                         (unsigned long long) (g_report_arena->registered_bytes >> 20),
                         (unsigned long long) ((dx.non_local_usage - dx.non_local_budget) >> 20));
        } else {
            std::fprintf(stderr, "  arena charge: %llu MiB registered, charged to the shared (NON_LOCAL) segment: within "
                                 "the non-local budget, %llu MiB of headroom left\n",
                         (unsigned long long) (g_report_arena->registered_bytes >> 20),
                         (unsigned long long) ((dx.non_local_budget - dx.non_local_usage) >> 20));
        }
    }
    // The file readers' queue: a rate on its own cannot tell a slow drive from a queue that never went deep.
    {
        const basalt::platform::IoStats io = basalt::platform::io_stats();
        if (io.issuing_threads == 0 && io.submitted == 0) {
            std::fprintf(stderr, "  io: no reader open\n");
        } else {
            std::fprintf(stderr, "  io: %d issuing threads, %llu queued, %llu in flight (peak %llu), %llu submitted, "
                                 "%llu completed (%llu failed), %llu MiB\n",
                         io.issuing_threads, (unsigned long long) io.queued, (unsigned long long) io.inflight,
                         (unsigned long long) io.peak_inflight, (unsigned long long) io.submitted,
                         (unsigned long long) io.completed, (unsigned long long) io.failed,
                         (unsigned long long) (io.bytes >> 20));
        }
    }
    if (!facts.ple_io.empty()) std::fprintf(stderr, "  io note: %s\n", facts.ple_io.c_str());
    // The two one-time probes. A probe that could not run says so: an unmeasured rate is not a zero.
    {
        const H2dProbe h2d = h2d_probe_once();
        if (h2d.ok) {
            std::fprintf(stderr, "  probe h2d: %.1f GiB/s from the registered arena in %llu MiB chunks (%llu MiB "
                                 "moved)\n", h2d.gib_per_s, (unsigned long long) (h2d.chunk >> 20),
                         (unsigned long long) (h2d.bytes >> 20));
        } else {
            std::fprintf(stderr, "  probe h2d: not measured (no registered arena)\n");
        }
    }
    // The small-read arms: the same file, size and count at two queue depths, measured once before any view of the
    // file existed and once with whatever this run maps. An arm that could not run says so rather than reading as 0.
    {
        auto arm = [](const ReportFacts::ReadArm& a) {
            for (const basalt::platform::ReadProbe* p : {&a.shallow, &a.deep}) {
                if (p->ok)
                    std::fprintf(stderr, "  probe reads (%s) %u B, depth %u: %.0f reads/s, %.1f MiB/s, %llu failed\n",
                                 a.note.c_str(), p->read_bytes, p->depth, p->reads_per_s(),
                                 p->gib_per_s() * 1024.0, (unsigned long long) p->failed);
                else
                    std::fprintf(stderr, "  probe reads (%s) depth %u: not measured%s%s\n", a.note.c_str(), p->depth,
                                 p->error.empty() ? "" : ": ", p->error.c_str());
            }
        };
        arm(facts.reads_before);
        arm(facts.reads_after);
        arm(facts.reads_held_view);
    }
    // What cudaHostRegister accepts here, measured before the arena was created: the sliced pin's cap is a formula
    // today, and this says whether the driver really allows that much.
    if (facts.host_register.ok) {
        const HostRegisterProbe& h = facts.host_register;
        std::fprintf(stderr, "  probe host register: %u slices of %llu MiB accepted = %llu MiB%s\n", h.slices,
                     (unsigned long long) (h.slice_bytes >> 20), (unsigned long long) (h.granted >> 20),
                     h.refused_at > 0 ? " (the driver refused the next one)" : "");
        if (h.refused_at > 0) std::fprintf(stderr, "  probe host register refusal: %s\n", h.error.c_str());
        if (h.formula_bytes > 0 || h.budget_bytes > 0) {
            // The margin matters more than the verdict: it is how much of the pinned ceiling the formula leaves unused.
            const bool more = h.granted >= h.formula_bytes;
            const uint64_t margin = more ? h.granted - h.formula_bytes : h.formula_bytes - h.granted;
            if (h.formula_bytes == 0)
                std::fprintf(stderr, "  probe host register vs the caps: the formula is not computable, DXGI shared "
                                     "budget %llu MiB\n", (unsigned long long) (h.budget_bytes >> 20));
            else
                std::fprintf(stderr, "  probe host register vs the caps: the driver allows %llu MiB %s the RAM/2-8GiB "
                                     "formula (%llu MiB), and the DXGI shared budget is %llu MiB\n",
                             (unsigned long long) (margin >> 20), more ? "more than" : "less than",
                             (unsigned long long) (h.formula_bytes >> 20),
                             (unsigned long long) (h.budget_bytes >> 20));
        }
    } else {
        std::fprintf(stderr, "  probe host register: not measured%s%s\n",
                     facts.host_register.error.empty() ? "" : ": ", facts.host_register.error.c_str());
    }
    // What the token says about the large-page privilege, and what the two documented allocation forms do with it.
    // A 1314 refusal has to be read against all of it: absent from the token, present but refused by the allocation,
    // and a filtered token are three different problems.
    {
        const basalt::platform::LargePagePrivilege lp = basalt::platform::probe_large_page_privilege();
        if (lp.ok) {
            std::fprintf(stderr, "  large-page privilege: looked up %s, present %s, enabled %s; token elevated %s, "
                                 "integrity 0x%llx, Administrators %s%s\n",
                         lp.looked_up ? "yes" : "no", lp.present ? "yes" : "no", lp.enabled ? "yes" : "no",
                         lp.elevated ? "yes" : "no", (unsigned long long) lp.integrity_rid,
                         lp.admin_group ? "present" : "absent",
                         lp.admin_deny_only ? " (deny-only: a filtered token)" : "");
            std::fprintf(stderr, "  large-page privilege allocations: one page reserve+commit %s (error %llu), reserve "
                                 "%s (error %llu), then commit large %s (error %llu)\n",
                         lp.single_ok ? "succeeded" : "FAILED", (unsigned long long) lp.single_error,
                         lp.reserve_ok ? "succeeded" : "FAILED", (unsigned long long) lp.reserve_error,
                         lp.commit_ok ? "succeeded" : "FAILED", (unsigned long long) lp.commit_error);
        } else {
            std::fprintf(stderr, "  large-page privilege: not read%s%s\n", lp.note.empty() ? "" : ": ", lp.note.c_str());
        }
    }
    // What the OS prefetch hint buys on a mapped view: the cost of faulting a window with and without it, and with
    // time for the hint to land. A rate near memory speed says the pages were cached, not read from the drive.
    if (facts.view_faults.ok) {
        const basalt::platform::ViewFaultProbe& f = facts.view_faults;
        const double off_ms = f.off_seconds * 1000.0, on_ms = f.on_seconds * 1000.0, early_ms = f.early_seconds * 1000.0;
        std::fprintf(stderr, "  probe view faults: %llu MiB x %u windows per arm: no hint %.1f ms (%.3f ms/MiB), "
                             "hint %.1f ms (%.3f ms/MiB), hint then %.0f ms of other work %.1f ms (%.3f ms/MiB); "
                             "hint/no-hint %.2fx, early/no-hint %.2fx\n",
                     (unsigned long long) (f.window_bytes >> 20), f.windows_per_arm, off_ms, f.off_ms_per_mib(), on_ms,
                     f.on_ms_per_mib(), f.wait_ms, early_ms, f.early_ms_per_mib(), off_ms > 0.0 ? on_ms / off_ms : 0.0,
                     off_ms > 0.0 ? early_ms / off_ms : 0.0);
        // The wait is real time, so the early arm is only a win if it survives it: the comparison that matters is the
        // wait plus the timed fault against the unhinted fault, not the two timed regions against each other.
        const double early_e2e = early_ms + f.wait_ms;
        std::fprintf(stderr, "  probe view faults end to end: unhinted %.1f ms, unhinted WITH the hint %.1f ms, "
                             "hinted early %.1f ms (%.1f ms of it the wait) = %.2fx against unhinted\n",
                     off_ms, on_ms, early_e2e, f.wait_ms, off_ms > 0.0 ? early_e2e / off_ms : 0.0);
        auto spread = [](double mean, double lo, double hi) { return mean > 0.0 && hi - lo > 0.2 * mean; };
        if (spread(f.off_seconds, f.off_min, f.off_max) || spread(f.on_seconds, f.on_min, f.on_max) ||
            spread(f.early_seconds, f.early_min, f.early_max))
            std::fprintf(stderr, "  probe view faults spread: no hint %.1f-%.1f ms, hint %.1f-%.1f ms, early %.1f-%.1f "
                                 "ms\n", f.off_min * 1000.0, f.off_max * 1000.0, f.on_min * 1000.0, f.on_max * 1000.0,
                         f.early_min * 1000.0, f.early_max * 1000.0);
    } else {
        std::fprintf(stderr, "  probe view faults: not measured%s%s\n",
                     facts.view_faults.error.empty() ? "" : ": ", facts.view_faults.error.c_str());
    }
}

}  // namespace basalt::core
