// include/basalt/platform/memory.hpp - keep a large host region resident.
//
// An unlocked expert arena is trimmed under memory pressure, which makes the CPU pool's speed vary between
// runs. With the n-gram table out of RAM there is headroom to lock it: mlock on Linux, the process working-set
// minimum plus VirtualLock on Windows.
#pragma once

#include <cstdint>
#include <string>

namespace basalt::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// Host memory whose address is a multiple of `alignment`, which has to be a power of two: a range registered with
/// the driver as a whole, or filled by reads that bypass the OS cache, needs its first byte on a boundary.
/// `free_aligned` releases what this returns, and only what this returns. nullptr when the memory is not there.
void* alloc_aligned(size_t bytes, size_t alignment);
/// The matching release for `alloc_aligned`. A null is fine.
void free_aligned(void* p);

/// Asks the OS to give back what it holds for [off, off + bytes) of a mapped file view that starts at `base` and is
/// `view_bytes` long: the range widened to whole pages, because those are what can be given back, and clipped to the
/// view. False when the OS refused. Windows has no per-range release for a view of a file mapping - its pages are
/// clean, since the view is read-only, so the manager takes them back on pressure by itself - and reports success
/// without claiming anything.
bool drop_mapped_view(const void* base, uint64_t view_bytes, uint64_t off, uint64_t bytes);

/// What backs a host region from `alloc_region`.
enum class RegionBacking { LargePages, NormalPages };

/// A host region of `bytes` for a large arena, with the best page size the OS gives it: 2 MiB huge pages (hugetlb on
/// Linux, large pages on Windows) first, then ordinary pages the OS is asked to back with huge ones where it can.
/// `got` says which, and `note` says what was done or why the better one was refused, for the startup print. The
/// bytes are zero-filled and committed (Windows) or demand-faulted (Linux). A non-empty `shared_file` instead maps one
/// file whose first 4 KiB identify the pack (`shared_hash`, which must not be 0) and whose remaining bytes are the
/// region - Linux only, meant for tmpfs; elsewhere it fails with a note. `mapping_base` and `mapping_bytes` are what
/// `free_region` releases: the region itself, or the file mapping that its header precedes; both are 0 when nothing
/// was allocated. nullptr on failure.
void* alloc_region(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                   uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes);
/// Releases what `alloc_region` returned through `mapping_base` and `mapping_bytes`. A null is fine.
void free_region(void* mapping_base, uint64_t mapping_bytes);

/// The most host memory the GPU may map page-locked before its shared-memory segment is full, when the OS limits it
/// beneath the driver's own refusal (`why` names where the figure comes from); false when it does not, and the driver's
/// refusal is the limit.  `luid` is the 8 bytes of the device's LUID when the caller has one, which picks the adapter
/// whose budget applies; without it the first adapter is asked.
bool pin_budget(uint64_t& limit, std::string& why, const void* luid = nullptr);

/// The machine's physical RAM in bytes (0 when unknown).
uint64_t total_physical_memory();

/// The RAM the OS can hand out without paging, in bytes (0 when unknown).
uint64_t available_memory();

/// What the OS could still commit, in bytes. A page-locked or fully reserved region is charged against RAM and the
/// page file together, so on Windows this is the free part of that pair - the reading a refused allocation names.
/// Linux keeps no separate charge, so there it is the available-RAM reading (0 when unknown).
uint64_t commit_headroom();

/// The whole of what the OS may commit, in bytes: RAM plus the page file on Windows, which is the ceiling a
/// page-locked arena has to fit under; the physical RAM on Linux, which keeps no separate charge (0 when unknown).
uint64_t commit_limit();

/// The working-set quota this process is held to, in bytes: the minimum the OS keeps resident and the maximum it may
/// take. Pages locked with `lock_resident` are charged against the minimum, so a refused lock is read against this.
/// False where the OS has no such quota.
bool working_set_quota(uint64_t& min_bytes, uint64_t& max_bytes);

/// The RAM left alone by anything that locks or fills memory: the larger of 2 GiB and 5% of the machine's RAM.
uint64_t ram_margin();

/// Host RAM the engine takes after the resident expert copy is sized and before it serves: the GPU libraries' own
/// host memory, the prompt path's and the verifier's staging.  A resident copy leaves this much beside the margin.
inline constexpr uint64_t kLateHostBytes = 2ull << 30;

/// Host RAM one running request allocates beside its checkpoints and images: token lists, the reply, the server
/// process's own buffers.
inline constexpr uint64_t kRequestHostBytes = 256ull << 20;

/// The expert bytes one prompt chunk may read from the expert file for the slots it borrows from the GPU cache (the
/// chunk is sized to stay within it).
inline constexpr uint64_t kPromptFileReadBytes = 3584ull << 20;

/// The least RAM kept for the OS file cache while any expert is read from the file.
inline constexpr uint64_t kMinFileCacheBytes = 4ull << 30;

/// The RAM the OS file cache needs beside ram_margin() while `file_bytes` of experts are read from the file through
/// it (0: every expert is held by a GPU tier or the RAM copy, nothing is read from the file).  A prompt chunk reads
/// up to kPromptFileReadBytes for its borrowed slots, plus the file-held experts of the layer it computes and of the
/// next one it reads ahead: at most two whole layers (`largest_layer_bytes`, every expert of the largest layer).
/// Pages evicted before that pass is over are read again, and the reclaim then runs at drive speed.
inline uint64_t file_cache_floor(uint64_t file_bytes, uint64_t largest_layer_bytes) {
    if (file_bytes == 0) return 0;
    const uint64_t pass = kPromptFileReadBytes + 2 * largest_layer_bytes;
    return pass > kMinFileCacheBytes ? pass : kMinFileCacheBytes;
}

/// The RAM left free once the experts are placed: `margin` (ram_margin()) plus the file cache's floor.
inline uint64_t ram_floor(uint64_t margin, uint64_t file_bytes, uint64_t largest_layer_bytes) {
    return margin + file_cache_floor(file_bytes, largest_layer_bytes);
}

/// Where a running request stops while serving: the margin plus one prompt chunk's file reads when experts are read
/// from the file (the margin alone when none are).  Below ram_floor() by the two layers' read-ahead, so the engine's
/// own growth after the load (conversation cache, an image's buffers) does not end requests on a machine still
/// sized as asked.
inline uint64_t ram_guard_floor(uint64_t margin, uint64_t file_bytes) {
    return margin + (file_bytes > 0 ? kPromptFileReadBytes : 0);
}

/// The largest RAM copy that leaves `floor` free, with `avail` bytes available and `late` bytes still to be taken by
/// the engine after the copy.
inline uint64_t max_safe_copy(uint64_t avail, uint64_t late, uint64_t floor) {
    return avail > late + floor ? avail - late - floor : 0;
}

/// The RAM a page-locked or fully resident allocation may take now: available_memory() less ram_margin().  `known`
/// is false when the OS gives no reading (nothing is refused then).
uint64_t ram_budget(bool* known = nullptr);

/// Polls available_memory() until `stop_memory_watchdog`.  Below `floor_bytes` it prints a `basalt error:` line and
/// _exit(1)s, so a load that is exhausting RAM ends before the machine starts to swap.
void start_memory_watchdog(uint64_t floor_bytes);
/// Tells the watchdog the load builds a resident RAM copy of the experts (`resident`) and how big it is so far, so
/// its error names the copy's cap instead of the resident mode.
void watch_resident_ram(bool resident, uint64_t copy_bytes);
void stop_memory_watchdog();

/// Whether an available-RAM reading of `avail` is below `floor` (a reading of 0, unknown, is not).
inline bool ram_below_floor(uint64_t avail, uint64_t floor) { return avail > 0 && avail < floor; }

/// While serving: polls available_memory() and keeps whether it is below `floor_bytes`, for ram_guard_low.  The request
/// that is running then stops with an error instead of the machine swapping; nothing is shrunk or exited.
void start_ram_guard(uint64_t floor_bytes);
/// True while the last reading was below the guard's floor; `avail` and `floor` get that reading and the floor.
bool ram_guard_low(uint64_t* avail = nullptr, uint64_t* floor = nullptr);

/// Whether the OS file cache could keep the `read_bytes` the expert file is read for, beside
/// `arena_bytes` of RAM held by the engine's own copy of the experts and `margin` for everything else, with `avail`
/// bytes of RAM available.  The file tier passes only the expert bytes it really reads from the file (the experts
/// outside its resident RAM copy) and the RAM that copy really holds.
inline bool file_cache_keeps(uint64_t avail, uint64_t arena_bytes, uint64_t read_bytes,
                             uint64_t margin = 4ull << 30) {
    const uint64_t room = avail > arena_bytes + margin ? avail - arena_bytes - margin : 0;
    return room >= read_bytes;
}

/// Whether `advise_willneed` asks the OS for anything: it does on both platforms.
bool read_ahead_enabled();

/// One adapter's video memory, per segment group, as the display driver reports it. This is the budget the driver
/// plans against: an allocation past the local one is placed in system memory instead of failing, so a plan that
/// ignores it cannot tell a resident tier from a spilled one. `ok` is false when the query could not run (no
/// adapter, no IDXGIAdapter3, or a platform without DXGI). `outputs` is -1 when the enumeration failed - which is
/// "unknown", not "no outputs": a session with no desktop cannot enumerate, and reading that as zero would size a
/// full budget on a machine that is driving a display.
struct DxgiMemory {
    bool ok = false;
    uint64_t local_budget = 0;
    uint64_t local_usage = 0;
    uint64_t non_local_budget = 0;
    uint64_t non_local_usage = 0;
    int outputs = -1;
};
/// The figures for the adapter whose LUID matches `luid` (the 8 bytes of `cudaDeviceProp::luid`), or for the first
/// adapter when `luid` is null.
DxgiMemory dxgi_memory(const void* luid = nullptr);

/// Print one adapter's video-memory budget and usage (both segment groups) and its output count, and return the same
/// figures so a caller can compare a plan against them. `luid_matched` says whether `luid` named the device the plan
/// is for, which is what the first line reports. Nothing prints on a platform that budgets no video memory.
DxgiMemory print_dxgi_memory_report(const void* luid = nullptr, bool luid_matched = false);
/// DXGI shared (non-local) memory budget in bytes for the adapter whose LUID matches `luid` (8 bytes),
/// or 0 when unknown/unavailable. NULL `luid` queries the first adapter.
uint64_t dxgi_non_local_budget(const void* luid);

/// Enable `SeLockMemoryPrivilege` in this process's token, which `MEM_LARGE_PAGES` needs. The account having the
/// right assigned is not enough: a privilege is absent from the token until the process asks for it. False when the
/// token could not be adjusted, which the callers report and fall back from.
bool enable_lock_memory_privilege();

/// How much of a large-page request the system's pool actually grants.
struct LargePageProbe {
    uint64_t minimum = 0;    ///< GetLargePageMinimum(): 0 when this system has no large pages at all
    uint64_t requested = 0;  ///< the largest size asked for
    uint64_t granted = 0;    ///< the largest size the pool granted, released again before returning
    uint64_t error = 0;      ///< the OS error of the first refusal (0 when `requested` was granted)
};
/// Ask the pool for `want` bytes of large pages, halving down to a floor, to find the largest block it really
/// grants: a pool that declines a whole arena may still fund part of one. Every grant is released before the
/// return, so this only measures. It is a probe, not an allocation: the answer describes this moment.
LargePageProbe probe_large_pages(uint64_t want);

/// Asks the OS to start bringing [p, p + bytes) of a mapped file view into memory, without waiting for the access.
/// A hint, never an error: the system skips what it will not prefetch and says nothing about it, and without the
/// hint the pages still arrive - one fault at a time, where they are touched.  The range is asked for in 128 KiB
/// windows, which keeps the reads in flight bounded.  Only a mapped view is reached this way.
void advise_willneed(const void* p, uint64_t bytes);
/// The same for [offset, offset + bytes) of an open descriptor.  Linux asks its page cache.  Windows asks nothing:
/// the cache manager takes no range hint for a handle, and a read through a descriptor is a cached read unless the
/// caller opted out of that cache - which already reads ahead around a sequential reader.  Bytes reached only
/// through a descriptor are therefore warmed as they are read, not ahead of it.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);

}  // namespace basalt::platform
