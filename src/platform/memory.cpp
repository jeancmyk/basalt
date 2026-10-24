// src/platform/memory.cpp - see include/basalt/platform/memory.hpp.
#include "basalt/platform/memory.hpp"

#include "basalt/platform/file.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#include <malloc.h>   // _aligned_malloc, _aligned_free
#include <dxgi1_4.h>  // IDXGIAdapter3::QueryVideoMemoryInfo
#include <psapi.h>    // QueryWorkingSetEx: the large-page flag of a mapping's own pages
#else
#include <fcntl.h>
#include <linux/mman.h>   // MAP_HUGE_2MB
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <cerrno>
#include <cstring>
#include "basalt/log.hpp"

namespace basalt::platform {

#if defined(_WIN32)
namespace {
/// Enables one privilege in this process's token. A privilege can be assigned to the account and still be absent
/// from the token until the process asks for it, which is the normal state for an interactive one. False when the
/// token could not be adjusted.
///
/// The error is cleared BEFORE the call: AdjustTokenPrivileges sets it only for ERROR_NOT_ALL_ASSIGNED, so on full
/// success the previous value stands and a stale error would make a working privilege look refused.
bool enable_privilege(LPCWSTR name) {
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return false;
    bool ok = false;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    if (LookupPrivilegeValueW(nullptr, name, &tp.Privileges[0].Luid)) {
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        SetLastError(ERROR_SUCCESS);
        if (AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) != 0)
            ok = GetLastError() == ERROR_SUCCESS;   // ERROR_NOT_ALL_ASSIGNED when the account has no such right
    }
    CloseHandle(tok);
    return ok;
}

/// What a committed range's own pages say about it.  A page reports its backing only once it is resident, so this
/// writes a zero byte at a bounded number of addresses spread over the range - the range is committed zeroed, so the
/// write cannot change it - and reads the flags back.  `sampled` counts the addresses the query answered for (0 when
/// it could not run, which is unknown rather than a refusal) and `large` those it flagged as large pages.
struct PageBacking {
    uint64_t sampled = 0, large = 0;
};
PageBacking page_backing(void* p, uint64_t bytes, size_t samples = 16) {
    PageBacking r;
    constexpr uint64_t kPage = 4096;
    if (p == nullptr || bytes < kPage || samples == 0) return r;
    uint64_t stride = bytes / samples;
    stride = (stride + kPage - 1) / kPage * kPage;
    if (stride == 0) stride = kPage;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> info;
    for (uint64_t off = 0; off < bytes; off += stride) {
        uint8_t* const at = static_cast<uint8_t*>(p) + off;
        *at = 0;   // fault the page in, then ask the OS what backs it
        info.push_back(PSAPI_WORKING_SET_EX_INFORMATION{at});
    }
    if (!QueryWorkingSetEx(GetCurrentProcess(), info.data(), (DWORD) (info.size() * sizeof info[0]))) return r;
    for (const PSAPI_WORKING_SET_EX_INFORMATION& e : info) {
        if (!e.VirtualAttributes.Valid) continue;
        ++r.sampled;
        if (e.VirtualAttributes.LargePage) ++r.large;
    }
    return r;
}
}  // namespace
#endif

LockResult lock_resident(void* p, uint64_t bytes, const LockProgress& progress) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
#if defined(_WIN32)
    enable_privilege(L"SeIncreaseWorkingSetPrivilege");   // to raise this process's working set
    enable_privilege(L"SeLockMemoryPrivilege");           // and to lock pages in it
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    // Locked pages count against the minimum working set, so it has to grow by the region plus room for the rest of
    // the process first: VirtualLock can lock no more pages than that minimum holds. A quota that cannot be raised
    // is reported and the lock tried anyway: whatever the existing quota covers is still worth having.
    std::string quota_note;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        quota_note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
    } else {
        const SIZE_T margin = (SIZE_T) 512 << 20;
        const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
        const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
        // The minimum goes in as the hard limit: only a hard minimum keeps the locked pages from being trimmed back
        // out. The maximum stays soft, so the process can still grow past it.
        if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                        QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE))
            quota_note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                         " MiB) failed (error " + std::to_string(GetLastError()) + ")";
    }
    // Chunked, because one call over the whole region loses everything past the first page that does not fit.
    const uint64_t chunk = progress ? 256ull << 20 : 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = (quota_note.empty() ? "" : quota_note + "; ") + "VirtualLock stopped at " +
                     std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;   // what it did lock stays locked; the rest of the region is merely pageable
            return r;
        }
        r.locked_bytes = off + n;
        if (progress) progress(r.locked_bytes);
    }
    r.ok = true;
    const std::string locked = std::to_string((unsigned long long) (bytes >> 20)) + " MiB";
    r.note = quota_note.empty() ? "locked " + locked + " via working-set minimum + VirtualLock"
                                : quota_note + "; locked " + locked + " with the quota already in place";
    return r;
#else
    if (!progress) {
        if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
        r.ok = true;
        r.locked_bytes = bytes;
        r.note = "mlock";
        return r;
    }
    const uint64_t chunk = 256ull << 20;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (mlock((uint8_t*) p + off, (size_t) n) != 0) {
            r.note = "mlock failed (raise ulimit -l)";
            r.ok = off > 0;   // what it did lock stays locked; the rest of the region is merely pageable
            return r;
        }
        r.locked_bytes = off + n;
        progress(r.locked_bytes);
    }
    r.ok = true;
    r.note = "mlock";
    return r;
#endif
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
#if defined(_WIN32)
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
#else
    munlock(p, bytes);
#endif
}

uint64_t total_physical_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
#endif
}

#if defined(_WIN32)
namespace {
/// The two figures the sizing reads, from one query: available and total physical RAM in bytes. Both are 0 when the
/// query fails, which the callers read as unknown rather than as none.
void memory_status(uint64_t& avail, uint64_t& total) {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) { avail = 0; total = 0; return; }
    avail = (uint64_t) ms.ullAvailPhys;
    total = (uint64_t) ms.ullTotalPhys;
}
}  // namespace

uint64_t available_memory() {
    uint64_t avail = 0, total = 0;
    memory_status(avail, total);
    return avail;
}

uint64_t ram_margin() {
    uint64_t avail = 0, total = 0;
    memory_status(avail, total);
    return std::max<uint64_t>(2ull << 30, total / 20);
}
#else
namespace {
// One "<key>: <n> kB" line of /proc/meminfo in bytes (0 when absent).
uint64_t meminfo_bytes(const char* key) {
    std::ifstream in("/proc/meminfo");
    std::string line;
    const std::string k = std::string(key) + ":";
    while (std::getline(in, line))
        if (line.compare(0, k.size(), k) == 0) return std::strtoull(line.c_str() + k.size(), nullptr, 10) * 1024;
    return 0;
}
}  // namespace

uint64_t available_memory() { return meminfo_bytes("MemAvailable"); }

uint64_t ram_margin() {
    return std::max<uint64_t>(2ull << 30, meminfo_bytes("MemTotal") / 20);
}
#endif

uint64_t commit_headroom() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullAvailPageFile : 0;
#else
    return available_memory();
#endif
}

uint64_t commit_limit() {
#if defined(_WIN32)
    // RAM plus the page file: the pair a page-locked or fully reserved arena has to fit inside.
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPageFile : 0;
#else
    return total_physical_memory();
#endif
}

bool working_set_quota(uint64_t& min_bytes, uint64_t& max_bytes) {
#if defined(_WIN32)
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &min_ws, &max_ws, &flags)) return false;
    min_bytes = (uint64_t) min_ws;
    max_bytes = (uint64_t) max_ws;
    return true;
#else
    // Linux sets no per-process working-set quota; the page-lock limit itself is what binds.
    (void) min_bytes;
    (void) max_bytes;
    return false;
#endif
}

uint64_t ram_budget(bool* known) {
    const uint64_t avail = available_memory();
    if (known) *known = avail > 0;
    if (avail == 0) return UINT64_MAX;
    const uint64_t margin = ram_margin();
    return avail > margin ? avail - margin : 0;
}

namespace {
std::atomic<bool> g_watch_stop{false};
std::atomic<bool> g_watch_on{false};
std::atomic<bool> g_resident_mode{false};
std::atomic<uint64_t> g_resident_copy{0};
}  // namespace

std::string ram_exhausted_message(uint64_t avail, bool resident, uint64_t copy_bytes, uint64_t margin) {
    const unsigned long long mib = (unsigned long long) (avail >> 20);
    char buf[384];
    if (!resident) {
        std::snprintf(buf, sizeof buf, "basalt error: RAM ran out loading (%llu MiB free); use "
                                       "--resident-experts or --mmap-experts", mib);
    } else if (copy_bytes < (1ull << 30)) {
        // a budget of 1 GiB or more would not shrink a copy this small, so it is not offered
        std::snprintf(buf, sizeof buf, "basalt error: RAM ran out loading (%llu MiB free); the resident copy is only "
                                       "%llu MiB, so something else is holding the memory (other programs, or file "
                                       "pages read while loading)", mib, (unsigned long long) (copy_bytes >> 20));
    } else {
        // the copy shrunk by what the margin is short of leaves the margin free again
        const uint64_t keep = copy_bytes - std::min(copy_bytes, margin > avail ? margin - avail : 0);
        std::snprintf(buf, sizeof buf, "basalt error: RAM ran out loading (%llu MiB free); use --resident-budget-gib %llu",
                      mib, (unsigned long long) std::max<uint64_t>(1, keep >> 30));
    }
    return buf;
}

void watch_resident_ram(bool resident, uint64_t copy_bytes) {
    g_resident_mode = resident;
    g_resident_copy = copy_bytes;
}

void start_memory_watchdog(uint64_t floor_bytes) {
    if (g_watch_on.exchange(true)) return;
    g_watch_stop = false;
    std::thread([floor_bytes] {
        while (!g_watch_stop.load()) {
            const uint64_t avail = available_memory();
            if (avail > 0 && avail < floor_bytes) {
                basalt::log::error("basalt", "%s", ram_exhausted_message(avail, g_resident_mode.load(), g_resident_copy.load(),
                                                                   ram_margin()).c_str());
                _exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();   // never joined: the process may end while it sleeps
}

void stop_memory_watchdog() {
    g_watch_stop = true;
    g_watch_on = false;
}

namespace {
std::atomic<bool> g_guard_on{false};
std::atomic<bool> g_guard_low{false};
std::atomic<uint64_t> g_guard_avail{0};
std::atomic<uint64_t> g_guard_floor{0};
#if defined(_WIN32)
constexpr uint64_t kReclaimLead = 1ull << 30;       // how far above the floor the guard gives the view back
constexpr uint64_t kReclaimChunk = 256ull << 20;    // bytes given back per lock hold
std::mutex g_reclaim_mu;                            // held per chunk, so a withdrawn view is not touched afterwards
const void* g_reclaim_base = nullptr;
uint64_t g_reclaim_bytes = 0;

// Gives the registered view back in chunks, releasing the lock between them so a withdrawal waits for one chunk.
void reclaim_view_now() {
    const void* base;
    uint64_t bytes;
    {
        std::lock_guard<std::mutex> lock(g_reclaim_mu);
        base = g_reclaim_base;
        bytes = g_reclaim_bytes;
    }
    for (uint64_t off = 0; base != nullptr && off < bytes; off += kReclaimChunk) {
        std::lock_guard<std::mutex> lock(g_reclaim_mu);
        if (g_reclaim_base != base) return;
        (void) drop_mapped_view(base, bytes, off, std::min(kReclaimChunk, bytes - off));
    }
}
#endif
}  // namespace

void ram_guard_reclaim_view(const void* base, uint64_t bytes) {
#if defined(_WIN32)
    std::lock_guard<std::mutex> lock(g_reclaim_mu);
    g_reclaim_base = bytes > 0 ? base : nullptr;
    g_reclaim_bytes = g_reclaim_base != nullptr ? bytes : 0;
#else
    (void) base, (void) bytes;   // MemAvailable already counts a view's pages
#endif
}

void start_ram_guard(uint64_t floor_bytes) {
    g_guard_floor = floor_bytes;
    if (g_guard_on.exchange(true)) return;
    std::thread([] {
        bool was_low = false;
#if defined(_WIN32)
        bool logged_reclaim = false;
        uint64_t avail_after = 0;   // the reading right after the last give-back
        auto last_reclaim = std::chrono::steady_clock::now();
#endif
        for (;;) {
            uint64_t avail = available_memory();
            const uint64_t floor = g_guard_floor.load();
#if defined(_WIN32)
            // A later give-back waits for the view to be read in again, so a long prompt read keeps it in check.
            const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - last_reclaim).count();
            if (reclaim_due(avail, floor, kReclaimLead, avail_after, (uint64_t) since)) {
                bool have;
                {
                    std::lock_guard<std::mutex> lock(g_reclaim_mu);
                    have = g_reclaim_base != nullptr;
                }
                if (have) {
                    if (!logged_reclaim) {
                        basalt::log::info("basalt", "RAM guard: mapped experts view given back at %llu MiB free",
                                     (unsigned long long) (avail >> 20));
                        logged_reclaim = true;
                    }
                    reclaim_view_now();
                    avail = available_memory();
                    avail_after = avail;
                    last_reclaim = std::chrono::steady_clock::now();
                }
            }
#endif
            const bool low = ram_below_floor(avail, floor);
            g_guard_avail = avail;
            g_guard_low = low;
            if (low && !was_low) {
                basalt::log::warn("basalt", "RAM low: %llu MiB free < %llu MiB guard",
                             (unsigned long long) (avail >> 20), (unsigned long long) (floor >> 20));
            }
            was_low = low;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();   // runs for the life of the process
}

bool ram_guard_low(uint64_t* avail, uint64_t* floor) {
    if (avail) *avail = g_guard_avail.load();
    if (floor) *floor = g_guard_floor.load();
    return g_guard_low.load(std::memory_order_relaxed);
}

#if defined(_WIN32)
namespace {
constexpr uint64_t kPrefetchStep = 128ull << 10;

/// The page size, so a prefetch window starts on a page boundary.
uintptr_t page_bytes() {
    static const uintptr_t ps = [] {
        SYSTEM_INFO si{};
        GetSystemInfo(&si);
        return si.dwPageSize > 0 ? (uintptr_t) si.dwPageSize : (uintptr_t) 4096;
    }();
    return ps;
}
}  // namespace

bool read_ahead_enabled() { return true; }

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    // A view of a file mapping takes its pages one fault at a time as they are touched, so asking for the range
    // ahead of the access replaces those faults with large background reads.  The hint is best effort: a range the
    // system will not serve is skipped without an error, and pages already in memory cost nothing.  The range is
    // asked for in bounded windows, which keeps the reads in flight - and the memory they hold - bounded instead of
    // asking for a whole layer at once.
    const uintptr_t pg = page_bytes(), end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kPrefetchStep) {
        WIN32_MEMORY_RANGE_ENTRY entry{};
        entry.VirtualAddress = (PVOID) a;
        entry.NumberOfBytes = (SIZE_T) std::min<uintptr_t>(kPrefetchStep, end - a);
        (void) PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
    }
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    // No descriptor-range hint on Windows: the cache manager takes no "will need" for an open handle.  Reads through
    // a descriptor are cached reads unless the caller opted out of the cache, and the cache manager already reads
    // ahead around the position a sequential reader is at.  Warming a range by reading it into a scratch buffer
    // would move the bytes twice and cost the file cache the room, so nothing is asked.  A caller holding a mapped
    // view of the same file gets the prefetch from the overload above.
    (void) fd; (void) offset; (void) bytes;
}
#else
namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() { return true; }

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
}
#endif

void* alloc_aligned(size_t bytes, size_t alignment) {
    if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) return nullptr;
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, bytes) != 0) return nullptr;
    return p;
#endif
}

void free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

bool drop_mapped_view(const void* base, uint64_t view_bytes, uint64_t off, uint64_t bytes) {
    if (base == nullptr || bytes == 0) return true;
#if defined(_WIN32)
    // Unlocking a range that is not locked takes its pages out of this process's working set, which is the only
    // per-range release a view has: a clean page then waits on the standby list, counted as available, and is read
    // back from the file if touched again. The call reports ERROR_NOT_LOCKED for pages that were never locked even
    // though they were released, so its result carries nothing and a range the OS declines is simply left alone.
    if (off > view_bytes || bytes > view_bytes - off) return false;
    const uintptr_t pg = page_bytes();
    const uintptr_t first = ((uintptr_t) base + (uintptr_t) off) & ~(pg - 1);
    const uintptr_t last = ((uintptr_t) base + (uintptr_t) (off + bytes) + pg - 1) & ~(pg - 1);
    constexpr uintptr_t kStep = (uintptr_t) 1 << 30;
    for (uintptr_t a = first; a < last; a += kStep)
        (void) VirtualUnlock((LPVOID) a, (SIZE_T) std::min<uintptr_t>(kStep, last - a));
    return true;
#else
    // Only whole pages can be given back: a page the range shares with a neighbour still holds bytes the caller may
    // read. So the range is widened to the page boundaries and then clipped to the view.
    const long ps = sysconf(_SC_PAGE_SIZE);
    if (ps <= 0) return false;
    const uint64_t page = (uint64_t) ps;
    const uint64_t end = off + bytes;
    if (end > view_bytes) return false;
    const uint64_t start = off - off % page;
    const uint64_t remainder = end % page;
    const uint64_t extra = remainder == 0 ? 0 : page - remainder;
    const uint64_t wide = extra > view_bytes - end ? view_bytes : end + extra;
    if (wide <= start) return true;
    return madvise((void*) ((const uint8_t*) base + (size_t) start), (size_t) (wide - start), MADV_DONTNEED) == 0;
#endif
}


// ---------------------------------------------------------------------------------------------------- host regions

#if defined(_WIN32)
namespace {
// A size in GiB, for the notes that have to say how big something was.
std::string gib_size(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f GiB", (double) bytes / (double) (1ull << 30));
    return buf;
}

// What the OS could still commit, for the note that has to say why the arena did not fit.  The commit charge is the
// sum of RAM and the page file, so its free part is the headroom the arena has to fit into.  Empty when the OS
// gives no reading.
std::string commit_headroom_note() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return "; the commit headroom is unknown";
    return "; " + gib_size((uint64_t) ms.ullAvailPageFile) + " of commit charge free out of " +
           gib_size((uint64_t) ms.ullTotalPageFile) + " (RAM plus page file), " +
           gib_size((uint64_t) ms.ullTotalPhys) + " of RAM";
}
}  // namespace


static void* alloc_region_raw(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                              uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;
    if (!shared_file.empty()) {
        note = "shared arena files are not available on Windows";
        return nullptr;
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege, and the account having it assigned is not enough: the process
    // enables it in its own token first. An account without the assignment fails here and falls through to the
    // ordinary allocation below, which is the expected outcome rather than an error. The large-page probe enables it
    // through the same call, so the two cannot drift apart.
    (void) enable_lock_memory_privilege();
    const SIZE_T large = GetLargePageMinimum();
    if (large > 0) {
        // The requested size must be an exact multiple of the large-page minimum; the slack is under one page.
        const uint64_t lp_bytes = ((uint64_t) bytes + (uint64_t) large - 1) / (uint64_t) large * (uint64_t) large;
        // ONE CALL, RESERVATION AND COMMIT TOGETHER.  MEM_LARGE_PAGES is honoured only when the reservation and the
        // commit are the same call: committing a subrange of an existing reservation with the flag is ACCEPTED and
        // hands back 4 KB pages, which is why the reserve-then-chunked-commit shape this used to have reported a
        // large-page arena that never was one.  Whether a smaller pool would have fitted is a separate question, and
        // the large-page probe answers it.
        if (void* p = VirtualAlloc(nullptr, (SIZE_T) lp_bytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                   PAGE_READWRITE)) {
            // Read the backing off the mapping rather than trust the flag: an accepted call that gave small pages is
            // the failure mode above, and it must not be reported as large pages.  A query that answered nothing is
            // unknown rather than a refusal, so the accepted form stands in that case.
            const PageBacking g = page_backing(p, lp_bytes);
            if (g.sampled == 0 || g.large > 0) {
                got = RegionBacking::LargePages;
                note = "large pages (" + std::to_string((unsigned long long) large) + " B), one reserve-and-commit of " +
                       std::to_string(lp_bytes >> 20) + " MiB, " +
                       (g.sampled > 0 ? std::to_string(g.large) + " of " + std::to_string(g.sampled) +
                                         " sampled addresses flagged large"
                                      : "the working-set query read no address back");
                return p;
            }
            VirtualFree(p, 0, MEM_RELEASE);
            note = "large pages accepted but not given (" + std::to_string(g.sampled) +
                   " addresses read back from the mapping, none flagged large); using 4 KB pages";
        } else {
            // ERROR_NO_SYSTEM_RESOURCES is the page pool declining, ERROR_PRIVILEGE_NOT_HELD is the token and
            // ERROR_INVALID_PARAMETER a size that is not a multiple of the minimum. They need different fixes, so the
            // note carries the number; the probe's granted figure says whether the pool is merely smaller than this.
            note = "large pages refused for " + gib_size(lp_bytes) + " (GetLargePageMinimum=" +
                   std::to_string((unsigned long long) large) + " B, VirtualAlloc error " +
                   std::to_string((unsigned long long) GetLastError()) + "); the probe says what the pool would grant";
        }
    } else {
        note = "this system has no large-page minimum; using 4 KB pages";
    }
    {
        // Reserve first, then commit in chunks.  A single MEM_COMMIT over the whole arena only ever says that the
        // allocation failed, and whether it can succeed depends on RAM plus page file together, which is not known
        // before the commit reaches that far.  Committing a chunk at a time makes the failure name the size it
        // reached, and the region is released before anything is returned, so a half-committed arena is never
        // handed to the caller and nothing is reported as pinned that was not.
        void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE, PAGE_READWRITE);
        if (p == nullptr) {
            got = RegionBacking::NormalPages;
            note = "cannot reserve " + gib_size(bytes) + " of arena (VirtualAlloc error " +
                   std::to_string((unsigned long long) GetLastError()) + ")" + commit_headroom_note() +
                   "; the arena needs RAM plus page file; " + note;
            return nullptr;
        }
        constexpr uint64_t kCommitChunk = 1ull << 30;
        for (uint64_t off = 0; off < bytes; off += kCommitChunk) {
            const uint64_t n = bytes - off < kCommitChunk ? bytes - off : kCommitChunk;
            if (VirtualAlloc((uint8_t*) p + off, (SIZE_T) n, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
                const DWORD err = GetLastError();
                VirtualFree(p, 0, MEM_RELEASE);
                got = RegionBacking::NormalPages;
                note = "cannot commit the arena: " + gib_size(off) + " of " + gib_size(bytes) +
                       " committed (VirtualAlloc error " + std::to_string((unsigned long long) err) + ")" +
                       commit_headroom_note() + "; the arena needs RAM plus page file; " + note;
                return nullptr;
            }
        }
        got = RegionBacking::NormalPages;
        return p;
    }
}

void free_region(void* p, uint64_t bytes) {
    if (!p) return;
    (void) bytes;   // one VirtualAlloc owns the whole region
    VirtualFree(p, 0, MEM_RELEASE);
}

bool pin_budget(uint64_t& limit, std::string& why, const void* luid) {
    constexpr uint64_t GiB = 1ull << 30;
    // Page-locked memory the GPU maps is charged to the device's shared (non-local) segment. Registering slice by
    // slice until the driver refuses leaves that segment full and every later device allocation fails "out of
    // memory", so the caller stops at the segment's budget instead, keeping room for what it allocates after.
    constexpr uint64_t kHeadroom = 8ull * GiB;
    char buf[192];
    const uint64_t budget = dxgi_non_local_budget(luid);
    if (budget != 0) {
        if (budget > kHeadroom) {
            limit = budget - kHeadroom;
            std::snprintf(buf, sizeof buf, "the GPU's shared-memory budget, %.1f GiB via DXGI (minus %.0f GiB headroom)",
                          (double) budget / (double) GiB, (double) kHeadroom / (double) GiB);
        } else {
            limit = budget;
            std::snprintf(buf, sizeof buf, "the GPU's shared-memory budget, %.1f GiB via DXGI (no headroom taken)",
                          (double) budget / (double) GiB);
        }
        why = buf;
        return true;
    }
    // DXGI did not answer. The shared segment is about half of installed RAM, so the RAM reading bounds it.
    const uint64_t ram = basalt::platform::total_physical_memory();
    if (ram == 0) return false;
    limit = ram / 2 > 8 * GiB ? ram / 2 - 8 * GiB : 0;
    std::snprintf(buf, sizeof buf, "the GPU's shared-memory budget, RAM/2 - 8 GiB of %.1f GiB RAM (DXGI unavailable)",
                  (double) ram / (double) GiB);
    why = buf;
    return true;
}

#else
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif

namespace {
constexpr uint64_t kSharedArenaHeaderBytes = 4096;
constexpr char kSharedArenaMagic[16] = "BASALT-ARENA-V1";

struct SharedArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t arena_hash;
    uint64_t reserved[4];
};
static_assert(sizeof(SharedArenaHeader) <= kSharedArenaHeaderBytes);
}  // namespace

static void* alloc_region_raw(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                              uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;
    if (!shared_file.empty()) {
        if (shared_hash == 0) {
            note = "shared arena requires a nonzero model hash";
            return nullptr;
        }
        if (bytes > UINT64_MAX - kSharedArenaHeaderBytes) {
            note = "shared arena size overflows its header";
            return nullptr;
        }
        const uint64_t file_bytes = kSharedArenaHeaderBytes + bytes;
        const int fd = open(shared_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            note = "cannot open shared arena " + shared_file + ": " + std::strerror(errno);
            return nullptr;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            const int e = errno;
            close(fd);
            note = "cannot stat shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }

        const bool fresh = st.st_size == 0;
        if (fresh) {
            if (ftruncate(fd, (off_t) file_bytes) != 0) {
                const int e = errno;
                close(fd);
                note = "cannot size shared arena " + shared_file + ": " + std::strerror(e);
                return nullptr;
            }
            SharedArenaHeader hdr{};
            std::memcpy(hdr.magic, kSharedArenaMagic, sizeof hdr.magic);
            hdr.version = 1;
            hdr.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
            hdr.arena_bytes = bytes;
            hdr.arena_hash = shared_hash;
            const ssize_t written = ::pwrite(fd, &hdr, sizeof hdr, 0);
            if (written != (ssize_t) sizeof hdr) {
                const int e = errno;
                close(fd);
                note = "cannot write shared arena header " + shared_file + ": " +
                       (written < 0 ? std::string(std::strerror(e)) : std::string("short write"));
                return nullptr;
            }
        } else {
            if (st.st_size < 0 || (uint64_t) st.st_size != file_bytes) {
                close(fd);
                note = "shared arena " + shared_file + " is " +
                       std::to_string((unsigned long long) st.st_size) + " B, expected " +
                       std::to_string((unsigned long long) file_bytes) + " B including its header";
                return nullptr;
            }
            SharedArenaHeader hdr{};
            const ssize_t got_header = ::pread(fd, &hdr, sizeof hdr, 0);
            if (got_header != (ssize_t) sizeof hdr ||
                std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 ||
                hdr.version != 1 || hdr.header_bytes != kSharedArenaHeaderBytes || hdr.arena_bytes != bytes) {
                close(fd);
                note = "shared arena " + shared_file + " has an incompatible or missing header";
                return nullptr;
            }
            if (hdr.arena_hash != shared_hash) {
                close(fd);
                char b[256];
                std::snprintf(b, sizeof b,
                              "shared arena %s was written for model hash %016llx, expected %016llx",
                              shared_file.c_str(), (unsigned long long) hdr.arena_hash,
                              (unsigned long long) shared_hash);
                note = b;
                return nullptr;
            }
        }

        void* map = mmap(nullptr, (size_t) file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int e = errno;
        close(fd);
        if (map == MAP_FAILED) {
            note = "cannot map shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }
        mapping_base = map;
        mapping_bytes = file_bytes;
        got = RegionBacking::NormalPages;
        note = "MAP_SHARED file " + shared_file + " (model hash checked)";
        return (uint8_t*) map + kSharedArenaHeaderBytes;
    }

    {
        // CHUNKED, NOT ALL-OR-NOTHING.  One mmap(MAP_HUGETLB) over the whole arena fails exactly like an absent
        // pool when the pool is merely smaller than the arena, and it cannot say how far a partial pool would have
        // got.  The arena has to stay ONE contiguous range (layer bounds index into it, the registration covers it
        // whole, the destructor unmaps it in one call), so the range is reserved first and then filled piece by
        // piece with MAP_FIXED into that reservation.  The chunk matches the Windows commit chunk: tens of syscalls
        // for an arena of this size, and a multiple of the 2 MiB page.
        // Linux needs two details Windows does not: the reservation is PROT_NONE and the pieces are MAP_FIXED over
        // it, and the base is aligned up to 2 MiB by hand with a page of padding, because the kernel only picks a
        // huge-page-aligned address itself when addr is NULL.  Without that alignment MAP_FIXED fails EINVAL on a
        // misaligned base and the run would look like a pool refusal.
        constexpr uint64_t kChunk = 1ull << 30;
        const uint64_t page = 1ull << 21;
        const uint64_t huge_bytes = (bytes + page - 1) / page * page;
        const uint64_t padded = huge_bytes + page;
        void* raw = mmap(nullptr, (size_t) padded, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw != MAP_FAILED) {
            void* reserved = (void*) (((uintptr_t) raw + page - 1) & ~(uintptr_t) (page - 1));
            uint64_t off = 0, granted = 0;
            int first_errno = 0;
            for (; off < huge_bytes; off += kChunk) {
                const uint64_t n = huge_bytes - off < kChunk ? huge_bytes - off : kChunk;
                void* piece = mmap((uint8_t*) reserved + off, (size_t) n, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB | MAP_FIXED, -1, 0);
                if (piece == MAP_FAILED) {
                    first_errno = errno;
                    break;
                }
                granted = off + n;
            }
            if (off >= huge_bytes) {
                got = RegionBacking::LargePages;
                note = "hugetlb 2 MB pages, " + std::to_string((huge_bytes + kChunk - 1) / kChunk) +
                       " chunks of up to " + std::to_string(kChunk >> 20) + " MiB";
                // the whole padded reservation, so the one unmap releases the slack too
                mapping_base = raw;
                mapping_bytes = padded;
                return reserved;
            }
            // ALL OR NOTHING, as on Windows: a half-converted arena would need a third backing state, so the whole
            // reservation goes back.  One unmap of the raw span covers the fixed pieces and the untouched
            // reservation alike, because the MAP_FIXED calls replaced subranges of it.
            munmap(raw, (size_t) padded);
            // Name the shortfall, and now also where the loop stopped: a pool that covers part of the arena and one
            // that will not grant a single page are different failures.
            const unsigned long long need = (unsigned long long) (huge_bytes / page);
            unsigned long long pool = 0;
            bool have_pool = false;
            if (std::FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r")) {
                have_pool = std::fscanf(f, "%llu", &pool) == 1;
                std::fclose(f);
            }
            note = "MAP_HUGETLB refused at " + std::to_string(off >> 20) + " MiB of " +
                   std::to_string(huge_bytes >> 20) + " MiB (granted " + std::to_string(granted >> 20) +
                   " MiB in " + std::to_string((off + kChunk - 1) / kChunk) + " chunks, errno " +
                   std::to_string(first_errno) + "; needed " + std::to_string(need) + " 2 MiB pages, vm.nr_hugepages=" +
                   (have_pool ? std::to_string(pool) : std::string("?")) + "); using 4 KB pages";
        } else {
            note = "MAP_HUGETLB: cannot reserve " + std::to_string(huge_bytes >> 20) + " MiB (errno " +
                   std::to_string(errno) + "); using 4 KB pages";
        }
    }
    // No hugetlb mapping: a 2 MiB-aligned anonymous mapping with MADV_HUGEPAGE, so transparent huge pages back
    // it where THP is "madvise" (the common distro default). The CPU expert pool streams whole experts out of
    // this arena; with 4 KB pages every 3 MB expert costs ~750 TLB misses.
    constexpr uint64_t kAlign = 2ull << 20;
    const uint64_t padded = bytes + kAlign;
    void* raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = RegionBacking::NormalPages;
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + padded;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* p = (void*) aligned;
    if (madvise(p, bytes, MADV_HUGEPAGE) == 0) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages requested (MADV_HUGEPAGE)";
    }
    return p;
}

void free_region(void* p, uint64_t bytes) {
    if (!p) return;
    munmap(p, bytes);
}

bool pin_budget(uint64_t&, std::string&, const void*) { return false; }   // no OS-side budget below the driver's own limit
#endif

void region_page_backing(void* p, uint64_t bytes, uint64_t& sampled, uint64_t& large) {
    sampled = 0;
    large = 0;
#if defined(_WIN32)
    const PageBacking g = page_backing(p, bytes);
    sampled = g.sampled;
    large = g.large;
#else
    if (p == nullptr || bytes < 4096) return;
    static_cast<uint8_t*>(p)[0] = 0;   // fault the first page, so the kernel describes the mapping at all
    if (std::FILE* f = std::fopen("/proc/self/smaps", "r")) {
        char line[512];
        const unsigned long long addr = (unsigned long long) (uintptr_t) p;
        unsigned long long lo = 0, hi = 0, page_kb = 0;
        while (std::fgets(line, sizeof line, f)) {
            unsigned long long s = 0, e = 0;
            if (std::sscanf(line, "%llx-%llx", &s, &e) == 2) { lo = s; hi = e; page_kb = 0; continue; }
            if (addr >= lo && addr < hi && std::sscanf(line, "PageSize: %llu kB", &page_kb) == 1) break;
        }
        std::fclose(f);
        if (hi > lo) { sampled = 1; if (page_kb * 1024ull > 4096) large = 1; }
    }
#endif
}

void* alloc_region(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                   uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    void* p = alloc_region_raw(bytes, got, note, shared_file, shared_hash, mapping_base, mapping_bytes);
    if (p != nullptr && mapping_base == nullptr) {   // not a file mapping: the region is what gets released
        mapping_base = p;
        mapping_bytes = bytes;
    }
    return p;
}


DxgiMemory dxgi_memory(const void* luid) {
    DxgiMemory out;
#if defined(_WIN32)
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**) &factory)) || !factory) return out;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
        if (adapter == nullptr) break;
        bool match = luid == nullptr;
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && luid != nullptr) {
            const LUID* pluid = (const LUID*) luid;
            match = desc.AdapterLuid.LowPart == pluid->LowPart && desc.AdapterLuid.HighPart == pluid->HighPart;
        }
        if (!match) {
            adapter->Release();
            adapter = nullptr;
            continue;
        }
        IDXGIAdapter3* a3 = nullptr;
        if (SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr) {
            DXGI_QUERY_VIDEO_MEMORY_INFO local{}, shared{};
            if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local))) {
                out.local_budget = (uint64_t) local.Budget;
                out.local_usage = (uint64_t) local.CurrentUsage;
                out.ok = true;
            }
            if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &shared))) {
                out.non_local_budget = (uint64_t) shared.Budget;
                out.non_local_usage = (uint64_t) shared.CurrentUsage;
            }
            a3->Release();
        }
        // A failed enumeration is "unknown", not "none": a service or an OpenSSH session cannot enumerate outputs,
        // and a zero there would let the caller plan the whole budget on a machine that is driving a display.
        IDXGIOutput* output = nullptr;
        HRESULT hr = S_OK;
        int n = 0;
        while ((hr = adapter->EnumOutputs((UINT) n, &output)) == S_OK) {
            if (output != nullptr) output->Release();
            ++n;
        }
        out.outputs = hr == DXGI_ERROR_NOT_FOUND ? n : -1;
        adapter->Release();
        break;
    }
    factory->Release();
#else
    (void) luid;
#endif
    return out;
}

DxgiMemory print_dxgi_memory_report(const void* luid, bool luid_matched) {
    const DxgiMemory m = dxgi_memory(luid);
#if defined(_WIN32)
    if (!m.ok) {
        basalt::log::raw("  dxgi: no adapter reports a video-memory budget\n");
        return m;
    }
    // Say which adapter the figures came from: matching the device the caller named is the honest case, and taking
    // the first adapter is a guess the reader is entitled to know about.
    basalt::log::raw("  dxgi adapter: %s\n", luid_matched ? "matched to the device named" : "the first adapter: no device was named");
    basalt::log::raw("  dxgi local (device memory): budget %llu MiB, usage %llu MiB, headroom %llu MiB\n", (unsigned long long) (m.local_budget >> 20), (unsigned long long) (m.local_usage >> 20),
                 (unsigned long long) ((m.local_budget > m.local_usage ? m.local_budget - m.local_usage : 0) >> 20));
    basalt::log::raw("  dxgi non-local (shared memory): budget %llu MiB, usage %llu MiB, headroom %llu MiB\n", (unsigned long long) (m.non_local_budget >> 20), (unsigned long long) (m.non_local_usage >> 20),
                 (unsigned long long) ((m.non_local_budget > m.non_local_usage ? m.non_local_budget - m.non_local_usage : 0) >> 20));
    if (m.outputs < 0) basalt::log::raw("  dxgi outputs: unknown (the enumeration failed)\n");
    else basalt::log::raw("  dxgi outputs: %d\n", m.outputs);
#else
    (void) luid;
    (void) luid_matched;   // nothing budgets video memory here, and the figures say so
#endif
    return m;
}

bool budgets_video_memory() {
#if defined(_WIN32)
    return true;
#else
    return false;
#endif
}

uint64_t dxgi_non_local_budget(const void* luid) { return dxgi_memory(luid).non_local_budget; }

bool enable_lock_memory_privilege() {
#if defined(_WIN32)
    return enable_privilege(L"SeLockMemoryPrivilege");
#else
    return false;
#endif
}

LargePagePrivilege probe_large_page_privilege() {
    LargePagePrivilege p;
#if defined(_WIN32)
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES, &tok)) {
        p.note = "OpenProcessToken failed (error " + std::to_string(GetLastError()) + ")";
        return p;
    }
    // 1. Is SeLockMemoryPrivilege in the token at all, and is it enabled? This is the pair a 1314 has to be read
    //    against: a privilege that is absent cannot be enabled by AdjustTokenPrivileges, and one that is present but
    //    refused by the allocation is a different problem from one the account never had.
    LUID luid{};
    if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &luid)) {
        p.looked_up = true;
        DWORD need = 0;
        (void) GetTokenInformation(tok, TokenPrivileges, nullptr, 0, &need);
        if (need > 0) {
            std::vector<uint8_t> buf(need);
            if (GetTokenInformation(tok, TokenPrivileges, buf.data(), need, &need)) {
                const TOKEN_PRIVILEGES* tp = (const TOKEN_PRIVILEGES*) buf.data();
                for (DWORD i = 0; i < tp->PrivilegeCount; ++i) {
                    if (tp->Privileges[i].Luid.LowPart == luid.LowPart &&
                        tp->Privileges[i].Luid.HighPart == luid.HighPart) {
                        p.present = true;
                        p.enabled = (tp->Privileges[i].Attributes & SE_PRIVILEGE_ENABLED) != 0;
                        break;
                    }
                }
            }
        }
    }
    // 2. Elevation and integrity: a UAC-filtered token at Medium integrity is the ordinary interactive case, and a
    //    privilege that is present there can still be refused by the allocation.
    {
        TOKEN_ELEVATION el{};
        DWORD n = 0;
        if (GetTokenInformation(tok, TokenElevation, &el, sizeof el, &n)) p.elevated = el.TokenIsElevated != 0;
    }
    {
        DWORD need = 0;
        (void) GetTokenInformation(tok, TokenIntegrityLevel, nullptr, 0, &need);
        if (need > 0) {
            std::vector<uint8_t> buf(need);
            if (GetTokenInformation(tok, TokenIntegrityLevel, buf.data(), need, &need)) {
                const TOKEN_MANDATORY_LABEL* ml = (const TOKEN_MANDATORY_LABEL*) buf.data();
                const DWORD* rid = (const DWORD*) GetSidSubAuthority(
                    ml->Label.Sid, (DWORD) (GetSidSubAuthorityCount(ml->Label.Sid)[0] - 1));
                if (rid != nullptr) p.integrity_rid = *rid;   // 0x1000 medium, 0x2000 high, 0x3000 system
            }
        }
    }
    // 3. Is Administrators present but deny-only? That is what a filtered token looks like, and it is the reason a
    //    privilege can be listed and enabled while the allocation still refuses it.
    {
        DWORD need = 0;
        (void) GetTokenInformation(tok, TokenGroups, nullptr, 0, &need);
        if (need > 0) {
            std::vector<uint8_t> buf(need);
            if (GetTokenInformation(tok, TokenGroups, buf.data(), need, &need)) {
                const TOKEN_GROUPS* tg = (const TOKEN_GROUPS*) buf.data();
                PSID admins_sid = nullptr;
                SID_IDENTIFIER_AUTHORITY nt = SECURITY_NT_AUTHORITY;
                if (AllocateAndInitializeSid(&nt, 2, SECURITY_BUILTIN_DOMAIN_RID, DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0,
                                             0, 0, &admins_sid)) {
                    for (DWORD i = 0; i < tg->GroupCount; ++i) {
                        if (!EqualSid(tg->Groups[i].Sid, admins_sid)) continue;
                        p.admin_group = true;
                        p.admin_deny_only = (tg->Groups[i].Attributes & SE_GROUP_USE_FOR_DENY_ONLY) != 0;
                        break;
                    }
                    FreeSid(admins_sid);
                }
            }
        }
    }
    CloseHandle(tok);
    // 4. The two allocation forms, each with its own error: the documented single call, and reserve-then-commit.
    {
        const SIZE_T two_mib = (SIZE_T) GetLargePageMinimum();
        void* a = VirtualAlloc(nullptr, two_mib, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        p.single_ok = a != nullptr;
        p.single_error = a != nullptr ? 0 : (uint64_t) GetLastError();
        if (a != nullptr) VirtualFree(a, 0, MEM_RELEASE);
    }
    {
        const SIZE_T two_mib = (SIZE_T) GetLargePageMinimum();
        void* b = VirtualAlloc(nullptr, two_mib, MEM_RESERVE, PAGE_READWRITE);
        p.reserve_ok = b != nullptr;
        p.reserve_error = b != nullptr ? 0 : (uint64_t) GetLastError();
        if (b != nullptr) {
            void* c = VirtualAlloc(b, two_mib, MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
            p.commit_ok = c != nullptr;
            p.commit_error = c != nullptr ? 0 : (uint64_t) GetLastError();
            if (c != nullptr) {   // the form the arena commits with: read its pages back before releasing them
                const PageBacking g = page_backing(c, (uint64_t) two_mib);
                p.commit_sampled = g.sampled;
                p.commit_large = g.large;
            }
            VirtualFree(b, 0, MEM_RELEASE);
        }
    }
    p.ok = true;
#else
    p.note = "not applicable on this platform";
#endif
    return p;
}

LargePageProbe probe_large_pages(uint64_t want) {
    LargePageProbe p;
#if defined(_WIN32)
    p.minimum = (uint64_t) GetLargePageMinimum();
    p.requested = want;
    if (p.minimum == 0) return p;
    enable_lock_memory_privilege();   // the pool needs the privilege in this token, whatever the arena did
    const uint64_t floor = p.minimum * 32;   // the smallest block worth reporting
    for (uint64_t ask = want / p.minimum * p.minimum; ask >= floor; ask = ask / 2 / p.minimum * p.minimum) {
        void* got = VirtualAlloc(nullptr, (SIZE_T) ask, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (got != nullptr) {
            const PageBacking b = page_backing(got, ask);   // while it is still mapped
            p.sampled = b.sampled;
            p.large_pages = b.large;
            VirtualFree(got, 0, MEM_RELEASE);
            p.granted = ask;
            return p;
        }
        if (p.error == 0) p.error = (uint64_t) GetLastError();
    }
#else
    (void) want;
#endif
    return p;
}

double ViewFaultProbe::off_ms_per_mib() const {
    return window_bytes > 0 && off_seconds > 0 ? off_seconds * 1000.0 / ((double) window_bytes / 1048576.0) : 0.0;
}
double ViewFaultProbe::on_ms_per_mib() const {
    return window_bytes > 0 && on_seconds > 0 ? on_seconds * 1000.0 / ((double) window_bytes / 1048576.0) : 0.0;
}
double ViewFaultProbe::early_ms_per_mib() const {
    return window_bytes > 0 && early_seconds > 0 ? early_seconds * 1000.0 / ((double) window_bytes / 1048576.0) : 0.0;
}

namespace {
#if defined(_WIN32)
bool map_readonly(const std::string& path, void*& view, uint64_t& size) {
    HANDLE fh = CreateFileW(to_wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz{};
    const bool sized = GetFileSizeEx(fh, &sz) != 0;
    if (sized) size = (uint64_t) sz.QuadPart;
    HANDLE map = sized ? CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
    CloseHandle(fh);
    if (map == nullptr) return false;
    view = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(map);
    return view != nullptr;
}
void unmap_readonly(void* view, uint64_t) {
    if (view != nullptr) UnmapViewOfFile(view);
}
#else
bool map_readonly(const std::string& path, void*& view, uint64_t& size) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    const bool sized = fstat(fd, &st) == 0 && st.st_size > 0;
    if (sized) size = (uint64_t) st.st_size;
    void* v = sized ? mmap(nullptr, (size_t) size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    ::close(fd);
    if (v == MAP_FAILED) return false;
    view = v;
    return true;
}
void unmap_readonly(void* view, uint64_t size) {
    if (view != nullptr) munmap(view, (size_t) size);
}
#endif

/// Touch one byte of every page of [view, view + bytes): the cost measured is the fault, not the copy. `hint` issues
/// the prefetch inside the timed region, because its cost is part of the answer; `wait_first` instead hints, waits
/// `wait_ms` outside the clock, and then times the touch alone.
double fault_window(const uint8_t* view, uint64_t bytes, int mode, uint32_t wait_ms) {
    using clock = std::chrono::steady_clock;
    if (mode == 2) {
        advise_willneed(view, bytes);
        if (wait_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
    const auto t0 = clock::now();
    if (mode == 1) advise_willneed(view, bytes);
    volatile uint8_t sink = 0;
    for (uint64_t off = 0; off < bytes; off += 4096) sink = (uint8_t) (sink + view[off]);
    const auto t1 = clock::now();
    (void) sink;
    return std::chrono::duration<double>(t1 - t0).count();
}
}  // namespace

ViewFaultProbe probe_view_faults(const std::string& path, uint64_t window_bytes, uint32_t rounds, uint32_t wait_ms) {
    ViewFaultProbe p;
    p.window_bytes = window_bytes;
    p.wait_ms = (double) wait_ms;
    if (window_bytes < 4096 || rounds == 0) {
        p.error = "probe: a window of at least one page and one round are needed";
        return p;
    }
    constexpr uint32_t kArms = 3;
    const uint32_t windows = rounds * kArms;
    void* v = nullptr;
    uint64_t size = 0;
    if (!map_readonly(path, v, size)) { p.error = "probe: cannot map " + path; return p; }
    unmap_readonly(v, size);
    // one gap before the first window and one after the last, so no window is clipped
    const uint64_t stride = size / (uint64_t) (windows + 1);
    if (stride < window_bytes) { p.error = "probe: the file is too small for that many windows"; return p; }
    double sum[kArms] = {0, 0, 0}, lo[kArms] = {0, 0, 0}, hi[kArms] = {0, 0, 0};
    uint32_t n[kArms] = {0, 0, 0};
    for (uint32_t r = 0; r < rounds; ++r) {
        for (uint32_t k = 0; k < kArms; ++k) {
            const uint32_t idx = r * kArms + k;
            const uint64_t off = (uint64_t) (idx + 1) * stride;
            // rotate which arm gets which of the round's windows, so no arm is always the colder one
            const uint32_t mode = (k + r) % kArms;
            void* view = nullptr;
            uint64_t msize = 0;
            if (!map_readonly(path, view, msize)) { p.error = "probe: cannot map " + path; return p; }
            const double s = fault_window((const uint8_t*) view + off, window_bytes, (int) mode, wait_ms);
            unmap_readonly(view, msize);
            if (n[mode] == 0 || s < lo[mode]) lo[mode] = s;
            if (n[mode] == 0 || s > hi[mode]) hi[mode] = s;
            sum[mode] += s;
            ++n[mode];
        }
    }
    for (uint32_t a = 0; a < kArms; ++a)
        if (n[a] == 0) { p.error = "probe: an arm did not run"; return p; }
    p.off_seconds = sum[0] / (double) n[0];
    p.off_min = lo[0];
    p.off_max = hi[0];
    p.on_seconds = sum[1] / (double) n[1];
    p.on_min = lo[1];
    p.on_max = hi[1];
    p.early_seconds = sum[2] / (double) n[2];
    p.early_min = lo[2];
    p.early_max = hi[2];
    p.windows_per_arm = n[0];
    p.ok = true;
    return p;
}

}  // namespace basalt::platform
