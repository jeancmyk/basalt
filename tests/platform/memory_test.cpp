// tests/platform/memory_test.cpp - lock_resident on 256 MiB (or what the locked-memory limit allows), the RAM floor, the
// file cache decisions and the read-ahead hint on a mapped file view (CPU only, no GPU, no model; one temp file).
#include "basalt/platform/memory.hpp"

#include "basalt/platform/file.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#include <psapi.h>   // QueryWorkingSetEx
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif


namespace {
uint64_t gib(double g) { return (uint64_t) (g * 1073741824.0); }
uint64_t mib(double m) { return (uint64_t) (m * 1048576.0); }

// The file tier's unbuffered choice (true = the file cache keeps its reads, so they stay buffered)
int file_cache_keeps_cases() {
    using basalt::platform::file_cache_keeps;
    struct Case { const char* what; double avail, arena, read; bool keeps; };
    const double experts = 71.7;   // routed experts of a large model
    const Case cases[] = {
        // passing the budget asked for and every shard's bytes (103.7 GiB) goes unbuffered although the experts
        // outside the copy would fit the cache
        {"96 GB, budget 72, all shards' bytes", 83.0, 72.0, 103.7, false},
        {"96 GB, budget 72, before the copy (arena = all experts)", 83.0, experts, 0.0, true},
        {"96 GB, budget 72, copy built (48.94 resident)", 83.0 - 48.94, 0.0, experts - 48.94, true},
        {"96 GB, budget 40, before the copy", 83.0, 40.0, experts - 40.0, true},
        {"96 GB, budget 40, copy built", 83.0 - 40.0, 0.0, experts - 40.0, true},
        // low-RAM machines stay unbuffered
        {"64 GB, budget 40, before the copy", 55.0, 40.0, experts - 40.0, false},
        {"64 GB, budget 40, copy built", 15.0, 0.0, experts - 40.0, false},
        {"32 GB, budget 16, before the copy", 26.0, 16.0, experts - 16.0, false},
        {"32 GB, budget 16, copy built", 10.0, 0.0, experts - 16.0, false},
        {"32 GB, Q2_0 experts (34 GiB), budget 8", 26.0, 8.0, 34.0 - 8.0, false},
        {"nothing available", 0.0, 0.0, 1.0, false},
        {"arena larger than the RAM", 10.0, 20.0, 0.0, true},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const bool got = file_cache_keeps(gib(c.avail), gib(c.arena), gib(c.read));
        std::printf("file_cache_keeps %-58s -> %s%s\n", c.what, got ? "buffered" : "unbuffered",
                    got == c.keeps ? "" : "   <-- WRONG");
        fail |= got != c.keeps;
    }
    return fail;
}
// The RAM floor, the largest safe copy and the serve guard's decision, on injected MemAvailable readings
int ram_floor_cases() {
    using namespace basalt::platform;
    int fail = 0;
    auto check = [&](const char* what, double got, double want) {
        const bool ok = got > want - 0.01 && got < want + 0.01;
        std::printf("ram floor %-62s %8.2f GiB%s\n", what, got, ok ? "" : "   <-- WRONG");
        fail |= !ok;
    };
    const double G = 1073741824.0;
    const uint64_t margin = gib(2.0);
    const uint64_t layer_q4 = 512ull * 3170304;   // every expert of a layer of a 4-bit pack (1.51 GiB)
    const uint64_t layer_q2 = 512ull * 1382400;   // the same of the 2-bit pack (0.66 GiB)
    check("nothing read from the file: the margin alone", ram_floor(margin, 0, layer_q4) / G, 2.0);
    check("4-bit pack, experts read from the file: margin + 3.5 + 2 layers", ram_floor(margin, gib(15), layer_q4) / G,
          2.0 + 3.5 + 2.0 * (double) layer_q4 / G);
    check("2-bit pack: margin + 3.5 + 2 layers", ram_floor(margin, gib(1), layer_q2) / G,
          2.0 + 3.5 + 2.0 * (double) layer_q2 / G);
    check("small layers: the 4 GiB file cache minimum applies", file_cache_floor(1, gib(0.1)) / G, 4.0);
    check("one byte from the file is enough to need the cache", file_cache_floor(1, layer_q4) / G,
          3.5 + 2.0 * (double) layer_q4 / G);
    // a 30 GiB machine with 27 GiB available and 2 GiB taken by the server later
    const uint64_t fl = ram_floor(margin, gib(15), layer_q4);
    check("largest safe copy, 27 available, 2 late", max_safe_copy(gib(27), gib(2), fl) / G, 27.0 - 2.0 - fl / G);
    check("largest safe copy when the floor exceeds the RAM", max_safe_copy(gib(8), gib(2), fl) / G, 0.0);
    const uint64_t guard = ram_guard_floor(margin, gib(15));
    check("serve guard, experts read from the file: margin + 3.5", guard / G, 5.5);
    check("serve guard, nothing read from the file: the margin", ram_guard_floor(margin, 0) / G, 2.0);
    struct Case { const char* what; double avail; bool low; };
    const Case cases[] = {
        {"well above the floor", 20.0, false},
        {"at the load floor (sized as asked)", fl / G, false},
        {"exactly at the guard line", guard / G, false},
        {"just below the guard line", guard / G - 0.01, true},
        {"the freeze's reading (2.5 GB)", 2.33, true},
        {"unknown (0)", 0.0, false},
    };
    for (const Case& c : cases) {
        const bool got = ram_below_floor(gib(c.avail), guard);
        std::printf("ram guard %-62s -> %s%s\n", c.what, got ? "stop" : "run", got == c.low ? "" : "   <-- WRONG");
        fail |= got != c.low;
    }
    return fail;
}
// The read-ahead seam on a real mapped file view. Whether the OS actually prefetched is not observable portably and
// is not what the seam promises, so what is checked is that every range shape returns without faulting and that the
// mapping's bytes stay exactly as written.
int read_ahead_cases() {
    using namespace basalt::platform;
    int fail = 0;
    const bool on = read_ahead_enabled();
    std::printf("read-ahead %-58s -> %s%s\n", "read_ahead_enabled (both platforms)", on ? "on" : "off",
                on ? "" : "   <-- WRONG");
    fail |= !on;

    const uint64_t bytes = 256ull << 10;   // spans more than one 128 KiB hint window
    std::vector<uint8_t> pattern(bytes);
    for (uint64_t i = 0; i < bytes; ++i) pattern[i] = (uint8_t) (i * 13u + (i >> 8) * 7u);
    const std::string path = (std::filesystem::temp_directory_path() / "basalt_memory_test_view.bin").string();
    {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*) pattern.data(), (std::streamsize) bytes);
        if (!f) {
            std::printf("read-ahead %-58s   <-- cannot write %s\n", "temp file", path.c_str());
            return fail | 1;
        }
    }
    // mapped the way the engine maps its file views: whole file, read-only
    const uint8_t* base = nullptr;
#if defined(_WIN32)
    const std::wstring wpath = basalt::platform::to_wide(path);
    if (!wpath.empty()) {
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
            base = (const uint8_t*) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
            // the view holds the section and the file open, so both handles go now
            CloseHandle(m);
            CloseHandle(h);
        }
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        void* view = mmap(nullptr, (size_t) bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (view != MAP_FAILED) base = (const uint8_t*) view;
    }
#endif
    if (base == nullptr) {
        std::printf("read-ahead %-58s   <-- cannot map %s\n", "read-only view", path.c_str());
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return fail | 1;
    }
    auto spots_intact = [&](const char* when) {
        bool ok = true;
        for (const uint64_t off : {(uint64_t) 0, (uint64_t) 1234, bytes / 2, bytes - 1}) ok &= base[off] == pattern[off];
        std::printf("read-ahead %-58s -> %s%s\n", when, ok ? "bytes intact" : "bytes CHANGED", ok ? "" : "   <-- WRONG");
        fail |= !ok;
    };
    spots_intact("view reads back as written, before the hints");
    struct HintCase { const char* what; const void* p; uint64_t n; };
    const HintCase cases[] = {
        {"whole view", base, bytes},
        {"page-unaligned interior range", base + 1234, 8ull << 10},
        {"one-byte range", base + bytes / 2, 1},
        {"range running past the end of the view", base + bytes - 100, 1ull << 20},
        {"null pointer", nullptr, 4096},
        {"zero length", base, 0},
    };
    for (const HintCase& c : cases) {
        // the hint is void and silent: reaching the print at all (no fault) and the identical read-back after the
        // whole sequence are what these cases assert
        advise_willneed(c.p, c.n);
        std::printf("read-ahead %-58s -> returned\n", c.what);
    }
    spots_intact("view reads back as written, after every hint");
#if defined(_WIN32)
    UnmapViewOfFile((void*) base);
#else
    munmap((void*) base, (size_t) bytes);
#endif
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return fail;
}
// drop_mapped_view on a real mapped view: ranges of every shape return, bad ranges are refused, and the bytes read
// back as written after the pages were given up (they are re-read from the file when touched again).
int drop_view_cases() {
    using namespace basalt::platform;
    int fail = 0;
    const uint64_t bytes = 1ull << 20;
    std::vector<uint8_t> pattern(bytes);
    for (uint64_t i = 0; i < bytes; ++i) pattern[i] = (uint8_t) (i * 31u + (i >> 9) * 5u);
    const std::string path = (std::filesystem::temp_directory_path() / "basalt_memory_test_drop.bin").string();
    {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*) pattern.data(), (std::streamsize) bytes);
        if (!f) return 1;
    }
    const uint8_t* base = nullptr;
#if defined(_WIN32)
    const std::wstring wpath = basalt::platform::to_wide(path);
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        base = (const uint8_t*) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(m);
        CloseHandle(h);
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        void* view = mmap(nullptr, (size_t) bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (view != MAP_FAILED) base = (const uint8_t*) view;
    }
#endif
    if (base == nullptr) {
        std::printf("drop view %-58s   <-- cannot map\n", "read-only view");
        std::filesystem::remove(path);
        return 1;
    }
    auto intact = [&]() {
        for (uint64_t i = 0; i < bytes; i += 997) if (base[i] != pattern[i]) return false;
        return base[bytes - 1] == pattern[bytes - 1];
    };
    struct Case { const char* what; uint64_t off, n; bool want; };
    const Case cases[] = {
        {"whole view", 0, bytes, true},
        {"page-unaligned interior range", 1234, 300ull << 10, true},
        {"one byte", bytes / 2, 1, true},
        {"the last byte", bytes - 1, 1, true},
        {"zero length", 5, 0, true},
        {"range past the end of the view", bytes - 100, 200, false},
        {"offset past the end of the view", bytes + 1, 1, false},
    };
    for (const Case& c : cases) {
        volatile uint8_t sink = 0;
        for (uint64_t i = 0; i < bytes; i += 4096) sink = (uint8_t) (sink + base[i]);   // pages in, so there is something to give up
        const bool got = drop_mapped_view(base, bytes, c.off, c.n);
        const bool ok = got == c.want && intact();
        std::printf("drop view %-58s -> %s%s\n", c.what, got ? "ok" : "refused", ok ? "" : "   <-- WRONG");
        fail |= !ok;
    }
    const bool null_ok = drop_mapped_view(nullptr, 0, 0, 4096);
    std::printf("drop view %-58s -> %s%s\n", "null base", null_ok ? "ok" : "refused", null_ok ? "" : "   <-- WRONG");
    fail |= !null_ok;
#if defined(_WIN32)
    UnmapViewOfFile((void*) base);
#else
    munmap((void*) base, (size_t) bytes);
#endif
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return fail;
}
// The serve guard with a view handed to it: with a reading within its lead of the floor the guard gives the view's
// pages back (Windows; Linux leaves them to MemAvailable) and they read back as written; once the view is withdrawn
// the guard leaves it alone. The floor is left at 0 afterwards, so the guard thread never reads low again.
int guard_reclaim_cases() {
    using namespace basalt::platform;
    int fail = 0;
    const uint64_t bytes = 64ull << 20;
    std::vector<uint8_t> pattern(bytes);
    for (uint64_t i = 0; i < bytes; ++i) pattern[i] = (uint8_t) (i * 29u + (i >> 10) * 3u);
    const std::string path = (std::filesystem::temp_directory_path() / "basalt_memory_test_guard.bin").string();
    {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*) pattern.data(), (std::streamsize) bytes);
        if (!f) return 1;
    }
    const uint8_t* base = nullptr;
#if defined(_WIN32)
    const std::wstring wpath = basalt::platform::to_wide(path);
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        base = (const uint8_t*) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(m);
        CloseHandle(h);
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        void* view = mmap(nullptr, (size_t) bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (view != MAP_FAILED) base = (const uint8_t*) view;
    }
#endif
    if (base == nullptr) {
        std::printf("guard view %-57s   <-- cannot map\n", "read-only view");
        std::filesystem::remove(path);
        return 1;
    }
    auto touch = [&]() {
        volatile uint8_t sink = 0;
        for (uint64_t i = 0; i < bytes; i += 4096) sink = (uint8_t) (sink + base[i]);
    };
    auto intact = [&]() {
        for (uint64_t i = 0; i < bytes; i += 997) if (base[i] != pattern[i]) return false;
        return base[bytes - 1] == pattern[bytes - 1];
    };
    // the share of the view's pages in the process's working set; -1 where it is not measured
    auto resident = [&]() -> double {
#if defined(_WIN32)
        const uint64_t pages = bytes / 4096;
        std::vector<PSAPI_WORKING_SET_EX_INFORMATION> ws(pages);
        for (uint64_t i = 0; i < pages; ++i) ws[i].VirtualAddress = (PVOID) (base + i * 4096);
        if (!QueryWorkingSetEx(GetCurrentProcess(), ws.data(), (DWORD) (pages * sizeof ws[0]))) return 0.0;
        uint64_t in = 0;
        for (const PSAPI_WORKING_SET_EX_INFORMATION& w : ws) in += w.VirtualAttributes.Valid;
        return (double) in / (double) pages;
#else
        return -1.0;
#endif
    };
    // half a GiB under the reading: inside the guard's lead without reading as low; set again on every poll, so the
    // machine's own drift does not carry the reading out of the lead
    auto near_floor = [] {
        const uint64_t a = available_memory();
        return a > (512ull << 20) ? a - (512ull << 20) : 0;
    };
    auto report = [&](const char* what, double share, bool ok) {
        if (share < 0) std::printf("guard view %-57s -> %s%s\n", what, ok ? "ok" : "failed", ok ? "" : "   <-- WRONG");
        else std::printf("guard view %-57s -> %.0f%% resident%s\n", what, share * 100.0, ok ? "" : "   <-- WRONG");
        fail |= !ok;
    };
    const bool measured = resident() >= 0;

    touch();
    const double before = resident();
    report("read, before the guard has it", before, !measured || before > 0.9);

    ram_guard_reclaim_view(base, bytes);
    double given = before;
    for (int i = 0; i < 60; ++i) {
        start_ram_guard(near_floor());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        given = resident();
        if (measured && given < 0.1) break;
    }
    report("handed to the guard, reading within the lead", given, !measured || given < 0.1);
    report("given back, the bytes read as written", -1.0, intact());

    ram_guard_reclaim_view(nullptr, 0);
    touch();
    for (int i = 0; i < 10; ++i) {
        start_ram_guard(near_floor());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    const double kept = resident();
    report("withdrawn, read again, reading within the lead", kept, !measured || kept > 0.9);
    start_ram_guard(0);

#if defined(_WIN32)
    UnmapViewOfFile((void*) base);
#else
    munmap((void*) base, (size_t) bytes);
#endif
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return fail;
}
// When the guard gives the mapped view back, on injected readings
int reclaim_due_cases() {
    using namespace basalt::platform;
    int fail = 0;
    const uint64_t floor = gib(6.0), lead = gib(1.0), after = gib(7.0), step = 256ull << 20;
    auto check = [&](const char* what, bool got, bool want) {
        std::printf("reclaim due %-62s %s%s\n", what, got ? "yes" : "no", got == want ? "" : "   <-- WRONG");
        fail |= got != want;
    };
    check("above the lead: no", reclaim_due(floor + lead, floor, lead, 0, 0), false);
    check("within the lead, none given yet: yes", reclaim_due(floor + lead - 1, floor, lead, 0, 0), true);
    check("reading unknown: no", reclaim_due(0, floor, lead, 0, 0), false);
    check("view refilled a chunk, cooldown over: yes", reclaim_due(after - step, floor, lead, after, 250), true);
    check("view refilled a chunk, cooldown running: no", reclaim_due(after - step, floor, lead, after, 249), false);
    check("less than a chunk refilled: no", reclaim_due(after - step + 1, floor, lead, after, 10000), false);
    check("low and nothing refilling: no", reclaim_due(floor + lead - 1, floor, lead, floor + lead - 1, 10000), false);
    check("rose above the lead, fell back a chunk: yes", reclaim_due(floor + lead - 1, floor, lead, gib(9.0), 250), true);
    return fail;
}
// The watchdog's line: the budget flag is named only when the resident copy is big enough for it to shrink.
int exhausted_message_cases() {
    using basalt::platform::ram_exhausted_message;
    int fail = 0;
    const uint64_t margin = gib(4.0);
    struct Case { const char* what; bool resident; double copy_gib; bool names_budget; bool names_mmap; };
    const Case cases[] = {
        {"mapped experts, nothing resident", false, 0.0, false, true},
        {"tiny resident copy", true, 0.008, false, false},
        {"empty resident copy", true, 0.0, false, false},
        {"large resident copy", true, 60.0, true, false},
    };
    for (const Case& c : cases) {
        const std::string m = ram_exhausted_message(gib(1.0), c.resident, gib(c.copy_gib), margin);
        const bool budget = m.find("--resident-budget-gib") != std::string::npos;
        const bool mmap = m.find("--mmap-experts") != std::string::npos;
        const bool ok = budget == c.names_budget && mmap == c.names_mmap && m.find("1024 MiB free") != std::string::npos;
        std::printf("ram message %-56s -> %s%s\n", c.what, m.c_str(), ok ? "" : "   <-- WRONG");
        fail |= !ok;
    }
    return fail;
}
// The size lock_resident is exercised with: the full 256 MiB where the locked-memory limit allows it, otherwise half
// of what the limit leaves. Zero means the limit is too low to lock anything meaningful.
uint64_t lock_test_bytes(bool& reduced) {
    const uint64_t want = 256ull << 20;
    reduced = false;
#if defined(_WIN32)
    return want;
#else
    struct rlimit rl {};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0 || rl.rlim_cur == RLIM_INFINITY) return want;
    // headroom for locks the process already holds (the engine's own allocations, thread stacks)
    const uint64_t limit = (uint64_t) rl.rlim_cur;
    if (limit >= want + (want >> 2)) return want;
    reduced = true;
    const uint64_t page = (uint64_t) sysconf(_SC_PAGE_SIZE);
    const uint64_t half = limit / 2 / page * page;
    return half >= 16 * page ? half : 0;
#endif
}

// The VRAM a plan may take: the smaller of the device's free figure and what the process's budget leaves, and the
// two exhausted corners must be 0 rather than an underflow. An unanswered query leaves the free figure alone.
int vram_plan_cases() {
    using basalt::platform::vram_free_for_plan;
    struct Case { const char* what; double device_free, budget, usage; bool known; double want; };
    const Case cases[] = {
        {"budget roomy: the device's free figure is smaller", 20.0, 30.0, 10.0, true, 20.0},
        {"budget tight: what the budget leaves is smaller", 20.0, 30.0, 25.0, true, 5.0},
        {"usage exactly at the budget", 20.0, 30.0, 30.0, true, 0.0},
        {"usage past the budget (no underflow)", 20.0, 30.0, 40.0, true, 0.0},
        {"budget unknown: the free figure stands", 20.0, 0.0, 0.0, false, 20.0},
        {"nothing free either way", 0.0, 30.0, 30.0, true, 0.0},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const uint64_t got = vram_free_for_plan(gib(c.device_free), c.known, gib(c.budget), gib(c.usage));
        const bool ok = got == gib(c.want);
        std::printf("vram plan %-58s -> %llu MiB%s\n", c.what, (unsigned long long) (got >> 20),
                    ok ? "" : "   <-- WRONG");
        fail |= !ok;
    }
    return fail;
}
// The reserve cap: a want the budget cannot fund is reduced to what the headroom leaves, never below the stall floor,
// and a want that fits is untouched. Each case says whether the cap fired, because that flag decides whether the
// startup line is printed.
int vram_reserve_cases() {
    using basalt::platform::vram_reserve_for_plan;
    const uint64_t floor = 256ull << 20;
    struct Case { const char* what; double want, plan_free, got; bool capped; };
    const Case cases[] = {
        {"want fits the headroom: untouched", 700.0, 2048.0, 700.0, false},
        {"want equals the headroom", 700.0, 700.0, 700.0, false},
        {"budget headroom is smaller: capped to it", 20000.0, 900.0, 900.0, true},
        {"headroom under the floor: the floor stands", 20000.0, 64.0, 256.0, true},
        {"headroom exactly the floor", 20000.0, 256.0, 256.0, true},
        {"no headroom at all: the floor stands", 20000.0, 0.0, 256.0, true},
    };
    int fail = 0;
    for (const Case& c : cases) {
        bool capped = false;
        const uint64_t got = vram_reserve_for_plan(mib(c.want), mib(c.plan_free), floor, &capped);
        const bool ok = got == mib(c.got) && capped == c.capped;
        std::printf("vram reserve %-54s -> %llu MiB, capped %s%s\n", c.what, (unsigned long long) (got >> 20),
                    capped ? "yes" : "no", ok ? "" : "   <-- WRONG");
        fail |= !ok;
    }
    return fail;
}
// The same cap when the plan also needs room of its own: the reserve stops `keep` short of the headroom, so a large
// reserve leaves a small cache rather than none. The floor still wins when there is not enough room for both.
int vram_reserve_keep_cases() {
    using basalt::platform::vram_reserve_for_plan_keeping;
    const uint64_t floor = 256ull << 20;
    struct Case { const char* what; double want, plan_free, keep, got; bool capped; };
    const Case cases[] = {
        {"want fits even after the plan's room", 700.0, 2048.0, 512.0, 700.0, false},
        {"capped to the headroom less the plan's room", 20000.0, 900.0, 400.0, 500.0, true},
        {"the plan's room is most of the headroom", 20000.0, 400.0, 300.0, 256.0, true},
        {"headroom below the plan's room and the floor", 20000.0, 200.0, 300.0, 256.0, true},
        {"no plan room: the whole headroom is available", 20000.0, 900.0, 0.0, 900.0, true},
    };
    int fail = 0;
    for (const Case& c : cases) {
        bool capped = false;
        const uint64_t got = vram_reserve_for_plan_keeping(mib(c.want), mib(c.plan_free), mib(c.keep), floor, &capped);
        const bool ok = got == mib(c.got) && capped == c.capped;
        std::printf("vram reserve keep %-49s -> %llu MiB, capped %s%s\n", c.what, (unsigned long long) (got >> 20),
                    capped ? "yes" : "no", ok ? "" : "   <-- WRONG");
        fail |= !ok;
    }
    return fail;
}
// The spill check: an allocation the driver placed partly in system memory lowers the free figure by less than it
// asked for. Only a shortfall past the tolerance is a spill; a free figure that did not fall says nothing at all.
int spill_cases() {
    using basalt::platform::spill_shortfall;
    const uint64_t tol = 32ull << 20;
    struct Case { const char* what; double before, after, allocated, want; };
    const Case cases[] = {
        {"free fell by exactly the allocation", 100.0, 70.0, 30.0, 0.0},
        {"free fell by more than the allocation", 100.0, 60.0, 30.0, 0.0},
        {"free unchanged: nothing to judge", 100.0, 100.0, 30.0, 0.0},
        {"free rose: something else was freed", 100.0, 110.0, 30.0, 0.0},
        // a fall just short of the allocation: the shortfall is inside the tolerance the check allows
        {"shortfall inside the tolerance", 100.0, 70.001, 30.0, 0.0},
        {"half of it placed in system memory", 200.0, 150.0, 100.0, 50.0},
        {"a zero-byte allocation", 100.0, 90.0, 0.0, 0.0},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const uint64_t got = spill_shortfall(gib(c.before), gib(c.after), gib(c.allocated), tol);
        const bool ok = got == gib(c.want);
        std::printf("spill %-58s -> %llu MiB%s\n", c.what, (unsigned long long) (got >> 20),
                    ok ? "" : "   <-- WRONG");
        fail |= !ok;
    }
    return fail;
}
// The DXGI figures the memory report and the tier check read: a query that ran must carry a budget, and an output
// count is never a fake zero. A platform without DXGI (or a session that cannot see an adapter) reports not-ok,
// which the callers read as unknown.
int dxgi_cases() {
    using namespace basalt::platform;
    const DxgiMemory m = dxgi_memory(nullptr);
    if (!m.ok) {
        std::printf("dxgi %-58s -> no budget reported (unknown)\n", "first adapter");
        return 0;
    }
    const bool budget = m.local_budget > 0;
    const bool outputs = m.outputs == -1 || m.outputs >= 0;
    std::printf("dxgi %-58s -> local budget %llu MiB, usage %llu MiB, non-local budget %llu MiB, outputs %s%s\n",
                "first adapter", (unsigned long long) (m.local_budget >> 20),
                (unsigned long long) (m.local_usage >> 20), (unsigned long long) (m.non_local_budget >> 20),
                m.outputs < 0 ? "unknown" : std::to_string(m.outputs).c_str(),
                budget && outputs ? "" : "   <-- WRONG");
    return !(budget && outputs);
}

// The large-page probe must describe the pool without keeping anything: a grant is a multiple of the minimum and no
// larger than the request, and a refusal names an error. Nothing here fails when the pool declines - that is an
// answer, not a bug - so only the shape of the report is asserted.
int large_page_cases() {
    using namespace basalt::platform;
    const LargePageProbe p = probe_large_pages(30ull << 30);
    if (p.minimum == 0) {
        std::printf("large pages %-50s -> none on this system\n", "30 GiB requested");
        return 0;
    }
    const bool shape = p.granted <= p.requested && p.granted % p.minimum == 0;
    const bool explained = p.granted > 0 || p.error != 0;
    std::printf("large pages %-50s -> minimum %llu MiB, granted %llu MiB of %llu MiB%s%s\n", "30 GiB requested",
                (unsigned long long) (p.minimum >> 20), (unsigned long long) (p.granted >> 20),
                (unsigned long long) (p.requested >> 20), p.error != 0 ? ", refusal error " : "",
                p.error != 0 ? std::to_string(p.error).c_str() : "");
    if (!shape) std::printf("large pages %-50s   <-- WRONG\n", "shape");
    return !(shape && explained);
}
}  // namespace

int main() {
    int fail_keeps = file_cache_keeps_cases() | ram_floor_cases() | read_ahead_cases() | drop_view_cases() |
                    guard_reclaim_cases() | reclaim_due_cases() | exhausted_message_cases() | dxgi_cases() | large_page_cases() | vram_plan_cases() | spill_cases() | vram_reserve_cases() | vram_reserve_keep_cases();
    bool reduced = false;
    const uint64_t bytes = lock_test_bytes(reduced);
    if (bytes == 0) {
        std::printf("lock_resident: SKIPPED, the locked-memory limit is too low to lock a region (raise ulimit -l)\n");
        std::printf("platform_memory_test: %s\n", fail_keeps ? "FAILED" : "OK (lock_resident skipped)");
        return fail_keeps;
    }
    if (reduced)
        std::printf("lock_resident: the 256 MiB case is SKIPPED, the locked-memory limit is lower; locking %llu KiB instead\n",
                    (unsigned long long) (bytes >> 10));
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const basalt::platform::LockResult r = basalt::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu KiB of %llu KiB (%s)\n", (int) r.ok,
                (unsigned long long) (r.locked_bytes >> 10), (unsigned long long) (bytes >> 10), r.note.c_str());
    int fail = !(r.ok && r.locked_bytes == bytes) | fail_keeps;
    basalt::platform::unlock_resident(p, bytes);
#if defined(_WIN32)
    // A successful lock proves nothing on its own: an elevated token locks the region with a soft minimum too. The
    // flag readback is what shows the quota call set the minimum as hard, and only a hard minimum keeps the locked
    // pages from being trimmed back out.
    SIZE_T ws_min = 0, ws_max = 0;
    DWORD ws_flags = 0;
    const bool hard_min = GetProcessWorkingSetSizeEx(GetCurrentProcess(), &ws_min, &ws_max, &ws_flags) &&
                          (ws_flags & QUOTA_LIMITS_HARDWS_MIN_ENABLE);
    std::printf("working set: min=%llu KiB max=%llu KiB hard-min=%d%s\n", (unsigned long long) (ws_min >> 10),
                (unsigned long long) (ws_max >> 10), (int) hard_min, hard_min ? "" : "   <-- WRONG");
    fail |= r.ok && !hard_min;
#endif
    // With a progress callback the lock runs in chunks: the counts only rise and the last one is the whole region.
    std::vector<uint64_t> seen;
    const basalt::platform::LockResult rp =
        basalt::platform::lock_resident(p, bytes, [&](uint64_t done) { seen.push_back(done); });
    bool rising = !seen.empty() && seen.back() == bytes;
    for (size_t i = 1; i < seen.size(); ++i) rising = rising && seen[i] > seen[i - 1];
    std::printf("lock_resident with progress: ok=%d %zu reports%s\n", (int) rp.ok, seen.size(),
                rising ? "" : "   <-- WRONG");
    fail |= !(rp.ok && rp.locked_bytes == bytes && rising);
    basalt::platform::unlock_resident(p, bytes);
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
