// src/platform/direct_file.cpp - see include/basalt/platform/direct_file.hpp.
#include "basalt/platform/direct_file.hpp"

#include "basalt/platform/file.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace basalt::platform {

double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

void* DirectFile::alloc_aligned(size_t bytes) {
#if defined(_WIN32)
    // A fresh VirtualAlloc block starts at an allocation granularity boundary, so it satisfies the unbuffered
    // contract without the cache manager ever having held these pages.
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment(), bytes) == 0 ? p : nullptr;
#endif
}

void DirectFile::free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

namespace {
/// A queued read: `submit` only queues it; the pool's threads issue it .  One thread
/// issuing every read would limit the n-gram table's prompt reads to the kernel time per read, while an NVMe
/// drive serves several times that at depth.
struct Pending {
    uint64_t offset;
    void* buffer;
    uint32_t length;
    uint64_t tag;
};

/// The number of issuing threads, clamped to 1..64. On Linux each thread does one blocking pread, so the thread
/// count is the queue depth. On Windows a read stays in flight after its issuing thread moves on, so the count
/// only bounds how fast the submitted queue reaches the drive; the depth itself is what the caller queued.
int io_threads(int dflt) { return std::clamp(dflt, 1, 64); }

/// A reader's counters, with the depth they describe. Every update happens under the reader's own lock, so none of
/// this is atomic: `enter` is the moment a read leaves the caller's queue for the OS.
struct Counters {
    IoStats s;
    size_t inflight = 0;
    void enter() {
        ++inflight;
        if ((uint64_t) inflight > s.peak_inflight) s.peak_inflight = (uint64_t) inflight;
    }
    void leave() { --inflight; }
    void completion(uint32_t bytes, bool ok) {
        ++s.completed;
        s.bytes += bytes;
        if (!ok) ++s.failed;
    }
    void reset() { s = IoStats{}; inflight = 0; }
};

/// Every open reader, so a diagnostic can add their counters up. This lock is always taken before a reader's own, so
/// a reader can be added and removed without the two orders ever crossing.
std::mutex g_registry_mu;
std::vector<DirectFile*> g_registry;

void register_reader(DirectFile* f) {
    std::lock_guard<std::mutex> lk(g_registry_mu);
    g_registry.push_back(f);
}

void unregister_reader(DirectFile* f) {
    std::lock_guard<std::mutex> lk(g_registry_mu);
    g_registry.erase(std::remove(g_registry.begin(), g_registry.end(), f), g_registry.end());
}
}  // namespace

IoStats io_stats() {
    std::lock_guard<std::mutex> lk(g_registry_mu);
    IoStats all;
    for (DirectFile* f : g_registry) {
        const IoStats s = f->stats();
        all.issuing_threads += s.issuing_threads;   // threads add up: the count of threads that issue
        all.queued += s.queued;
        all.inflight += s.inflight;
        all.peak_inflight = std::max(all.peak_inflight, s.peak_inflight);   // a peak, so the deepest reader's
        all.submitted += s.submitted;
        all.completed += s.completed;
        all.bytes += s.bytes;
        all.failed += s.failed;
    }
    return all;
}

#if defined(_WIN32)
// ------------------------------------------------------------------------------------------------ Windows
namespace {
/// One in-flight request. The OVERLAPPED is the first member so that a completion packet's OVERLAPPED* converts
/// back to the request it belongs to.
struct Req {
    OVERLAPPED ov;
    uint64_t tag;
};

/// The completion key of a posted packet that carries no request. A `wake` is turned into a WAKE_TAG completion
/// for the caller; the nudge of an immediate completion only says the internal buffer has something in it, and
/// never reaches the caller, so `wake()` keeps its meaning. The key of a real read is the one the file was
/// associated with (0) and it always carries an OVERLAPPED.
constexpr ULONG_PTR WAKE_KEY = 1;
constexpr ULONG_PTR NUDGE_KEY = 2;
}  // namespace

struct DirectFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE port = nullptr;
    uint64_t size = 0;
    std::mutex mu;                      // guards everything below
    std::deque<Req*> free_reqs;
    std::vector<Req*> all_reqs;
    std::deque<Completion> immediate;   // completions ready to hand out: a read that hit end of file or failed
                                        // without a port packet, a wake, and any that did not fit the last wait
    std::deque<Pending> queue;          // submitted, not yet issued
    Counters c;                         // reads whose completion packet has not been retrieved yet, and the totals
    std::condition_variable cv;
    bool stop = false;
    std::vector<std::thread> pool;

    Req* take() {
        if (free_reqs.empty()) {
            Req* r = new Req();
            all_reqs.push_back(r);
            return r;
        }
        Req* r = free_reqs.front();
        free_reqs.pop_front();
        return r;
    }

    void issue(const Pending& p) {
        Req* r;
        {
            std::lock_guard<std::mutex> lk(mu);
            r = take();
            // Counted before the read starts: a read that finishes at once can have its packet retrieved by a
            // waiter before this thread would get the chance to record it, and a count that goes past zero leaves
            // close() waiting for a packet that has already been taken.
            c.enter();
        }
        std::memset(&r->ov, 0, sizeof r->ov);
        r->ov.Offset = (DWORD) (p.offset & 0xFFFFFFFFull);
        r->ov.OffsetHigh = (DWORD) (p.offset >> 32);
        r->tag = p.tag;
        if (!ReadFile(file, p.buffer, p.length, nullptr, &r->ov)) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                std::lock_guard<std::mutex> lk(mu);
                free_reqs.push_back(r);
                c.leave();   // a read the kernel never took produces no packet
                // at or past end of file: a zero-byte completion; anything else: a failed one
                c.completion(0, e == ERROR_HANDLE_EOF);   // no packet will arrive to count it
                immediate.push_back(Completion{p.tag, 0, e == ERROR_HANDLE_EOF});
                PostQueuedCompletionStatus(port, 0, NUDGE_KEY, nullptr);   // tell a waiter the buffer has work
                return;
            }
        }
    }

    /// Retrieve the packet of every read still in flight and hand its request back to the free list. A packet is
    /// queued only once the kernel has stopped writing the OVERLAPPED and the caller's buffer, so after this no
    /// late completion can land in memory the caller has released. Cancelling keeps a teardown from waiting out
    /// reads nobody wants any more; a cancelled read still produces a packet.
    void drain() {
        if (file != INVALID_HANDLE_VALUE) CancelIoEx(file, nullptr);
        if (port == nullptr) return;
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(mu);
                if (c.inflight == 0) return;
            }
            DWORD bytes = 0;
            ULONG_PTR key = 0;
            OVERLAPPED* ov = nullptr;
            if (!GetQueuedCompletionStatus(port, &bytes, &key, &ov, INFINITE)) return;
            if (ov == nullptr) continue;   // a posted wake or nudge: it carries no request
            // The status is in the request's OVERLAPPED, which is still ours until it goes back to the free list.
            const uint32_t status = (uint32_t) ((Req*) ov)->ov.Internal;
            std::lock_guard<std::mutex> lk(mu);
            free_reqs.push_back((Req*) ov);
            c.leave();
            c.completion(bytes, status == 0 || status == 0xC0000011u);
        }
    }

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            issue(p);
            lk.lock();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() {
    close();
    for (Req* r : impl_->all_reqs) delete r;
    delete impl_;
}

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
    // NO_BUFFERING: the cache manager never holds these pages, so the table cannot grow into RAM.
    // RANDOM_ACCESS: no read-ahead. OVERLAPPED: many reads in flight, completed through the port below.
    impl_->file = CreateFileW(to_wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        err = "DirectFile: cannot open " + path + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(impl_->file, &sz)) {
        err = "DirectFile: cannot size " + path;
        close();
        return false;
    }
    impl_->size = (uint64_t) sz.QuadPart;
    impl_->port = CreateIoCompletionPort(impl_->file, nullptr, 0, 1);
    if (impl_->port == nullptr) {
        err = "DirectFile: CreateIoCompletionPort failed (error " + std::to_string(GetLastError()) + ")";
        close();
        return false;
    }
    const int n = io_threads(16);
    for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { impl_->worker(); });
    register_reader(this);
    return true;
}

void DirectFile::close() {
    unregister_reader(this);   // a report taken while this runs no longer sees these counters
    impl_->stop_pool();   // after this no thread is left that could issue another read
    impl_->drain();       // ... and every read already in flight has finished writing
    if (impl_->port != nullptr) CloseHandle(impl_->port);
    if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
    impl_->port = nullptr;
    impl_->file = INVALID_HANDLE_VALUE;
    impl_->size = 0;
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->immediate.clear();
    impl_->queue.clear();
    impl_->c.reset();   // the port is gone, so no packet is left to retrieve
}

bool DirectFile::is_open() const { return impl_->file != INVALID_HANDLE_VALUE; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
        ++impl_->c.s.submitted;
    }
    impl_->cv.notify_one();
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    if (max <= 0) return 0;
    using clock = std::chrono::steady_clock;
    const bool timed = timeout_ms >= 0;
    const auto deadline = clock::now() + std::chrono::milliseconds(timed ? timeout_ms : 0);
    int n = 0;
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(impl_->mu);
            while (n < max && !impl_->immediate.empty()) {
                out[n++] = impl_->immediate.front();
                impl_->immediate.pop_front();
            }
        }
        if (n > 0 || !is_open()) return n;   // hand out whatever is ready rather than waiting to fill `max`
        DWORD t = INFINITE;
        if (timed) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - clock::now()).count();
            t = (DWORD) (left > 0 ? left : 0);
        }
        // Packets are taken in batches of up to 64 whatever `max` is: a completion that does not fit the caller's
        // array goes into the buffer for the next call, so `out` never holds more than `max` and one call leaves
        // at most one batch behind in the buffer.
        OVERLAPPED_ENTRY entries[64];
        ULONG got = 0;
        if (!GetQueuedCompletionStatusEx(impl_->port, entries, 64, &got, t, FALSE)) return n;
        std::lock_guard<std::mutex> lk(impl_->mu);
        for (ULONG i = 0; i < got; ++i) {
            if (entries[i].lpOverlapped == nullptr) {
                // The nudge of an immediate completion is already represented by the buffered completion, so it
                // is swallowed here; only a real wake is turned into a completion of its own.
                if (entries[i].lpCompletionKey == WAKE_KEY) {
                    Completion w{WAKE_TAG, 0, true};
                    if (n < max) out[n++] = w;
                    else impl_->immediate.push_back(w);
                }
                continue;
            }
            Req* r = (Req*) entries[i].lpOverlapped;
            const uint32_t status = (uint32_t) r->ov.Internal;   // an NTSTATUS
            Completion c;
            c.tag = r->tag;
            c.bytes = entries[i].dwNumberOfBytesTransferred;
            // STATUS_END_OF_FILE is a legal short read at the table's last page.
            c.ok = status == 0 || status == 0xC0000011u;
            if (n < max) out[n++] = c;
            else impl_->immediate.push_back(c);
            impl_->free_reqs.push_back(r);
            impl_->c.leave();
            impl_->c.completion(c.bytes, c.ok);
        }
        // A batch of nothing but nudges hands the caller nothing: keep waiting until the deadline instead of
        // reporting a completion-less wait that had time left.
        if (n > 0) return n;
    }
}

void DirectFile::wake() {
    if (impl_->port != nullptr) PostQueuedCompletionStatus(impl_->port, 0, WAKE_KEY, nullptr);
}

#else
// ------------------------------------------------------------------------------------------------ POSIX
// A pool of threads, each doing blocking O_DIRECT preads, so reads run in parallel (the thread
// count is the queue depth).
struct DirectFile::Impl {
    int fd = -1;
    uint64_t size = 0;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::deque<Pending> queue;
    std::deque<Completion> done;
    Counters c;                          // the reads with the OS, and the totals
    bool stop = false;
    std::vector<std::thread> pool;

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_work.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            c.enter();   // counted before the unlock: the read is with the OS from here
            lk.unlock();
            const ssize_t got = pread(fd, p.buffer, p.length, (off_t) p.offset);
            lk.lock();
            c.leave();
            c.completion(got < 0 ? 0u : (uint32_t) got, got >= 0);
            done.push_back(Completion{p.tag, got < 0 ? 0u : (uint32_t) got, got >= 0});
            cv_done.notify_all();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_work.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() { close(); delete impl_; }

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }
    struct stat st;
    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;
    const int n = io_threads(16);
    for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { impl_->worker(); });
    register_reader(this);
    return true;
}

void DirectFile::close() {
    unregister_reader(this);   // a report taken while this runs no longer sees these counters
    impl_->stop_pool();
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->size = 0;
    impl_->done.clear();
    impl_->queue.clear();
    impl_->c.reset();
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
        ++impl_->c.s.submitted;
    }
    impl_->cv_work.notify_one();
    return true;
}

void DirectFile::wake() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->done.push_back(Completion{WAKE_TAG, 0, true});
    impl_->cv_done.notify_all();
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    std::unique_lock<std::mutex> lk(impl_->mu);
    if (impl_->done.empty() && timeout_ms != 0) {
        auto ready = [&] { return !impl_->done.empty(); };
        if (timeout_ms < 0) impl_->cv_done.wait(lk, ready);
        else impl_->cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    }
    int n = 0;
    while (n < max && !impl_->done.empty()) {
        out[n++] = impl_->done.front();
        impl_->done.pop_front();
    }
    return n;
}
#endif

IoStats DirectFile::stats() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    IoStats s = impl_->c.s;
    s.issuing_threads = (int) impl_->pool.size();
    s.queued = (uint64_t) impl_->queue.size();
    s.inflight = (uint64_t) impl_->c.inflight;
    return s;
}

ReadProbe probe_reads(const std::string& path, uint32_t read_bytes, uint32_t depth, uint32_t reads) {
    ReadProbe r;
    const uint32_t align = DirectFile::alignment();
    if (read_bytes == 0 || read_bytes % align != 0) { r.error = "probe: the read size is not a multiple of the alignment"; return r; }
    if (reads == 0) { r.error = "probe: no reads asked for"; return r; }
    depth = std::clamp<uint32_t>(depth, 1, 64);
    if (depth > reads) depth = reads;
    DirectFile f;
    std::string err;
    if (!f.open(path, err)) { r.error = err; return r; }
    const uint64_t size = f.size();
    if (size < read_bytes) { r.error = "probe: the file is smaller than one read"; return r; }

    // One buffer per slot of the queue, so a request never lands in a read still in flight.
    std::vector<void*> buf(depth, nullptr);
    for (uint32_t i = 0; i < depth; ++i) buf[i] = DirectFile::alloc_aligned(read_bytes);
    for (uint32_t i = 0; i < depth; ++i) {
        if (buf[i] != nullptr) continue;
        for (uint32_t j = 0; j < depth; ++j) DirectFile::free_aligned(buf[j]);
        r.error = "probe: cannot allocate the read buffers";
        return r;
    }

    // Spread over the whole file, aligned down: this is the small-read contract, not a sequential rate.
    const uint64_t span = size - read_bytes;
    uint64_t stride = reads > 1 ? span / reads / align * align : 0;
    if (stride < align) stride = align;
    auto offset_of = [&](uint64_t i) -> uint64_t {
        const uint64_t off = i * stride;
        return std::min<uint64_t>(off, span) / align * align;
    };

    // `count` reads from `first`, `depth` in flight. Returns the bytes and whether every read came back whole.
    auto run = [&](uint64_t first, uint64_t count, uint64_t* out_bytes, uint64_t* out_failed) -> bool {
        std::vector<uint32_t> free_slots;
        for (uint32_t i = 0; i < depth; ++i) free_slots.push_back(depth - 1 - i);
        uint64_t next = first, done = 0;
        while (done < count) {
            while (next < first + count && !free_slots.empty()) {
                const uint32_t slot = free_slots.back();
                free_slots.pop_back();
                if (!f.submit(offset_of(next), buf[slot], read_bytes, slot, err)) return false;
                ++next;
            }
            Completion got[64];
            const int n = f.wait(got, 64, -1);
            if (n <= 0) return false;
            for (int i = 0; i < n; ++i) {
                if (got[i].tag == DirectFile::WAKE_TAG) continue;
                free_slots.push_back((uint32_t) got[i].tag);
                *out_bytes += got[i].bytes;
                if (!got[i].ok) ++*out_failed;
                ++done;
            }
        }
        return true;
    };

    // One untimed batch first: the file's metadata and the drive's own wake-up are not part of the rate.
    uint64_t warm_bytes = 0, warm_failed = 0;
    if (!run(0, depth, &warm_bytes, &warm_failed)) {
        for (uint32_t j = 0; j < depth; ++j) DirectFile::free_aligned(buf[j]);
        r.error = err.empty() ? "probe: the warm-up reads failed" : err;
        return r;
    }
    const double t0 = now_us();
    if (!run(depth, reads, &r.bytes, &r.failed)) {
        for (uint32_t j = 0; j < depth; ++j) DirectFile::free_aligned(buf[j]);
        r.error = err.empty() ? "probe: the timed reads failed" : err;
        return r;
    }
    r.seconds = (now_us() - t0) / 1e6;
    r.reads = reads;
    r.read_bytes = read_bytes;
    r.depth = depth;
    for (uint32_t j = 0; j < depth; ++j) DirectFile::free_aligned(buf[j]);
    r.ok = r.seconds > 0;
    return r;
}

}  // namespace basalt::platform
