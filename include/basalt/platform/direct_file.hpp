// include/basalt/platform/direct_file.hpp - unbuffered asynchronous file reads.
//
// The n-gram table (26.8 GiB) stays on the SSD and must never occupy RAM, including the OS file cache. A memory
// map cannot promise that; an unbuffered read can. Linux reads it with O_DIRECT preads issued from a thread
// pool (submit only queues), so many reads are in flight. Windows reads the same file with FILE_FLAG_NO_BUFFERING
// overlapped reads completed through an I/O completion port; the queue of reads is the same either way.
//
// Contract of every read: offset, length and buffer address are multiples of `alignment()` (4096 here).
// Reads past end of file return the bytes that exist; `Completion::bytes` says how many.
#pragma once

#include <cstdint>
#include <string>

namespace basalt::platform {

struct Completion {
    uint64_t tag = 0;      ///< the caller's tag from `submit`
    uint32_t bytes = 0;    ///< bytes transferred (short only at end of file)
    bool ok = false;
};

/// What one reader, or every open reader together, has done: the queue a diagnostic has to see to tell a slow drive
/// from a queue that never went deep.
struct IoStats {
    int issuing_threads = 0;     ///< threads that submit reads
    uint64_t queued = 0;         ///< submitted, not yet handed to the OS
    uint64_t inflight = 0;       ///< with the OS, not yet completed
    uint64_t peak_inflight = 0;  ///< the deepest that queue ever got: the concurrency the drive actually saw
    uint64_t submitted = 0;      ///< requests accepted since open
    uint64_t completed = 0;      ///< completions retrieved (a caller's or a teardown's)
    uint64_t bytes = 0;          ///< bytes those completions carried
    uint64_t failed = 0;         ///< completions that reported an error
};

class DirectFile {
public:
    DirectFile();
    ~DirectFile();
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;

    bool open(const std::string& path, std::string& err);
    void close();
    bool is_open() const;
    uint64_t size() const;
    static constexpr uint32_t alignment() { return 4096; }

    /// Queue one read. Returns false (and sets `err`) if the request could not be queued; a queued request
    /// always produces exactly one completion.
    bool submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err);

    /// Wait for up to `max` completions; returns how many were written to `out`. `timeout_ms` < 0 waits
    /// forever; 0 polls. A `wake()` shows up as one completion with tag `WAKE_TAG`.
    int wait(Completion* out, int max, int timeout_ms);

    /// Make a thread blocked in `wait` return (thread-safe). Hands it newly queued work.
    static constexpr uint64_t WAKE_TAG = ~0ull;
    void wake();

    /// Allocation helpers for aligned read buffers (page-aligned, never touched by the cache manager).
    static void* alloc_aligned(size_t bytes);
    static void free_aligned(void* p);

    /// This reader's own counters. Reading them takes the reader's lock, so call it from a diagnostic, not from the
    /// read path.
    IoStats stats() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

/// Monotonic time in microseconds, for latency accounting.
double now_us();

/// Every open reader's counters added up: the engine's file I/O as one line. Zero when nothing is open.
IoStats io_stats();

/// What a one-time small-read probe measured.
struct ReadProbe {
    bool ok = false;
    uint64_t reads = 0;
    uint64_t bytes = 0;
    uint64_t failed = 0;
    uint32_t read_bytes = 0;   ///< the size of each read, as asked for
    uint32_t depth = 0;        ///< how many were in flight at once
    double seconds = 0;
    std::string error;
    double reads_per_s() const { return seconds > 0 ? (double) reads / seconds : 0.0; }
    double gib_per_s() const { return seconds > 0 ? (double) bytes / 1073741824.0 / seconds : 0.0; }
};
/// The rate `reads` reads of `read_bytes` reach on `path`, issued `depth` at a time from a reader of its own (the
/// callers' readers are untouched). Spread over the file, so this is the small-read contract and not a sequential
/// rate. `ok` is false when the file could not be opened or the request is not one the contract allows.
ReadProbe probe_reads(const std::string& path, uint32_t read_bytes, uint32_t depth, uint32_t reads);

}  // namespace basalt::platform
