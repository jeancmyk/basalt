// include/basalt/core/gdn_slots.hpp - device slots that hold a checkpoint's GDN recurrence.
//
// The recurrence cannot be recomputed from K/V, so a checkpoint without it is unusable, and it is the largest part of
// one (~118 MB).  A slot keeps a checkpoint's state device-resident, so restoring it copies device to device instead
// of over the host.  The session keeps its own buffer: the commit graph has the state's address baked in (verify.cpp,
// capture_commit), so a slot is a mirror of that state, never a donation of it.
#pragma once

#include "basalt/core/conversation_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

/// Which slot holds which seqlen, and which are free.  Pure bookkeeping: the caller owns the buffers, so the
/// allocation policy can be walked in a test without a device.
class GdnSlotBook {
public:
    /// `slots` free slots, none holding a state.
    void reset(int slots) {
        seqlen_.assign((size_t) (slots > 0 ? slots : 0), -1);
        free_.clear();
        for (int i = (int) seqlen_.size(); i-- > 0;) free_.push_back(i);
    }
    int slots() const { return (int) seqlen_.size(); }
    /// A free slot for a checkpoint of `seqlen` tokens, or -1 when none is free.
    int take(int64_t seqlen) {
        if (free_.empty()) return -1;
        const int slot = free_.back();
        free_.pop_back();
        seqlen_[(size_t) slot] = seqlen;
        return slot;
    }
    void release(int slot) {
        if (slot < 0 || slot >= slots() || seqlen_[(size_t) slot] < 0) return;
        seqlen_[(size_t) slot] = -1;
        free_.push_back(slot);
    }
    /// The seqlen the slot holds, or -1 for a free or invalid slot.
    int64_t seqlen(int slot) const {
        return slot >= 0 && slot < slots() ? seqlen_[(size_t) slot] : -1;
    }
    int used() const {
        int n = 0;
        for (int64_t s : seqlen_) n += s >= 0 ? 1 : 0;
        return n;
    }
    /// Slots a checkpoint may still take.  `slots() - used()` is the same number; this names it for the tests that
    /// check a dropped checkpoint gives its slot back.
    int free() const { return (int) free_.size(); }

private:
    std::vector<int64_t> seqlen_;   // -1: free
    std::vector<int> free_;
};

/// The device buffers behind the book, on one device.
class GdnSlots {
public:
    GdnSlots() = default;
    ~GdnSlots();
    GdnSlots(const GdnSlots&) = delete;
    GdnSlots& operator=(const GdnSlots&) = delete;

    /// Allocates `n` buffers of `bytes` each on the current device.  False (and none allocated) when they do not
    /// fit: a checkpoint then keeps host bytes.
    bool reserve(int n, size_t bytes, std::string& error);
    bool empty() const { return book_.slots() == 0; }
    int slots() const { return book_.slots(); }
    uint64_t bytes() const { return (uint64_t) book_.slots() * bytes_; }
    /// The slot's device buffer (null when the slot is invalid or the pool is empty).
    void* ptr(int slot) const;
    GdnSlotBook& book() { return book_; }
    const GdnSlotBook& book() const { return book_; }
    int device() const { return device_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    GdnSlotBook book_;
    size_t bytes_ = 0;
    int device_ = -1;
};

/// A checkpoint copied out of the session's chain keeps no slot: the host bytes are what travels with it.  The caller
/// fills them (D2H from `ptr(slot)`) and then calls this; a slot index is valid only while that chain owns it.
inline void gdn_slot_detach(GdnSlotBook& book, ConversationCheckpoint& c) {
    if (c.gdn_slot < 0) return;
    book.release(c.gdn_slot);
    c.gdn_slot = -1;
}

}  // namespace basalt::core
