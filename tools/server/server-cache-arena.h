#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <algorithm>

// Contiguous host-byte buffer with a best-fit free-list + lazy commit.
//
// Replaces the per-entry std::vector<uint8_t> alloc/clear/shrink_to_fit churn in
// server_prompt_cache that caused heap fragmentation. All cached KV state blobs
// live as {offset, size} slices inside ONE backing reservation, so the cache
// requests its virtual budget exactly once (on resize) and reuses slices via a
// coalescing free-list.
//
// Memory model:
//   - resize(cap) does a single `new uint8_t[cap]`. For large caps glibc serves
//     this with a demand-paged mmap region, so untouched pages cost no RSS -
//     only pages actually written (via alloc + the caller filling the slice)
//     commit physical memory. Total RSS therefore grows on demand, bounded by
//     cap, exactly like the old per-vector model but without alloc/free churn.
//   - alloc() first tries the free-list (best-fit), then carves from the
//     high-water committed tail. Returns SIZE_MAX if neither has room (caller
//     evicts LRU entries and retries).
//   - free() returns a slice to the free-list and coalesces neighbours.
//
// This mirrors ggml_dyn_tallocr (ggml-alloc.c) but for raw host bytes.
class server_kv_arena {
public:
    server_kv_arena() = default;

    explicit server_kv_arena(size_t capacity) {
        resize(capacity);
    }

    // (Re)claim a virtual reservation of up to `capacity` bytes. Drops all
    // outstanding slices. Large reservations are demand-paged (mmap), but the
    // allocation itself can still fail under restrictive overcommit, so we halve
    // the request down to a floor rather than give up. The actual capacity() may
    // therefore be smaller than requested; alloc() returns SIZE_MAX if no room.
    //
    // Note: the free-list starts EMPTY and `committed_` starts at 0 - the whole
    // buffer is the uncommitted tail. alloc() carves fresh slices from the tail
    // and recycles freed slices from the free-list. Mixing a pre-seeded free-list
    // block with a zero-based tail would let both paths hand out offset 0 and
    // overlap, so we deliberately do NOT seed the free-list here.
    void resize(size_t capacity) {
        capacity_  = 0;
        committed_ = 0;
        used_      = 0;
        free_.clear();
        buf_.reset();

        if (capacity == 0) {
            return;
        }
        // try the requested capacity directly; if that fails (e.g. restrictive
        // overcommit on a large reservation) halve down to a 256 MiB floor.
        // Sub-floor requests are tried once as-is.
        size_t cap = capacity;
        const size_t floor = 256ull * 1024 * 1024;
        do {
            buf_.reset(new (std::nothrow) uint8_t[cap]);
            if (buf_) {
                capacity_ = cap;
                return;
            }
            if (cap <= floor) {
                break;
            }
            cap = std::max(floor, cap / 2);
        } while (true);
    }

    size_t capacity()   const { return capacity_; }
    size_t committed()  const { return committed_; }
    size_t used()       const { return used_; }
    size_t free_bytes() const { return capacity_ - used_; }
    bool   ready()      const { return buf_ != nullptr; }

    // Allocate `n` bytes (8-byte aligned). Returns offset, or SIZE_MAX if no
    // room (caller must evict and retry).
    size_t alloc(size_t n) {
        if (n == 0) {
            return 0;
        }
        if (!ready()) {
            return SIZE_MAX;
        }
        n = align_up(n, kAlign);

        // 1. best-fit in the free-list
        size_t best   = SIZE_MAX;
        size_t best_s = SIZE_MAX;
        for (size_t i = 0; i < free_.size(); ++i) {
            const size_t s = free_[i].size;
            if (s >= n && s < best_s) {
                best   = i;
                best_s = s;
                if (s == n) {
                    break; // exact fit
                }
            }
        }
        if (best != SIZE_MAX) {
            const block b = free_[best];
            const size_t off = b.off;
            if (b.size > n) {
                free_[best] = block{b.off + n, b.size - n};
            } else {
                free_.erase(free_.begin() + best);
            }
            used_ += n;
            return off;
        }

        // 2. carve from the committed tail
        if (committed_ + n > capacity_) {
            return SIZE_MAX;
        }
        const size_t off = committed_;
        committed_ += n;
        used_      += n;
        return off;
    }

    // Release a slice previously returned by alloc(). `n` is the ORIGINAL
    // (unaligned) request size; internally re-aligned to match alloc().
    void free(size_t off, size_t n) {
        if (n == 0) {
            return;
        }
        n = align_up(n, kAlign);
        if (!ready() || off + n > committed_) {
            return;
        }
        used_ -= n;

        const block nb{off, n};
        auto it = std::lower_bound(free_.begin(), free_.end(), nb,
            [](const block & a, const block & b) { return a.off < b.off; });
        it = free_.insert(it, nb);

        // coalesce with the following block
        if (it + 1 != free_.end() && it->off + it->size == (it + 1)->off) {
            it->size += (it + 1)->size;
            free_.erase(it + 1);
        }
        // coalesce with the preceding block
        if (it != free_.begin() && (it - 1)->off + (it - 1)->size == it->off) {
            (it - 1)->size += it->size;
            free_.erase(it);
        }
    }

    uint8_t *       data(size_t off)       { return buf_.get() + off; }
    const uint8_t * data(size_t off) const { return buf_.get() + off; }

    void clear() {
        committed_ = 0;
        used_      = 0;
        free_.clear();
        // keep the reservation (buf_) intact
    }

private:
    struct block {
        size_t off;
        size_t size;
    };

    static constexpr size_t kAlign = 8;
    static size_t align_up(size_t x, size_t a) {
        return (x + a - 1) & ~(a - 1);
    }

    size_t                      capacity_  = 0;
    size_t                      committed_ = 0; // high-water mark of the tail
    size_t                      used_      = 0; // bytes currently handed out
    std::unique_ptr<uint8_t[]>  buf_;           // lazy-commit virtual reservation
    std::vector<block>          free_;          // free blocks, sorted by offset
};
