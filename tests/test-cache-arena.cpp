// Unit tests for server_kv_arena (contiguous buffer + best-fit free-list).
// Header-only, no llama deps, so this test stays self-contained.
//
// These exercise the properties that matter for the prompt-cache redesign:
//   - allocations are bounded by capacity (no overflow even under churn)
//   - freed slices are recycled by the free-list (RSS bounded, no fragmentation)
//   - best-fit + coalescing keep the arena compact over alloc/free cycles

#include "../tools/server/server-cache-arena.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// fill a slice with a marker so we can detect overlap/corruption
static void fill(server_kv_arena & a, size_t off, size_t n, uint8_t v) {
    memset(a.data(off), v, n);
}

// verify every live allocation holds a stable, non-overlapping region
struct alloc { size_t off; size_t n; uint8_t v; };

static void check_no_overlap(const std::vector<alloc> & live) {
    std::set<size_t> seen;
    for (const auto & x : live) {
        for (size_t i = 0; i < x.n; ++i) {
            size_t p = x.off + i;
            CHECK(seen.find(p) == seen.end()); // no double-use of a byte
            seen.insert(p);
        }
    }
}

static void test_basic_alloc_free() {
    server_kv_arena a(1024);
    CHECK(a.ready());
    CHECK(a.capacity() == 1024);

    size_t o1 = a.alloc(100);
    CHECK(o1 == 0);
    CHECK(a.used() == 104); // 100 -> 8-aligned = 104

    size_t o2 = a.alloc(100);
    CHECK(o2 == 104);
    CHECK(a.used() == 208);

    fill(a, o1, 100, 0xAA);
    fill(a, o2, 100, 0xBB);
    CHECK(a.data(o1)[0] == 0xAA);
    CHECK(a.data(o2)[0] == 0xBB);

    a.free(o1, 100);
    CHECK(a.used() == 104); // o1 returned
    a.free(o2, 100);
    CHECK(a.used() == 0);

    std::printf("test_basic_alloc_free: ok\n");
}

static void test_free_list_reuse() {
    // capacity 512, churn many allocations -> used must stay bounded by capacity
    server_kv_arena a(512);
    std::vector<alloc> live;
    uint8_t v = 1;
    size_t total_alloc_calls = 0;
    for (int i = 0; i < 1000; ++i) {
        size_t n = 32 + (i % 17) * 8; // varied sizes, always <= capacity
        size_t off = a.alloc(n);
        while (off == SIZE_MAX) {
            // arena full for this request: free live entries until it fits.
            // (a single freed entry may be too small, so free in a loop)
            if (live.empty()) {
                CHECK(!"arena cannot fit a request that should fit in capacity");
                break;
            }
            a.free(live.back().off, live.back().n);
            live.pop_back();
            off = a.alloc(n);
        }
        if (off == SIZE_MAX) {
            break;
        }
        fill(a, off, n, v);
        live.push_back({off, n, v++});
        ++total_alloc_calls;

        // invariant: never exceed capacity
        CHECK(a.used() <= a.capacity());
        check_no_overlap(live);
    }
    // after heavy churn, committed (high-water) must still be within capacity
    CHECK(a.committed() <= a.capacity());
    CHECK(a.used() <= a.capacity());
    std::printf("test_free_list_reuse: %zu allocs, used=%zu/%zu, committed=%zu -> ok\n",
                total_alloc_calls, a.used(), a.capacity(), a.committed());
}

static void test_coalescing() {
    // allocate 3 contiguous slices, free them all -> they must coalesce so a
    // single big allocation fits again (no external fragmentation)
    server_kv_arena a(300);
    size_t o1 = a.alloc(80); // -> 80 (8-aligned)
    size_t o2 = a.alloc(80); // -> 80
    size_t o3 = a.alloc(80); // -> 80  total 240
    CHECK(o1 != SIZE_MAX && o2 != SIZE_MAX && o3 != SIZE_MAX);
    CHECK(a.used() == 240);

    a.free(o1, 80);
    a.free(o2, 80);
    a.free(o3, 80);
    CHECK(a.used() == 0);

    // after coalescing, the whole region should be reusable as one slice
    // (free-list merges the three freed blocks: {0, 240})
    size_t obig = a.alloc(240);
    CHECK(obig != SIZE_MAX);
    CHECK(a.used() == 240);
    std::printf("test_coalescing: ok\n");
}

static void test_oversize_fails() {
    server_kv_arena a(128);
    CHECK(a.alloc(200) == SIZE_MAX); // bigger than capacity
    CHECK(a.used() == 0);
    std::printf("test_oversize_fails: ok\n");
}

static void test_clear() {
    server_kv_arena a(256);
    a.alloc(100);
    CHECK(a.used() > 0);
    a.clear();
    CHECK(a.used() == 0);
    CHECK(a.committed() == 0);
    CHECK(a.capacity() == 256); // reservation kept
    std::printf("test_clear: ok\n");
}

int main() {
    test_basic_alloc_free();
    test_free_list_reuse();
    test_coalescing();
    test_oversize_fails();
    test_clear();

    if (g_failures == 0) {
        std::printf("all arena tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d arena test failures\n", g_failures);
    return 1;
}
