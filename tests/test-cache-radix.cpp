// Unit tests for server_radix_index_impl (per-token trie for prompt-cache LCP).
//
// The index is generic over the entry handle, so this test feeds it a
// lightweight stand-in (test_entry with n_tokens()) and std::vector<llama_token>
// as the token sequence. No server/llama linking is exercised - this validates
// the trie/refcount/prune/best-candidate logic in isolation.
//
// These exercise the properties that matter for the prompt-cache redesign:
//   - longest_prefix returns the true shared prefix depth
//   - a shorter query still hits a longer cached entry (shared system prompt)
//   - among candidates sharing a prefix the smallest entry wins (max keep ratio)
//   - the f_keep >= 0.25 threshold prevents trashing large prompts
//   - erase refcounts and prunes; clear drops everything

#include "../tools/server/server-cache-radix.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// minimal entry: just needs n_tokens() and address stability
struct test_entry {
    int n = 0;
    int n_tokens() const { return n; }
};

using seq = std::vector<llama_token>;
using radix = server_radix_index_impl<test_entry*>;

static seq s(std::initializer_list<int> xs) {
    return seq(xs.begin(), xs.end());
}

static void test_exact_match() {
    radix r;
    test_entry e{3};
    r.insert(s({1, 2, 3}), &e);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({1, 2, 3}), 3, lcp);
    CHECK(got == &e);
    CHECK(lcp == 3);
    std::printf("test_exact_match: ok\n");
}

static void test_no_match() {
    radix r;
    test_entry e{3};
    r.insert(s({1, 2, 3}), &e);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({4, 5, 6}), 3, lcp);
    CHECK(got == nullptr); // diverges at the root
    CHECK(lcp == 0);
    std::printf("test_no_match: ok\n");
}

static void test_query_longer_than_entry() {
    // cached entry is a prefix of the query: full entry is reusable
    radix r;
    test_entry e{3};
    r.insert(s({1, 2, 3}), &e);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({1, 2, 3, 4, 5}), 5, lcp);
    CHECK(got == &e);
    CHECK(lcp == 3);
    std::printf("test_query_longer_than_entry: ok\n");
}

static void test_shared_prefix_longer_entry() {
    // shared system-prompt case: entry longer than query, query shares a prefix
    radix r;
    test_entry e{5};
    r.insert(s({1, 2, 3, 4, 5}), &e);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({1, 2, 3}), 3, lcp);
    CHECK(lcp == 3);
    // f_keep = lcp/esize = 3/5 = 0.6 >= 0.25 -> selected
    CHECK(got == &e);
    std::printf("test_shared_prefix_longer_entry: ok\n");
}

static void test_pick_smallest_candidate() {
    // two entries share the queried prefix; the smaller one maximizes keep
    radix r;
    test_entry big{4};
    test_entry small{3};
    r.insert(s({1, 2, 3, 4}), &big);
    r.insert(s({1, 2, 3}), &small);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({1, 2, 3}), 3, lcp);
    CHECK(lcp == 3);
    CHECK(got == &small); // esize 3 < 4
    std::printf("test_pick_smallest_candidate: ok\n");
}

static void test_f_keep_threshold_rejects_large() {
    // a single very large entry sharing only 1 token: keep ratio too low
    radix r;
    test_entry huge{10};
    r.insert(s({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}), &huge);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({1}), 1, lcp);
    CHECK(lcp == 1);
    // f_keep = 1/10 = 0.1 < 0.25 -> not selected
    CHECK(got == nullptr);
    std::printf("test_f_keep_threshold_rejects_large: ok\n");
}

static void test_erase_then_miss() {
    radix r;
    test_entry e{3};
    r.insert(s({1, 2, 3}), &e);

    r.erase(s({1, 2, 3}), &e);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({1, 2, 3}), 3, lcp);
    // path pruned (n_entries hit 0) -> no candidate
    CHECK(got == nullptr);
    std::printf("test_erase_then_miss: ok\n");
}

static void test_erase_refcount_keeps_shared_branch() {
    // insert two entries along the same path; erasing one must not prune the
    // branch still referenced by the other
    radix r;
    test_entry e1{3};
    test_entry e2{4};
    r.insert(s({1, 2, 3}), &e1);
    r.insert(s({1, 2, 3, 4}), &e2);

    r.erase(s({1, 2, 3}), &e1);

    int lcp = -1;
    // e2 must still be reachable through the shared [1,2,3] prefix
    test_entry * got = r.longest_prefix(s({1, 2, 3, 4}), 4, lcp);
    CHECK(got == &e2);
    CHECK(lcp == 4);

    // and the shorter query still finds e2 (shared-prefix-longer-entry rule)
    got = r.longest_prefix(s({1, 2, 3}), 3, lcp);
    CHECK(got == &e2);
    std::printf("test_erase_refcount_keeps_shared_branch: ok\n");
}

static void test_erase_wrong_entry_noop() {
    radix r;
    test_entry e1{3};
    test_entry e2{3};
    r.insert(s({1, 2, 3}), &e1);

    // erasing a handle that was never the terminal entry does nothing harmful
    r.erase(s({1, 2, 3}), &e2);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({1, 2, 3}), 3, lcp);
    CHECK(got == &e1); // e1 still indexed
    std::printf("test_erase_wrong_entry_noop: ok\n");
}

static void test_clear() {
    radix r;
    test_entry e1{3}, e2{3};
    r.insert(s({1, 2, 3}), &e1);
    r.insert(s({4, 5, 6}), &e2);

    r.clear();

    int lcp = -1;
    CHECK(r.longest_prefix(s({1, 2, 3}), 3, lcp) == nullptr);
    CHECK(r.longest_prefix(s({4, 5, 6}), 3, lcp) == nullptr);
    std::printf("test_clear: ok\n");
}

static void test_empty_entry() {
    // an empty-length entry sits at the root and matches any query
    radix r;
    test_entry e{0};
    r.insert(seq{}, &e);

    int lcp = -1;
    test_entry * got = r.longest_prefix(s({1, 2, 3}), 3, lcp);
    CHECK(lcp == 0);
    CHECK(got == &e); // esize 0 -> f_keep clamped to 1.0
    std::printf("test_empty_entry: ok\n");
}

static void test_churn_no_leak() {
    // many insert/erase cycles over overlapping prefixes must not leave phantom
    // candidates (refcounts/pruning stay consistent)
    radix r;
    std::vector<test_entry> es(200);
    for (int i = 0; i < 200; ++i) {
        es[i].n = 3;
        r.insert(s({i, i + 1, i + 2}), &es[i]);
    }
    for (int i = 0; i < 200; ++i) {
        r.erase(s({i, i + 1, i + 2}), &es[i]);
    }
    int lcp = -1;
    for (int i = 0; i < 200; ++i) {
        CHECK(r.longest_prefix(s({i, i + 1, i + 2}), 3, lcp) == nullptr);
    }
    std::printf("test_churn_no_leak: ok\n");
}

int main() {
    test_exact_match();
    test_no_match();
    test_query_longer_than_entry();
    test_shared_prefix_longer_entry();
    test_pick_smallest_candidate();
    test_f_keep_threshold_rejects_large();
    test_erase_then_miss();
    test_erase_refcount_keeps_shared_branch();
    test_erase_wrong_entry_noop();
    test_clear();
    test_empty_entry();
    test_churn_no_leak();

    if (g_failures == 0) {
        std::printf("all radix tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d radix test failures\n", g_failures);
    return 1;
}
