#pragma once

#include "llama.h"

#include <memory>
#include <unordered_map>

struct server_prompt;
struct server_tokens;

// Per-token radix/trie index over cached prompts.
//
// Replaces the O(N*L) linear get_common_prefix scan in server_prompt_cache::load
// with an O(matched-prefix-length) longest-common-prefix lookup. Each cached
// prompt is inserted token-by-token, and EVERY node along its path is marked
// (refcounted), so a query that shares only a prefix with a longer cached entry
// still hits (the shared system-prompt case). Prompts sharing a prefix share
// trie nodes.
//
// Lookup walks the query tokens until it diverges from every cached prompt; the
// depth reached is the longest prefix shared with any cached entry. The best
// candidate entry (maximizing keep/sim ratios, matching the legacy heuristic) is
// then located with a bounded DFS in the subtree under that depth.
class server_radix_index {
public:
    // Index entry `e` along its full token sequence. `e` must remain stable and
    // its tokens unchanged while indexed.
    void insert(const server_tokens & toks, server_prompt * e);

    // Remove entry `e` (refcount-decrement along its path, prune empty branches).
    void erase(const server_tokens & toks, server_prompt * e);

    // Drop everything.
    void clear();

    // Walk `toks`. Returns the longest prefix length shared with any cached
    // entry, and the best candidate entry sharing it (or nullptr). `qsize` is
    // the query length, used for the similarity tie-break.
    server_prompt * longest_prefix(const server_tokens & toks, int qsize, int & lcp_len) const;

private:
    struct node {
        // entries whose path passes through this node (refcount)
        int n_entries = 0;
        // non-null when an entry terminates exactly at this node
        server_prompt * entry = nullptr;
        // children keyed by next token
        std::unordered_map<llama_token, std::unique_ptr<node>> children;
    };

    std::unique_ptr<node> root_ = std::make_unique<node>();

    // bounded DFS: collect candidate entries in subtree, return best by keep/sim
    server_prompt * best_in_subtree(const node * n, int lcp_len, int qsize, int budget) const;
};
