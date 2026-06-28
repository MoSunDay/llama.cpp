#pragma once

#include "llama.h"

#include <climits>
#include <memory>
#include <unordered_map>
#include <vector>

struct server_prompt;

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
//
// The index is generic over the entry handle type (so it can be unit-tested with
// a lightweight stand-in). The sequence type is deduced per call and only needs
// size() and operator[] yielding llama_token. Production aliases it below.
template <typename Entry>
class server_radix_index_impl {
    struct node {
        // entries whose path passes through this node (refcount)
        int n_entries = 0;
        // non-null when an entry terminates exactly at this node
        Entry entry = nullptr;
        // children keyed by next token
        std::unordered_map<llama_token, std::unique_ptr<node>> children;
    };

public:
    // Index entry `e` along its full token sequence. `e` must remain stable and
    // its tokens unchanged while indexed.
    template <typename Seq>
    void insert(const Seq & toks, Entry e) {
        node * cur = root_.get();
        const int n = (int) toks.size();
        for (int i = 0; i < n; ++i) {
            const llama_token t = toks[i];
            auto & children = cur->children;
            auto it = children.find(t);
            if (it == children.end()) {
                it = children.emplace(t, std::make_unique<node>()).first;
            }
            cur = it->second.get();
            cur->n_entries++;
        }
        // terminal marker (entry may be empty-length; then it sits at the root)
        cur->entry = e;
        if (n == 0) {
            root_->n_entries++;
        }
    }

    // Remove entry `e` (refcount-decrement along its path, prune empty branches).
    template <typename Seq>
    void erase(const Seq & toks, Entry e) {
        const int n = (int) toks.size();

        // record the path so we can prune empty branches bottom-up
        std::vector<node *> path;
        path.reserve(n + 1);
        path.push_back(root_.get());

        node * cur = root_.get();
        for (int i = 0; i < n; ++i) {
            auto it = cur->children.find(toks[i]);
            if (it == cur->children.end()) {
                return; // not present
            }
            cur = it->second.get();
            path.push_back(cur);
        }

        if (cur->entry != e) {
            return; // not the indexed entry
        }
        cur->entry = nullptr;

        // decrement refcounts along the path (terminal first, then ancestors)
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            (*it)->n_entries--;
        }

        // prune childless, entry-less branches (except the root)
        for (size_t i = path.size() - 1; i >= 1; --i) {
            node * child = path[i];
            if (child->n_entries <= 0 && child->entry == nullptr && child->children.empty()) {
                path[i - 1]->children.erase(toks[i - 1]);
            } else {
                break;
            }
        }
    }

    // Drop everything.
    void clear() {
        root_ = std::make_unique<node>();
    }

    // Walk `toks`. Returns the longest prefix length shared with any cached
    // entry, and the best candidate entry sharing it (or nullptr). `qsize` is
    // the query length, used for the similarity tie-break.
    template <typename Seq>
    Entry longest_prefix(const Seq & toks, int qsize, int & lcp_len) const {
        const node * cur = root_.get();
        const int n = (int) toks.size();
        int depth = 0;
        for (int i = 0; i < n; ++i) {
            auto it = cur->children.find(toks[i]);
            if (it == cur->children.end()) {
                break;
            }
            cur = it->second.get();
            depth++;
        }
        lcp_len = depth;

        if (cur->n_entries <= 0) {
            return nullptr;
        }

        return best_in_subtree(cur, depth, qsize, kDfsBudget);
    }

private:
    Entry best_in_subtree(const node * n, int lcp_len, int qsize, int budget) const {
        if (budget <= 0) {
            return nullptr;
        }

        Entry best = nullptr;
        int best_size = INT32_MAX; // smaller entry => larger keep = lcp_len/size
        float best_sim = -1.0f;

        // iterative DFS
        std::vector<const node *> stack;
        stack.push_back(n);
        int visited = 0;

        while (!stack.empty()) {
            const node * cur = stack.back();
            stack.pop_back();

            if (visited++ > budget) {
                break;
            }

            if (cur->entry != nullptr) {
                const int esize = cur->entry->n_tokens();
                // legacy threshold: don't trash large prompts
                const float f_keep = esize > 0 ? float(lcp_len) / float(esize) : 1.0f;
                const float sim    = qsize > 0 ? float(lcp_len) / float(qsize) : 0.0f;
                if (f_keep >= 0.25f) {
                    // with lcp fixed, sim is constant => pick the smallest entry (max keep)
                    if (esize < best_size || (esize == best_size && sim > best_sim)) {
                        best = cur->entry;
                        best_size = esize;
                        best_sim = sim;
                    }
                }
            }

            for (auto & kv : cur->children) {
                stack.push_back(kv.second.get());
            }
        }

        return best;
    }

    // Number of trie nodes best_in_subtree is willing to visit before giving up.
    // Keeps worst-case lookup bounded; the legacy linear scan had no such bound.
    static constexpr int kDfsBudget = 256;

    std::unique_ptr<node> root_ = std::make_unique<node>();
};

// Production alias: entries are server_prompt handles.
using server_radix_index = server_radix_index_impl<server_prompt*>;
