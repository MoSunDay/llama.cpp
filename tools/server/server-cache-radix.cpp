#include "server-cache-radix.h"

#include "server-task.h"

#include <climits>
#include <vector>

// Number of trie nodes best_in_subtree is willing to visit before giving up.
// Keeps worst-case lookup bounded; the legacy linear scan had no such bound.
static constexpr int kDfsBudget = 256;

void server_radix_index::insert(const server_tokens & toks, server_prompt * e) {
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

void server_radix_index::erase(const server_tokens & toks, server_prompt * e) {
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

void server_radix_index::clear() {
    root_ = std::make_unique<node>();
}

server_prompt * server_radix_index::longest_prefix(const server_tokens & toks, int qsize, int & lcp_len) const {
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

server_prompt * server_radix_index::best_in_subtree(const node * n, int lcp_len, int qsize, int budget) const {
    if (budget <= 0) {
        return nullptr;
    }

    server_prompt * best = nullptr;
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
