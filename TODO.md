# Strands

- [x] Filter by adjacent bitmasks (did not improve performance)
- [x] Update validity matrix for trapped nodes (implemented with span + bm_i + bm_j)
- [x] Track visited combinations with trie of visited subsets (no improvement)
- [x] Merge isolated zones
- [x] Sort bitmasks by size (doubtful but easy to try - no improvement)
- [ ] New implementation idea: Union Find or other spanning algorithm to join connected components? Connect each bitmask with edge weight BIG_NUMBER if overlap, adjacent bitmasks connected with edge weight 0.
- [ ] Convert into minimal set cover problem using embedding distance (involves semantics - no longer a "blind solve")
- [ ] Refactoring and readability improvements
