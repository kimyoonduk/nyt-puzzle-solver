import numpy as np

DIRECTIONS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


"""
Opt to use adjacent pairs (valid sequence of length 2) instead.
Since finding all paths is not a significant bottleneck, a more complex check may result in erroneously omitting potentially valid paths.
"""


def get_valid_sequences(matrix, length):
    n = len(matrix)
    m = len(matrix[0])
    sequences = set()

    def dfs(x, y, current_sequence):
        if len(current_sequence) == length:
            sequences.add(tuple(current_sequence))
            return

        for dx, dy in DIRECTIONS_8:
            ni, nj = x + dx, y + dy
            if 0 <= ni < n and 0 <= nj < m:
                dfs(ni, nj, current_sequence + [matrix[ni][nj]])

    for i in range(n):
        for j in range(m):
            dfs(i, j, [matrix[i][j]])

    return sequences


def filter_words_by_valid_sequences(word_list, matrix, max_length=5):
    filtered_words = word_list.copy()
    for length in range(2, max_length + 1):  # Adjust the range as needed
        current_valid_sequences = get_valid_sequences(matrix, length)
        filtered_words = [
            word
            for word in filtered_words
            if not any(
                tuple(word[i : i + length]) not in current_valid_sequences
                for i in range(len(word) - length + 1)
            )
        ]
    return filtered_words


def divide_board_into_zones_orig(spangram_path, spangram_bm, n, m):
    zone1_mask = 0
    zone2_mask = 0

    # Determine if the spangram_path spans the x or y dimension
    spans_x = any(x == 0 for x, y in spangram_path) and any(
        x == n - 1 for x, y in spangram_path
    )
    spans_y = any(y == 0 for x, y in spangram_path) and any(
        y == m - 1 for x, y in spangram_path
    )

    if spans_x:
        # Spans the x dimension (top to bottom)
        for x, y in spangram_path:
            for i in range(n):
                min_y = min(y, m - 1 - y)
                for j in range(m):
                    if j < y:
                        zone1_mask |= 1 << (i * m + j)
                    elif j > y:
                        zone2_mask |= 1 << (i * m + j)
    elif spans_y:
        # Spans the y dimension (left to right)
        for x, y in spangram_path:
            for i in range(n):
                for j in range(m):
                    if i < x:
                        zone1_mask |= 1 << (i * m + j)
                    elif i > x:
                        zone2_mask |= 1 << (i * m + j)

    else:
        raise ValueError("Spangram does not span x or y dimension")

    # Remove spangram from zone1 and zone2
    zone1_mask &= ~spangram_bm
    zone2_mask &= ~spangram_bm

    return zone1_mask, zone2_mask


def get_valid_bm_idx_orig(valid, idx_list, valid_idx_list=None):

    # get AND of valid paths for all idx in idx_list
    if isinstance(idx_list, int):
        idx_list = [idx_list]

    # if filtering index is empty, all idx is valid
    if len(idx_list) == 0:
        valid_idx = [i for i in range(valid.shape[0])]
    else:
        valid_paths = valid[idx_list[0]]

        if len(idx_list) > 1:
            for idx in idx_list[1:]:
                valid_paths = np.logical_and(valid_paths, valid[idx])

        valid_idx = np.where(valid_paths)[0]

    # if valid_idx_list is provided, filter valid_idx with only those that are in valid_idx_list
    if valid_idx_list is not None:
        valid_idx = [idx for idx in valid_idx if idx in valid_idx_list]

    return valid_idx


# matrix for checking bitmasks that are adjacent in a cardinal direction
# does not exclude overlaps
def init_adjacent_matrix(k, combined_bitmasks, n, m):
    adjacent = np.zeros((k, k), dtype=bool)

    for i in range(k):
        bm_i = combined_bitmasks[i]

        # get a bitmask of adjacent nodes
        adj_bm = 0

        # loop on all 1 bits in bm_i
        for bit_idx in range(n * m):
            if bm_i & (1 << bit_idx):
                x, y = divmod(bit_idx, m)
                # only check cardinal directions
                for dx, dy in DIRECTIONS_4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m:
                        adj_bm |= 1 << (nx * m + ny)

        for j in range(i + 1, k):
            bm_j = combined_bitmasks[j]

            # if bm_j overlaps with adj_bm, then bm_i and bm_j are "adjacent"
            if bm_j & adj_bm and not bm_i & bm_j:
                adjacent[i, j] = True
                adjacent[j, i] = True

    return adjacent


def find_all_covering_paths_v3(all_paths, span_paths, matrix, solution_size):

    n, m = len(matrix), len(matrix[0])

    # Convert paths to bitmasks
    bitmask_list, bitmask_path_dict = get_bitmask_list(all_paths, n, m)
    spanmask_list, spanmask_path_dict = get_bitmask_list(span_paths, n, m)
    combined_bitmasks = spanmask_list + bitmask_list
    combined_path_dict = {**bitmask_path_dict, **spanmask_path_dict}

    num_spans = len(spanmask_list)
    k = len(combined_bitmasks)

    print(f"searching solution of size: {solution_size}")
    print(f"span bitmasks: {num_spans}")
    print(f"all bitmasks: {k}")

    start = time.time()
    print(f"initializing validity matrix")
    # initialize matrix
    valid = init_valid_matrix(combined_bitmasks, combined_path_dict, matrix)
    print(f"valid matrix initialized: {time.time() - start:.4f}s")

    start = time.time()
    print(f"initializing adjacent bitmask matrix")
    adjacent = init_adjacent_matrix(k, combined_bitmasks, n, m)
    print(f"adj matrix initialized: {time.time() - start:.4f}s")

    all_solutions = []
    all_solutions_by_zone = []

    for span_idx, span_bitmask in enumerate(spanmask_list):

        # if span_idx != 8:
        #     continue

        spangram_path = combined_path_dict[span_bitmask][0]
        span_word = path_to_word(spangram_path, matrix)

        # get list of valid bitmasks
        valid_idx_list = get_valid_bm_idx(valid, span_idx)
        span_start = time.time()
        print()
        print(f"[{span_idx}]{span_word}: {len(valid_idx_list)}")

        # divide board into two zones
        zonemask_list = divide_board_into_zones_bm(span_bitmask, n, m)

        if len(zonemask_list) == 0:
            continue

        zone_bitmasks = [[] for _ in range(len(zonemask_list))]

        # assign path bitmasks to zones
        for bm_idx in valid_idx_list:
            bitmask = combined_bitmasks[bm_idx]

            for zone_idx, zonemask in enumerate(zonemask_list):

                if bitmask & zonemask:
                    zone_bitmasks[zone_idx].append(bm_idx)
                    break

        # sort zonemask_list and zone_bitmasks by size of zone_bitmask in ascending order
        paired_zones = list(zip(zone_bitmasks, zonemask_list))

        # sort paired_zones by the size of zone_bitmasks
        paired_zones.sort(key=lambda pair: len(pair[0]))

        # unzip the sorted pairs
        zone_bitmasks, zonemask_list = zip(*paired_zones)

        # convert the zip objects back to lists
        zone_bitmasks = list(zone_bitmasks)
        zonemask_list = list(zonemask_list)

        zone_solutions = [set() for _ in range(len(zonemask_list))]
        span_has_solution = True

        available_zone_count = solution_size - 1  # subtract 1 for the spangram

        for zone_idx, path_bitmask_list in enumerate(zone_bitmasks):
            visited_subsets = set()
            explored_nodes = build_integer_trie([])

            def backtrack_zone(
                current_cover,
                bm_list,
                solution_set,
                solution_mask,
                valid_idx_list,
                available_zone_count,
            ):

                current_subset = frozenset(bm_list)
                # print(bm_list)

                if current_cover == solution_mask:

                    solution_set.add(current_subset)
                    return

                if len(bm_list) >= available_zone_count:
                    return

                # if current_subset in visited_subsets:
                #     return
                # visited_subsets.add(current_subset)
                if explored_nodes.search_subset(current_subset):
                    # print("ALREADY EXPLORED")
                    return

                # limit search to adjacent bitmasks, since solution is collectively exhaustive
                new_valid_idx_list, valid_adjacent = get_valid_bm_with_adjacency(
                    valid, adjacent, bm_list, valid_idx_list
                )

                # new_valid_idx_list = get_valid_bm_idx(valid, bm_list, valid_idx_list)

                for i in valid_adjacent:
                    if current_cover & combined_bitmasks[i] == 0:
                        bm_list.append(i)
                        backtrack_zone(
                            current_cover | combined_bitmasks[i],
                            bm_list,
                            solution_set,
                            solution_mask,
                            new_valid_idx_list,
                            available_zone_count,
                        )
                        explored_nodes.insert(frozenset(bm_list))
                        # print(f"INSERTING {bm_list}")
                        bm_list.pop()
                        # set all values in the i-th column of row in bm_list to False
                        # if len(bm_list) > 0:
                        #     valid[bm_list[-1], i] = False
                        #     valid[i, bm_list[-1]] = False

            print(f"Zone {span_idx}-{zone_idx}: {len(path_bitmask_list)} paths")

            zone_solution_set = zone_solutions[zone_idx]

            for i in path_bitmask_list:
                backtrack_zone(
                    combined_bitmasks[i],
                    [i],
                    zone_solution_set,
                    zonemask_list[zone_idx],
                    path_bitmask_list,
                    available_zone_count,
                )
                explored_nodes.insert(frozenset([i]))

            if len(zone_solution_set) == 0:
                print(f"Zone {span_idx}-{zone_idx} has no solution")

                span_has_solution = False
                break

            # update available zone count
            # get unique solutions
            zone_solution_list = list(zone_solution_set)
            # subtract size of smallest zone_solution

            min_zone_size = len(min(zone_solution_list, key=len))
            available_zone_count -= min_zone_size
            print(
                f"Zone {span_idx}-{zone_idx} solutions: {len(zone_solution_list)}, min size: {min_zone_size}"
            )

        print(f"span finished: {time.time() - span_start:.4f}s")

        if not span_has_solution:
            continue

        # combine zone solutions
        span_solutions = []
        solutions_by_zone = [zone_solutions[i] for i in range(len(zone_solutions))]

        running_solutions = [[] for _ in range(len(zone_solutions))]

        for zone_idx in range(len(zone_solutions)):
            zone_solution_list = list(zone_solutions[zone_idx])

            for sol in zone_solution_list:
                if zone_idx == 0:
                    # first put in span_idx
                    new_sol = [span_idx] + list(sol)
                    running_solutions[zone_idx].append(new_sol)
                else:
                    for prev_sol in running_solutions[zone_idx - 1]:
                        new_sol = prev_sol + list(sol)
                        running_solutions[zone_idx].append(new_sol)

        span_solutions = running_solutions[-1]

        if len(span_solutions) > 0:
            print(f"[{span_idx}]{span_word}: {len(span_solutions)} solutions found")
            for zone_idx in range(len(zone_solutions)):
                print(f"Zone {span_idx}-{zone_idx}: {len(zone_solutions[zone_idx])}")

            all_solutions.extend(span_solutions)

    solution_path_list = []

    for solution in all_solutions:
        solution_path = [combined_path_dict[combined_bitmasks[idx]] for idx in solution]
        solution_path_list.append(solution_path)

    return solution_path_list
