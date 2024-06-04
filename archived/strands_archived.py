import numpy as np

"""
Functions that are deprecated or not used in the final implementation

"""


# cardinal directions only
DIRECTIONS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# includes diagonals
DIRECTIONS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


"""
Opt to use adjacent pairs (valid sequence of length 2) instead.
Since finding all paths is not a significant bottleneck, a more complex check may result in erroneously omitting potentially valid paths.
"""


# get valid + adjacent bitmasks
def get_valid_bm_with_adjacency(valid, adjacent, idx_list, valid_idx_list=None):
    # Convert idx_list to a numpy array for efficient operations
    idx_list = np.atleast_1d(idx_list)

    # If filtering index is empty, all indices are valid
    if len(idx_list) == 0:
        valid_idx = np.arange(valid.shape[0])
    else:
        # Get the AND of valid paths for all indices in idx_list
        valid_paths = np.logical_and.reduce(valid[idx_list])
        valid_idx = np.where(valid_paths)[0]

    # If valid_idx_list is provided, filter valid_idx with only those that are in valid_idx_list
    if valid_idx_list is not None:
        valid_idx = np.intersect1d(valid_idx, valid_idx_list, assume_unique=True)

    # Get the OR of adjacent paths for all indices in idx_list
    adjacent_paths = np.logical_or.reduce(adjacent[idx_list])
    adjacent_idx = np.where(adjacent_paths)[0]

    # if 19 in idx_list and 593 in idx_list and 389 in idx_list:
    #     print("VALID")
    #     print(valid_idx)
    #     print("ADJACENT")
    #     print(adjacent_idx)
    #     print("VALID_IDX")
    #     print(valid_idx_list)

    # Get the AND of valid and adjacent paths
    valid_adjacent = np.intersect1d(valid_idx, adjacent_idx, assume_unique=True)

    return valid_idx, valid_adjacent


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


def find_all_covering_paths_zone_track(all_paths, span_paths, matrix, solution_size):

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
    print(f"matrix initialized: {time.time() - start:.4f}s")

    all_solutions = []
    all_solutions_by_zone = []

    for span_idx, span_bitmask in enumerate(spanmask_list):

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
                explored_nodes,
            ):

                current_subset = frozenset(bm_list)
                # print(bm_list)

                if current_cover == solution_mask:

                    solution_set.add(current_subset)
                    return

                if len(bm_list) >= available_zone_count:
                    return

                if explored_nodes.search_subset(current_subset):
                    # print("ALREADY EXPLORED")
                    return

                # one time check to prune large zones
                if available_zone_count >= 5 and len(bm_list) == 3:
                    has_trapped_zone = is_trapped(current_cover, n, m, min_zone_size=4)
                    if has_trapped_zone:
                        return

                new_valid_idx_list = get_valid_bm_idx(valid, bm_list, valid_idx_list)

                for i in new_valid_idx_list:
                    if current_cover & combined_bitmasks[i] == 0:
                        bm_list.append(i)
                        backtrack_zone(
                            current_cover | combined_bitmasks[i],
                            bm_list,
                            solution_set,
                            solution_mask,
                            new_valid_idx_list,
                            available_zone_count,
                            explored_nodes,
                        )
                        # print(f"INSERTING {bm_list}")
                        explored_nodes.insert(bm_list)
                        bm_list.pop()

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
                    explored_nodes,
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


def find_all_covering_paths_track(all_paths, span_paths, n, m):

    # Convert paths to bitmasks
    bitmask_list, bitmask_path_dict = get_bitmask_list(all_paths, n, m)
    spanmask_list, spanmask_path_dict = get_bitmask_list(span_paths, n, m)
    combined_bitmasks = spanmask_list + bitmask_list
    combined_path_dict = {**bitmask_path_dict, **spanmask_path_dict}

    print(f"span bitmasks: {len(spanmask_list)}")
    print(f"regular bitmasks: {len(bitmask_list)}")

    full_cover = (1 << (n * m)) - 1  # Bitmask with all nodes covered
    all_solutions = []

    explored_combination = set()
    num_spans = len(spanmask_list)

    if num_spans == 0:
        print("No spangrams found")
        return []

    def backtrack(current_cover, current_combination):
        # print(bitmask_to_paths(current_combination))

        if current_combination in explored_combination:
            return
        # Mark the combination as explored
        explored_combination.add(current_combination)

        if current_cover == full_cover:
            all_solutions.append(current_combination)

            # if len(all_solutions) % 10 == 0:
            # print(len(all_solutions))
            # return

        for i in range(len(combined_bitmasks)):
            next_combination = current_combination | (1 << i)

            if current_cover & combined_bitmasks[i] == 0:  # No overlap
                backtrack(current_cover | combined_bitmasks[i], next_combination)

    # Try each spangram path as the starting point
    for i, span_bitmask in enumerate(spanmask_list):
        # begin after spangrams
        backtrack(span_bitmask, 1 << i)

    solution_path_list = []

    for solution in all_solutions:
        mask_idx_list = bitmask_to_paths(solution)
        solution_path = [
            combined_path_dict[combined_bitmasks[i]] for i in mask_idx_list
        ]
        solution_path_list.append(solution_path)

    return solution_path_list


# SOLUTION 1:
def find_all_covering_paths(all_paths, span_paths, n, m):

    # Convert paths to bitmasks
    bitmask_list, bitmask_path_dict = get_bitmask_list(all_paths, n, m)
    spanmask_list, spanmask_path_dict = get_bitmask_list(span_paths, n, m)
    combined_bitmasks = spanmask_list + bitmask_list
    combined_path_dict = {**bitmask_path_dict, **spanmask_path_dict}
    num_spans = len(spanmask_list)

    print(f"span bitmasks: {len(spanmask_list)}")
    print(f"regular bitmasks: {len(bitmask_list)}")

    full_cover = (1 << (n * m)) - 1  # Bitmask with all nodes covered
    all_solutions = []

    def backtrack(current_cover, path_index, selected_paths):
        # print(selected_paths)

        if current_cover == full_cover:
            all_solutions.append(selected_paths[:])

            return

        for i in range(path_index, len(combined_bitmasks)):
            if current_cover & combined_bitmasks[i] == 0:  # No overlap
                selected_paths.append(i)
                backtrack(current_cover | combined_bitmasks[i], i + 1, selected_paths)
                selected_paths.pop()

    # Try each spangram path as the starting point
    for i, span_bitmask in enumerate(spanmask_list):
        backtrack(span_bitmask, 0, [i])

    solution_path_list = []

    for solution in all_solutions:
        solution_path = [combined_path_dict[combined_bitmasks[idx]] for idx in solution]
        solution_path_list.append(solution_path)

    return solution_path_list


def __main__():

    parent_dir = Path(__file__).parent

    resource_dir = "resources"
    word_file = "wordlist-v4.txt"
    input_file = "strands_input.json"

    word_path = Path(parent_dir, resource_dir, word_file)
    with open(word_path, "r") as f:
        word_list = f.read().splitlines()

    input_path = Path(parent_dir, resource_dir, input_file)
    with open(input_path, "r") as f:
        input_data = json.load(f)

    big_input = input_data["big"]
    solution_size = 9

    small_input = input_data["small"]
    big = True

    if big:
        matrix = big_input
    else:
        matrix = small_input
    n = len(matrix)
    m = len(matrix[0])

    all_paths, span_paths = get_all_words(matrix, word_list)
    all_words = [path_to_word(path, matrix) for path in all_paths]
    span_words = [path_to_word(path, matrix) for path in span_paths]
    print(f"Word paths found: {len(all_words)}")
    print(f"Span paths found: {len(span_words)}")

    # for i, span in enumerate(span_paths):
    #     print(f"Span {i}: {path_to_word(span, matrix)}")

    if not big:
        # start_time = time.time()
        # cover_list = find_all_covering_paths(all_paths, span_paths, n, m)
        # print(
        #     f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s"
        # )

        # start_time = time.time()
        # cover_list = find_all_covering_paths_track(all_paths, span_paths, n, m)
        # print(
        #     f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s"
        # )

        start_time = time.time()
        cover_list = find_all_covering_paths_zone(
            all_paths, span_paths, matrix, solution_size
        )
        print(
            f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s"
        )

    # start_time = time.time()
    # cover_list = find_all_covering_paths_zone_track(
    #     all_paths, span_paths, matrix, solution_size
    # )
    # print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

    # start_time = time.time()
    # cover_list = find_all_covering_paths_v3(
    #     all_paths, span_paths, matrix, solution_size
    # )
    # print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

    start_time = time.time()
    cover_list = find_all_covering_paths_v4(
        all_paths, span_paths, matrix, solution_size
    )
    print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

    for i, word in enumerate(all_words):
        if word == "TIME":
            print(all_paths[i])

    for cover in cover_list:
        print(cover_to_word(cover, matrix))

    """
    Word List Length: 198422
    Optimized Word List Length: 602
    building trie: 602 words
    trie built: 0.0005s
    found 417 words and 87 spangrams: 0.0020s
    found 298 covering paths: 0.5721s
    """

    """
    # before cross check
    No tracking
    Word List Length: 207769
    Optimized Word List Length: 309
    span bitmasks: 89
    regular bitmasks: 250
    found 58 covering paths: 0.2323s

    # after cross check (revert to using adjacency pairs since performance difference is negligible)
    Word List Length: 207769
    Optimized Word List Length: 654
    span bitmasks: 86
    regular bitmasks: 233
    found 41 covering paths: 0.2011s
    """

    """
    # before cross check
    With tracking
    found 25 covering paths: 1.3840s

    # after cross check
    found 15 covering paths: 1.1298s    
    > all versions had correct answer
    """

    len(solutions)  # 298 for 5x4

    real_solution = [
        [(2, 0), (2, 1), (2, 2), (1, 2), (1, 3)],  # FRUIT
        [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)],  # BANANA
        [(4, 1), (3, 1), (4, 0), (3, 0)],  # LIME
        [(4, 2), (4, 3), (3, 3), (2, 3), (3, 2)],  # APPLE
    ]

    # 8x6
    solutions = get_all_words(big_input, word_list)
    """
    Word List Length: 198422
    Optimized Word List Length: 11433
    building trie: 11433 words
    trie built: 0.0109s
    found 1811 words and 16 spangrams: 0.0191s
    Timeout
    """


def find_all_covering_paths_v4(all_paths, span_paths, matrix, solution_size):

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
    print(f"matrix initialized: {time.time() - start:.4f}s")

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

        visited_subsets = set()

        available_zone_count = solution_size - 1  # subtract 1 for the spangram

        for zone_idx, path_bitmask_list in enumerate(zone_bitmasks):
            print(f"Zone {span_idx}-{zone_idx}: {len(path_bitmask_list)} paths")

            def backtrack_zone(
                current_cover,
                current_path,
                solution_set,
                solution_mask,
                valid_idx_list,
                available_zone_count,
                valid_matrix,
                bitmask_list,
            ):

                current_subset = frozenset(current_path)

                if current_cover == solution_mask:
                    # print("SOLUTION FOUND")

                    solution_set.add(current_subset)
                    return

                if len(current_path) >= available_zone_count:
                    # print(f"Exceeded zone count: {len(current_path)} / {available_zone_count}")
                    visited_subsets.add(current_subset)
                    return

                if current_subset in visited_subsets:
                    return
                visited_subsets.add(current_subset)

                new_valid_idx_list = get_valid_bm_idx(
                    valid_matrix, current_path, valid_idx_list
                )
                for i in new_valid_idx_list:
                    if current_cover & bitmask_list[i] == 0:
                        current_path.append(i)
                        backtrack_zone(
                            current_cover | bitmask_list[i],
                            current_path,
                            solution_set,
                            solution_mask,
                            new_valid_idx_list,
                            available_zone_count,
                            valid_matrix,
                            bitmask_list,
                        )
                        current_path.pop()

            # if too many paths in zone, resize matrix and filter further
            if len(path_bitmask_list) > 300:

                # create new valid matrix, combined_bitmasks, and combined_path_dict based on valid_idx_list
                zone_valid = valid[path_bitmask_list][:, path_bitmask_list]

                print(f"resizing validity matrix: {zone_valid.shape}")

                # combined_bitmasks are reindexed based on zone_valid
                zone_valid_bitmasks = [combined_bitmasks[i] for i in path_bitmask_list]
                new_bm_index = range(len(path_bitmask_list))

                # update valid span
                zone_valid = update_valid_matrix(
                    zone_valid, span_bitmask, zone_valid_bitmasks, n, m
                )

                # maintain a dict from new index to original index
                resized_to_orig = {i: path_bitmask_list[i] for i in new_bm_index}

                zone_solution_resized = set()

                for i in new_bm_index:
                    backtrack_zone(
                        zone_valid_bitmasks[i],
                        [i],
                        zone_solution_resized,
                        zonemask_list[zone_idx],
                        new_bm_index,
                        available_zone_count,
                        zone_valid,
                        zone_valid_bitmasks,
                    )

                # convert zone_solution_resized (set of frozensets) to original indices
                zone_solution_resized = {
                    frozenset(resized_to_orig[i] for i in sol)
                    for sol in zone_solution_resized
                }

                zone_solutions[zone_idx] = zone_solution_resized

            else:
                for i in path_bitmask_list:
                    backtrack_zone(
                        combined_bitmasks[i],
                        [i],
                        zone_solutions[zone_idx],
                        zonemask_list[zone_idx],
                        path_bitmask_list,
                        available_zone_count,
                        valid,
                        combined_bitmasks,
                    )

            # if zone does not have solution, skip this spangram
            if len(zone_solutions[zone_idx]) == 0:
                print(f"Zone {span_idx}-{zone_idx} has no solution")

                span_has_solution = False
                break

            # update available zone count
            # get unique solutions
            zone_solution_list = list(zone_solutions[zone_idx])
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


def identify_clusters(coords):
    visited = set()
    clusters = []

    def dfs(x, y, cluster):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) not in visited:
                visited.add((cx, cy))
                cluster.append((cx, cy))
                for dx, dy in DIRECTIONS_4:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in coords and (nx, ny) not in visited:
                        stack.append((nx, ny))

    for x, y in coords:
        if (x, y) not in visited:
            cluster = []
            dfs(x, y, cluster)
            clusters.append(cluster)

    return clusters


# we normally expect a spangram to produce 2 zones, but some may produce 3 or more
def divide_board_into_zones(span_bitmask, n, m, min_zone_size=4):

    full_cover = (1 << (n * m)) - 1  # Bitmask with all nodes covered
    available = full_cover & ~span_bitmask
    avail_coords = bitmask_to_coordinates(available, n, m)

    clusters = identify_clusters(avail_coords)

    if len(clusters) == 0:
        raise ValueError("No available zone clusters")

    # check if any clusters are too small
    for cluster in clusters:
        if len(cluster) < min_zone_size:
            # print(f"Cluster too small: {len(cluster)}. Returning empty zones")
            return []

    # sort clusters by size in ascending order
    clusters.sort(key=len)

    zonemask_list, _ = get_bitmask_list(clusters, n, m)

    return zonemask_list


def bitmask_to_paths(bitmask):
    paths = []
    i = 0
    while bitmask > 0:
        if bitmask & 1:
            paths.append(i)
        bitmask >>= 1
        i += 1
    return paths


def find_all_covering_paths_zone(all_paths, span_paths, matrix, solution_size):

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
    print(f"matrix initialized: {time.time() - start:.4f}s")

    # print indices of valid bitmasks for bitmask 394
    # print("394")  # SERIAL / no 258
    # print(get_valid_bm_idx(valid, 394))

    # print("599")  # CHILLY
    # print(get_valid_bm_idx(valid, 599))

    # print("258")  # STAKE
    # print(get_valid_bm_idx(valid, 258))

    # TIME 1474
    # print(get_valid_bm_idx(valid, 1474))

    """
    MOOSE (1353), 
    """

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

        # if span_idx == 81:
        #     # print(f"Span {span_idx}")
        #     # bitmask_to_matrix(span_bitmask, n, m)
        #     # for i, zonemask in enumerate(zonemask_list):
        #     #     if i == 1:
        #     #         print(f"Zone {i}")
        #     #         bitmask_to_matrix(zonemask, n, m)

        if len(zonemask_list) == 0:
            continue

        zone_bitmasks = [[] for _ in range(len(zonemask_list))]

        # assign path bitmasks to zones
        for bm_idx in valid_idx_list:
            bitmask = combined_bitmasks[bm_idx]

            # if bm_idx == 1474:
            #     print(f"1474: {bitmask}")
            #     bitmask_to_matrix(bitmask, n, m)

            for zone_idx, zonemask in enumerate(zonemask_list):
                # if bm_idx == 1474:
                #     print(f"Zone {zone_idx}: {zonemask}")
                #     bitmask_to_matrix(zonemask, n, m)

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

        visited_subsets = set()

        def backtrack_zone(
            current_cover,
            bm_list,
            solution_set,
            solution_mask,
            valid_idx_list,
            available_zone_count,
        ):

            current_subset = frozenset(bm_list)

            if current_cover == solution_mask:
                # print("SOLUTION FOUND")

                # print for first solution, then for every 1000 solutions
                # if len(solution_set) == 0 or len(solution_set) % 1000 == 0:
                # if 1474 in bm_list:
                #     print(f"Solution found: {len(solution_set)+1}")

                #     print(f"Span {span_idx}: {span_word}")
                #     bitmask_to_matrix(span_bitmask, n, m)

                #     for i in bm_list:
                #         print(f"Path {i}")
                #         bitmask_to_matrix(combined_bitmasks[i], n, m)
                solution_set.add(current_subset)
                return

            if len(bm_list) >= available_zone_count:
                visited_subsets.add(current_subset)
                # print(f"Exceeded zone count: {len(bm_list)} / {available_zone_count}")
                return

            if current_subset in visited_subsets:
                return
            visited_subsets.add(current_subset)

            # # one time check to prune large zones
            # if available_zone_count >= 5 and len(bm_list) == 3:
            #     has_trapped_zone = is_trapped(current_cover, n, m, min_zone_size=4)
            #     if has_trapped_zone:
            #         return

            # if 1474 in bm_list:
            #     print(f"Current subset: {bm_list}")

            new_valid_idx_list = get_valid_bm_idx(valid, bm_list, valid_idx_list)
            for i in new_valid_idx_list:
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
                    bm_list.pop()

        available_zone_count = solution_size - 1  # subtract 1 for the spangram

        for zone_idx, path_bitmask_list in enumerate(zone_bitmasks):
            print(f"Zone {span_idx}-{zone_idx}: {len(path_bitmask_list)} paths")

            # # HOMOPHONES (8)
            # if span_idx == 8 and zone_idx == 1:

            #     # ZONE 0: CHILLY (599), SERIAL (394), STAKE (258)
            #     bitmask_to_matrix(zonemask_list[zone_idx], n, m)
            #     for i in path_bitmask_list:
            #         print(f"Path {i}")
            #         print(
            #             path_to_word(
            #                 combined_path_dict[combined_bitmasks[i]][0], matrix
            #             )
            #         )
            #         bitmask_to_matrix(combined_bitmasks[i], n, m)

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

            # backtrack_zone(
            #     0,
            #     [],
            #     zone_solution_set,
            #     zonemask_list[zone_idx],
            #     path_bitmask_list,
            #     available_zone_count,
            # )
            # if zone does not have solution, skip this spangram
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


# v6 - sorting bitmask, hybrid tracking
# no performance improvements
def find_all_covering_paths_v6(
    all_paths, span_paths, matrix, solution_size=None, timeout=300
):

    n, m = len(matrix), len(matrix[0])

    # Convert paths to bitmasks
    bitmask_list, bitmask_path_dict = get_bitmask_list_sorted(all_paths, n, m)
    spanmask_list, spanmask_path_dict = get_bitmask_list(span_paths, n, m)
    print(bitmask_list[:10])
    combined_bitmasks = spanmask_list + bitmask_list
    combined_path_dict = {**bitmask_path_dict, **spanmask_path_dict}

    num_spans = len(spanmask_list)
    k = len(combined_bitmasks)

    if solution_size:
        filter_solutions_by_size = True
        print(f"searching solution of size: {solution_size}")
    else:
        filter_solutions_by_size = False
        solution_size = BIG_NUMBER
    print(f"span bitmasks: {num_spans}")
    print(f"all bitmasks: {k}")

    start = time.time()
    game_deadline = start + timeout * 2
    game_timed_out = False
    print(f"initializing validity matrix")
    # initialize matrix
    valid = init_valid_matrix(combined_bitmasks, combined_path_dict, matrix)
    print(f"matrix initialized: {time.time() - start:.4f}s")

    all_candidates = []
    all_solutions = []

    zones_by_span = []

    timed_out_span = -1
    time_to_first_solution = -1

    for span_idx, span_bitmask in enumerate(spanmask_list):

        spangram_path = combined_path_dict[span_bitmask][0]
        span_word = path_to_word(spangram_path, matrix)

        # get list of valid bitmasks
        valid_idx_list = get_valid_bm_idx(valid, span_idx)

        # divide board into two or more zones
        # zonemask_list = divide_board_into_zones_bm(span_bitmask, n, m)
        zonemask_list = divide_board_into_zones_with_merge(spangram_path, n, m)

        print(f"Span {span_idx}-{span_word}: {len(zonemask_list)} zones")

        if len(zonemask_list) == 0:
            print(f"No valid zones for span: {span_idx}-{span_word}")
            continue

        zone_bitmasks = [[] for _ in range(len(zonemask_list))]

        # assign path bitmasks to zones
        for bm_idx in valid_idx_list:
            bitmask = combined_bitmasks[bm_idx]

            for zone_idx, zonemask in enumerate(zonemask_list):

                if bitmask & zonemask:
                    zone_bitmasks[zone_idx].append(bm_idx)
                    break

        zones_by_span.append((span_idx, zone_bitmasks, zonemask_list))

    # sort spanmask_list by ascending order of max(zone sizes)
    sorted_spanmask_list = sorted(
        zones_by_span, key=lambda pair: max(len(zone) for zone in pair[1])
    )
    print(f"Searching {len(sorted_spanmask_list)} spans, sorted by max zone size")

    for span_idx, zone_bitmasks, zonemask_list in sorted_spanmask_list:

        if game_timed_out:
            print(f"Game timed out. Stopping search regardless of solutions.")
            break

        if (timed_out_span >= 0) and len(all_solutions) > 0:
            print(
                f"Span {timed_out_span} timed out. Stopping search and returning available solutions."
            )
            break

        span_start = time.time()
        span_deadline = span_start + timeout

        span_bitmask = combined_bitmasks[span_idx]
        spangram_path = combined_path_dict[span_bitmask][0]
        span_word = path_to_word(spangram_path, matrix)
        print(f"\n[{span_idx}]{span_word}: {len(zone_bitmasks)} zones")

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

        visited_subsets = set()
        explored_nodes = build_integer_trie([])

        available_zone_count = solution_size - 1  # subtract 1 for the spangram

        for zone_idx, path_bitmask_list in enumerate(zone_bitmasks):
            print(f"Zone {span_idx}-{zone_idx}: {len(path_bitmask_list)} paths")

            def backtrack_zone(
                current_cover,
                current_path,
                solution_set,
                solution_mask,
                valid_idx_list,
                valid_matrix,
                bitmask_list,
            ):

                current_subset = frozenset(current_path)

                if current_cover == solution_mask:

                    solution_set.add(current_subset)

                    return

                if len(current_path) >= available_zone_count:
                    # print(f"Exceeded zone count: {len(current_path)} / {available_zone_count}")
                    visited_subsets.add(current_subset)
                    return

                if time.time() > span_deadline:
                    visited_subsets.add(current_subset)
                    return

                if len(current_path) < 4:
                    if current_subset in visited_subsets:
                        return
                    visited_subsets.add(current_subset)
                else:
                    if explored_nodes.search_subset(current_subset):
                        return

                new_valid_idx_list = get_valid_bm_idx(
                    valid_matrix, current_path, valid_idx_list
                )
                for i in new_valid_idx_list:
                    if current_cover & bitmask_list[i] == 0:
                        current_path.append(i)
                        backtrack_zone(
                            current_cover | bitmask_list[i],
                            current_path,
                            solution_set,
                            solution_mask,
                            new_valid_idx_list,
                            valid_matrix,
                            bitmask_list,
                        )
                        explored_nodes.insert(frozenset(current_path))
                        current_path.pop()

            # if too many paths in zone, resize matrix and filter further
            if len(path_bitmask_list) > 300:

                # create new valid matrix, combined_bitmasks, and combined_path_dict based on valid_idx_list
                zone_valid = valid[path_bitmask_list][:, path_bitmask_list]

                print(f"resizing validity matrix: {zone_valid.shape}")

                # combined_bitmasks are reindexed based on zone_valid
                zone_valid_bitmasks = [combined_bitmasks[i] for i in path_bitmask_list]
                new_bm_index = range(len(path_bitmask_list))

                # update valid span
                zone_valid = update_valid_matrix(
                    zone_valid, span_bitmask, zone_valid_bitmasks, n, m
                )

                # maintain a dict from new index to original index
                resized_to_orig = {i: path_bitmask_list[i] for i in new_bm_index}

                zone_solution_resized = set()

                for i in new_bm_index:
                    backtrack_zone(
                        zone_valid_bitmasks[i],
                        [i],
                        zone_solution_resized,
                        zonemask_list[zone_idx],
                        new_bm_index,
                        zone_valid,
                        zone_valid_bitmasks,
                    )

                # convert zone_solution_resized (set of frozensets) to original indices
                zone_solution_resized = {
                    frozenset(resized_to_orig[i] for i in sol)
                    for sol in zone_solution_resized
                }

                zone_solutions[zone_idx] = zone_solution_resized

            else:
                for i in path_bitmask_list:
                    backtrack_zone(
                        combined_bitmasks[i],
                        [i],
                        zone_solutions[zone_idx],
                        zonemask_list[zone_idx],
                        path_bitmask_list,
                        valid,
                        combined_bitmasks,
                    )
                    explored_nodes.insert(frozenset([i]))

            # if zone timed out, skip this spangram
            if time.time() > span_deadline:
                print(f"Zone {span_idx}-{zone_idx} timed out. Skipping spangram.")
                timed_out_span = span_idx
                span_has_solution = False
                break

            # if zone does not have solution, skip this spangram
            if len(zone_solutions[zone_idx]) == 0:
                print(f"Zone {span_idx}-{zone_idx} has no solution. Skipping spangram.")

                span_has_solution = False
                break

            # update available zone count
            # get unique solutions
            zone_solution_list = list(zone_solutions[zone_idx])
            # subtract size of smallest zone_solution

            min_zone_size = len(min(zone_solution_list, key=len))
            available_zone_count -= min_zone_size
            print(
                f"Zone {span_idx}-{zone_idx}: {len(zone_solution_list)} solutions with min size {min_zone_size}."
            )

        print(f"span finished: {time.time() - span_start:.4f}s")

        if not span_has_solution:
            continue

        # combine zone solutions
        # TODO: refactor with list(itertools.product(*solution))
        span_solutions = []

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
            if (len(all_solutions) == 0) and (time_to_first_solution < 0):
                time_to_first_solution = time.time() - start
                print(f"First solution found in {time_to_first_solution:.4f}s")
            for zone_idx in range(len(zone_solutions)):
                print(f"Zone {span_idx}-{zone_idx}: {len(zone_solutions[zone_idx])}")

            if filter_solutions_by_size:
                # filter solutions by solution size
                filtered_solutions = [
                    sol for sol in span_solutions if len(sol) == solution_size
                ]
                print(
                    f"[{span_idx}]{span_word}: {len(span_solutions)} candidate solutions found. {len(filtered_solutions)} solutions of size {solution_size}"
                )
            else:
                filtered_solutions = span_solutions
                print(
                    f"[{span_idx}]{span_word}: {len(span_solutions)} candidate solutions found."
                )

            all_candidates.extend(span_solutions)
            all_solutions.extend(filtered_solutions)

        # update game timeout flag
        game_timed_out = time.time() > game_deadline

    solution_path_list = []
    candidate_path_list = []

    for solution in all_solutions:
        solution_path = [combined_path_dict[combined_bitmasks[idx]] for idx in solution]
        solution_path_list.append(solution_path)

    for candidate in all_candidates:
        candidate_path = [
            combined_path_dict[combined_bitmasks[idx]] for idx in candidate
        ]
        candidate_path_list.append(candidate_path)

    timed_out = (
        True if (timed_out_span >= 0 and len(solution_path_list) == 0) else False
    )

    elapsed_time = time.time() - start

    solution_object = {
        "solution_path_list": solution_path_list,
        "candidate_path_list": candidate_path_list,
        "timed_out": timed_out,
        "elapsed_time": elapsed_time,
        "time_to_first_solution": time_to_first_solution,
    }

    return solution_object
