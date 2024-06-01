import time
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from util.trie import build_trie
from util.integer_trie import build_integer_trie

DIRECTIONS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# includes diagonals
DIRECTIONS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def filter_words_by_char_count(matrix, word_list):
    char_count = Counter()
    for row in matrix:
        char_count.update(row)

    filtered_words = []
    for word in word_list:
        word_count = Counter(word)
        if all(char_count[char] >= word_count[char] for char in word_count):
            filtered_words.append(word)

    return filtered_words


def get_adjacent_pairs(matrix):
    n = len(matrix)
    m = len(matrix[0])
    pairs = set()

    for i in range(n):
        for j in range(m):
            for dx, dy in DIRECTIONS_8:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < m:
                    pairs.add((matrix[i][j], matrix[ni][nj]))
                    pairs.add(
                        (matrix[ni][nj], matrix[i][j])
                    )  # Add reverse pair as well

    return pairs


def filter_words_by_adjacent_pairs(word_list, valid_pairs):
    def has_invalid_pairs(word):
        for i in range(len(word) - 1):
            if (word[i], word[i + 1]) not in valid_pairs:
                return True
        return False

    return [word for word in word_list if not has_invalid_pairs(word)]


def optimize_word_list(matrix, word_list):

    print(f"Word List Length: {len(word_list)}")

    # convert all words to uppercase
    word_list = [word.upper() for word in word_list]

    # words that are 4 or more characters long
    word_list = [word for word in word_list if len(word) >= 4]

    # Filter words based on character counts
    word_list = filter_words_by_char_count(matrix, word_list)

    # Filter words based on adjacent character pairs
    valid_pairs = get_adjacent_pairs(matrix)
    word_list = filter_words_by_adjacent_pairs(word_list, valid_pairs)

    # word_list = filter_words_by_valid_sequences(word_list, matrix)

    print(f"Optimized Word List Length: {len(word_list)}")

    return word_list


def is_spangram(path, n, m):
    touches_top = any(x == 0 for x, y in path)
    touches_bottom = any(x == n - 1 for x, y in path)
    touches_left = any(y == 0 for x, y in path)
    touches_right = any(y == m - 1 for x, y in path)
    return (touches_top and touches_bottom) or (touches_left and touches_right)


def path_to_bitmask(path, n, m):
    bitmask = 0
    for x, y in path:
        bitmask |= 1 << (x * m + y)
    return bitmask


def get_bitmask_list(path_list, n, m):

    bitmask_path_dict = defaultdict(list)
    bitmasks = set()

    for path in path_list:

        bitmask = path_to_bitmask(path, n, m)
        bitmasks.add(bitmask)
        bitmask_path_dict[bitmask].append(path)

        # if path == [(6, 1), (6, 0), (7, 0), (7, 1)]:
        #     print("FOUND PATH")
        #     print(bitmask)
        #     found = bitmask

    bitmask_list = list(bitmasks)

    # 13400297963520
    # find indexin bitmask_list
    # print(bitmask_list)
    # print(bitmask_list.index(found))

    return bitmask_list, bitmask_path_dict


# can be defined as two lines having one solution
# TODO: optimize using matrix operations to parallelize computation
def check_crossing_seg(seg1, seg2):
    (ax1, ay1), (ax2, ay2) = seg1
    (bx1, by1), (bx2, by2) = seg2

    # Coefficients of linear equations
    A1 = ax2 - ax1
    A2 = ay2 - ay1
    B1 = bx1 - bx2
    B2 = by1 - by2
    C1 = bx1 - ax1
    C2 = by1 - ay1

    # Determinant
    D = A1 * B2 - A2 * B1

    if D == 0:
        return False  # Parallel or collinear segments

    # Calculate parameters t and s using Cramer's rule
    t = (C1 * B2 - C2 * B1) / D
    s = (A1 * C2 - A2 * C1) / D

    # print(t, s)

    return 0 < t < 1 and 0 < s < 1  # Intersection inside both segments


def check_crossing_path(path1, path2):
    for i in range(len(path1) - 1):
        for j in range(len(path2) - 1):
            if check_crossing_seg((path1[i], path1[i + 1]), (path2[j], path2[j + 1])):
                return True

    return False


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


# matrix for checking valid path pairings
def init_valid_matrix(combined_bitmasks, combined_path_dict, board, min_zone_size=4):

    k = len(combined_bitmasks)

    valid = np.ones((k, k), dtype=bool)
    np.fill_diagonal(valid, False)

    # check for overlaps and crossing
    # triangluar matrix to reduce computation
    for i in range(k):
        bm_i = combined_bitmasks[i]
        paths_i = combined_path_dict[combined_bitmasks[i]]
        words_i = set(path_to_word(path, board) for path in paths_i)

        for j in range(i + 1, k):
            bm_j = combined_bitmasks[j]
            paths_j = combined_path_dict[bm_j]
            words_j = set(path_to_word(path, board) for path in paths_j)

            # overlap
            if bm_i & bm_j:
                valid[i, j] = False
                valid[j, i] = False

            # crossing
            # any path is okay
            if check_crossing_path(
                paths_i[0],
                paths_j[0],
            ):
                valid[i, j] = False
                valid[j, i] = False

            # if the sets words_i and words_j are exactly the same, then invalid
            if words_i == words_j:
                valid[i, j] = False
                valid[j, i] = False

            # # check for size of "trapped" nodes
            # # from full_cover, remove bm_i and bm_j
            # available = full_cover & ~bm_i & ~bm_j

            # # run dfs on available. if any cluster is less than min_zone_size, invalid
            # clusters = dfs_bitmask(available, n, m, min_zone_size)

            # if i == 258 and j == 394:
            #     print(f"258: {clusters}")
            # # len(clusters) is 0 if any cluster is invalid
            # if len(clusters) == 0:
            #     valid[i, j] = False
            #     valid[j, i] = False

    # bitmask_to_matrix(combined_bitmasks[389], n, m)
    # print()
    # bitmask_to_matrix(combined_bitmasks[19], n, m)
    # print()
    # bitmask_to_matrix(combined_bitmasks[567], n, m)
    # print()
    # bitmask_to_matrix(combined_bitmasks[593], n, m)

    return valid


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


# updated with numpy operations for efficiency
def get_valid_bm_idx(valid, idx_list, valid_idx_list=None):
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

    return valid_idx


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


# for debugging
def bitmask_to_coordinates(bitmask, n, m):
    coordinates = []
    for i in range(n):
        for j in range(m):
            if bitmask & (1 << (i * m + j)):
                coordinates.append((i, j))
    return coordinates


# print coordinates as 2d n x m matrix marked with 1
def bitmask_to_matrix(bitmask, n, m):
    matrix = [["." for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if bitmask & (1 << (i * m + j)):
                matrix[i][j] = "X"

    for row in matrix:
        for elem in row:
            print(f"{elem:4}", end=" ")
        print()


# count number of 1s in bitmask
def count_set_bits(x):
    count = 0
    while x:
        x &= x - 1  # Remove the rightmost set bit
        count += 1
    return count


# bitmask version of dfs
def dfs_bitmask(input_bm, n, m, min_zone_size=4):
    # bitmask to track visited nodes
    visited = 0
    clusters = []

    def dfs(start, visited):
        stack = [start]
        # initialize bitmask for cluster
        cluster = 0
        while stack:
            pos = stack.pop()
            if not (visited & (1 << pos)):
                visited |= 1 << pos
                cluster |= 1 << pos

                # get "coordinates"
                x, y = divmod(pos, m)
                for dx, dy in DIRECTIONS_4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m:
                        npos = nx * m + ny
                        if (input_bm & (1 << npos)) and not (visited & (1 << npos)):
                            stack.append(npos)
        return cluster, visited

    for i in range(n * m):
        # if node is part of input_bm and not visited
        if (input_bm & (1 << i)) and not (visited & (1 << i)):
            cluster, visited = dfs(i, visited)
            if count_set_bits(cluster) >= min_zone_size:
                clusters.append(cluster)
            else:
                # print(f"Cluster too small: {count_set_bits(cluster)}")
                return []

    return clusters


def divide_board_into_zones_bm(span_bitmask, n, m, min_zone_size=4):
    full_cover = (1 << (n * m)) - 1  # Bitmask with all nodes covered
    available = full_cover & ~span_bitmask

    clusters = dfs_bitmask(available, n, m)

    if len(clusters) == 0:
        print("No available zone clusters")
        return []

    # check if any clusters are too small
    for cluster in clusters:
        if count_set_bits(cluster) < min_zone_size:
            print(
                f"Cluster too small: {count_set_bits(cluster)}. Returning empty zones"
            )
            return []

    return clusters


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
                # print(f"Exceeded zone count: {len(bm_list)} / {available_zone_count}")
                return

            if current_subset in visited_subsets:
                return
            visited_subsets.add(current_subset)

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


def bitmask_to_paths(bitmask):
    paths = []
    i = 0
    while bitmask > 0:
        if bitmask & 1:
            paths.append(i)
        bitmask >>= 1
        i += 1
    return paths


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


def path_to_word(path, matrix):
    return "".join(matrix[x][y] for x, y in path)


# matrix: List[List[str]] of size n x m
# word_list: List[str]
def get_all_words(matrix, word_list, verbose=False):

    word_list = optimize_word_list(matrix, word_list)

    n = len(matrix)
    m = len(matrix[0])

    longest_len = len(max(word_list, key=len))

    start_time = time.time()
    if verbose:
        print(f"building trie: {len(word_list)} words")
    trie = build_trie(word_list)
    if verbose:
        print(f"trie built: {time.time() - start_time:.4f}s")
    all_paths = []
    span_paths = []
    visited = [[False for _ in range(m)] for _ in range(n)]

    def dfs(x, y, path, current_word, current_node):
        if len(current_word) >= 4 and current_node.is_end_of_word:
            if is_spangram(path, n, m):
                span_paths.append(path)
            else:
                all_paths.append(path)

        if len(current_word) > longest_len:
            return

        for dx, dy in DIRECTIONS_8:
            nx, ny = x + dx, y + dy

            crossing = check_crossing_path(path, [(x, y), (nx, ny)])

            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny] and not crossing:
                next_char = matrix[nx][ny]
                if next_char in current_node.children:
                    visited[nx][ny] = True
                    dfs(
                        nx,
                        ny,
                        path + [(nx, ny)],
                        current_word + next_char,
                        current_node.children[next_char],
                    )
                    visited[nx][ny] = False

    for i in range(n):
        for j in range(m):
            start_char = matrix[i][j]
            if start_char in trie.root.children:
                visited[i][j] = True
                dfs(i, j, [(i, j)], start_char, trie.root.children[start_char])
                visited[i][j] = False

    if verbose:
        print(
            f"found {len(all_paths)} words and {len(span_paths)} spangrams: {time.time() - start_time:.4f}s"
        )

    return all_paths, span_paths


def cover_to_word(cover, matrix):
    return [[path_to_word(path, matrix) for path in path_list] for path_list in cover]


def test():

    print(check_crossing_seg(((0, 0), (1, 1)), ((0, 1), (1, 0))))  # Should return True
    print(check_crossing_seg(((2, 3), (3, 4)), ((3, 3), (2, 4))))  # Should return True
    print(check_crossing_seg(((0, 0), (1, 1)), ((3, 4), (3, 5))))  # Should return False


def __main__():

    parent_dir = Path(__file__).parent

    resource_dir = "resources"
    word_file = "wordlist-v4.txt"
    input_file = "strands_input.json"

    word_path = Path(parent_dir, resource_dir, word_file)
    input_path = Path(parent_dir, resource_dir, input_file)

    with open(word_path, "r") as f:
        word_list = f.read().splitlines()

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
        start_time = time.time()
        cover_list = find_all_covering_paths(all_paths, span_paths, n, m)
        print(
            f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s"
        )

        start_time = time.time()
        cover_list = find_all_covering_paths_track(all_paths, span_paths, n, m)
        print(
            f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s"
        )

        start_time = time.time()
        cover_list = find_all_covering_paths_zone(
            all_paths, span_paths, matrix, solution_size
        )
        print(
            f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s"
        )

    start_time = time.time()
    cover_list = find_all_covering_paths_zone_track(
        all_paths, span_paths, matrix, solution_size
    )
    print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

    start_time = time.time()
    cover_list = find_all_covering_paths_v3(
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

    """
    TODO
    1. further prune trie 
    2. find_all_covering_paths should be restructured as set cover problem with no overlap
    - sort by longer words 
    - memoize visited combinations: k x k compatibility matrix?
    3. finding best cover
    - word embedding for similarity and incorporating hints
    - iterative method (check against solution for every word - not ideal but more realistic)
    """
