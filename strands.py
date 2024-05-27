import time
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from util.trie import build_trie

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


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
            for dx, dy in DIRECTIONS:
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


def get_valid_sequences(matrix, length):
    n = len(matrix)
    m = len(matrix[0])
    sequences = set()

    def dfs(x, y, current_sequence):
        if len(current_sequence) == length:
            sequences.add(tuple(current_sequence))
            return

        for dx, dy in DIRECTIONS:
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


def get_bitmasks(path_list, n, m):

    bitmask_path_dict = defaultdict(list)
    bitmasks = set()

    for path in path_list:
        bitmask = path_to_bitmask(path, n, m)
        bitmasks.add(bitmask)
        bitmask_path_dict[bitmask].append(path)

    bitmask_list = list(bitmasks)

    return bitmask_list, bitmask_path_dict


# can be defined as two lines having one solution
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


def find_all_covering_paths(all_paths, span_paths, n, m):

    # Convert paths to bitmasks
    bitmask_list, bitmask_path_dict = get_bitmasks(all_paths, n, m)
    spanmask_list, spanmask_path_dict = get_bitmasks(span_paths, n, m)
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
def init_valid_matrix(k, combined_bitmasks, combined_path_dict):
    valid = np.ones((k, k), dtype=bool)
    np.fill_diagonal(valid, False)

    # check for overlaps and crossing
    for i in range(k):
        for j in range(i + 1, k):
            bm_i = combined_bitmasks[i]
            bm_j = combined_bitmasks[j]

            # overlap
            if bm_i & bm_j:
                valid[i, j] = False
                valid[j, i] = False

            # crossing
            # any path is okay
            if check_crossing_path(
                combined_path_dict[bm_i][0],
                combined_path_dict[bm_j][0],
            ):
                valid[i, j] = False
                valid[j, i] = False

    return valid


def get_valid_bm_idx(valid, idx_list):

    # get AND of valid paths for all idx in idx_list
    if isinstance(idx_list, int):
        idx_list = [idx_list]

    valid_paths = valid[idx_list[0]]

    if len(idx_list) > 1:
        for idx in idx_list[1:]:
            valid_paths = np.logical_and(valid_paths, valid[idx])

    valid_idx = np.where(valid_paths)[0]

    return valid_idx


def divide_board_into_zones(spangram_path, spangram_bm, n, m):
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


# for debugging
def bitmask_to_coordinates(bitmask, n, m):
    coordinates = []
    for i in range(n):
        for j in range(m):
            if bitmask & (1 << (i * m + j)):
                coordinates.append((i, j))
    return coordinates


def find_all_covering_paths_zone(all_paths, span_paths, n, m, solution_size):

    # Convert paths to bitmasks
    bitmask_list, bitmask_path_dict = get_bitmasks(all_paths, n, m)
    spanmask_list, spanmask_path_dict = get_bitmasks(span_paths, n, m)
    combined_bitmasks = spanmask_list + bitmask_list
    combined_path_dict = {**bitmask_path_dict, **spanmask_path_dict}

    num_spans = len(spanmask_list)
    k = len(combined_bitmasks)

    print(f"span bitmasks: {num_spans}")
    print(f"all bitmasks: {k}")

    # initialize matrix
    valid = init_valid_matrix(k, combined_bitmasks, combined_path_dict)

    full_cover = (1 << (n * m)) - 1  # Bitmask with all nodes covered
    all_solutions = []

    for span_idx, span_bitmask in enumerate(spanmask_list):

        spangram_path = combined_path_dict[span_bitmask][0]
        print(f"span {span_idx}: {path_to_word(spangram_path)}")

        # get list of valid bitmasks
        valid_idx_list = get_valid_bm_idx(valid, span_idx)
        print(f"valid count: {len(valid_idx_list)}")

        # divide board into two zones
        z1_mask, z2_mask = divide_board_into_zones(spangram_path, span_bitmask, n, m)

        print(f"z1: {bitmask_to_coordinates(z1_mask, n, m)}")
        print(f"z2: {bitmask_to_coordinates(z2_mask, n, m)}")
        print(f"span: {bitmask_to_coordinates(span_bitmask, n, m)}")

        # assign valid bitmasks to each zone
        z1_bm = []
        z2_bm = []

        span_bitmask = combined_bitmasks[span_idx]

        # for bm_idx in valid_idx_list:
        #     bitmask = combined_bitmasks[bm_idx]
        #     if bitmask & z1_mask:
        #         z1_bm.append(bm_idx)
        #     elif bitmask & z2_mask:
        #         z2_bm.append(bm_idx)
        #     else:
        #         # this shouldn't happen if get_valid_bm_idx is working correctly
        #         raise ValueError("Path does not belong to either zone")

        z1_solutions = []
        z2_solutions = []

        # TODO: fix zoning and run separately to combine answers later

        def backtrack_zone(current_cover, bm_list, solutions, zone_mask):

            print(f"current bms: {bm_list}")

            if current_cover == zone_mask:
                solutions.append(bm_list[:])
                print(bm_list)
                return

            new_valid_idx_list = get_valid_bm_idx(valid, bm_list)
            for i in new_valid_idx_list:
                if current_cover & combined_bitmasks[i] == 0:
                    bm_list.append(i)
                    backtrack_zone(
                        current_cover | combined_bitmasks[i],
                        bm_list,
                        solutions,
                        zone_mask,
                    )
                    bm_list.pop()

        backtrack_zone(span_bitmask, [span_idx], all_solutions, full_cover)

    return all_solutions

    return z1_bm, z2_bm

    # def backtrack(current_cover, path_index, selected_paths):
    #     # print(selected_paths)

    #     if current_cover == full_cover:
    #         all_solutions.append(selected_paths[:])

    #         return

    #     for i in range(path_index, len(combined_bitmasks)):
    #         if current_cover & combined_bitmasks[i] == 0:  # No overlap
    #             selected_paths.append(i)
    #             backtrack(current_cover | combined_bitmasks[i], i + 1, selected_paths)
    #             selected_paths.pop()

    # # Try each spangram path as the starting point
    # for i, span_bitmask in enumerate(spanmask_list):
    #     backtrack(span_bitmask, 0, [i])

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
    bitmask_list, bitmask_path_dict = get_bitmasks(all_paths, n, m)
    spanmask_list, spanmask_path_dict = get_bitmasks(span_paths, n, m)
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

        for dx, dy in DIRECTIONS:
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
    small_input = input_data["small"]

    solution_size = 9

    matrix = big_input
    matrix = small_input

    all_paths, span_paths = get_all_words(matrix, word_list)
    all_words = [path_to_word(path, matrix) for path in all_paths]
    span_words = [path_to_word(path, matrix) for path in span_paths]
    # print(len(all_words))
    # print(len(span_words))

    n = len(matrix)
    m = len(matrix[0])

    start_time = time.time()
    cover_list = find_all_covering_paths(all_paths, span_paths, n, m)
    print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

    start_time = time.time()
    cover_list = find_all_covering_paths_track(all_paths, span_paths, n, m)
    print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

    start_time = time.time()
    # z1_bm, z2_bm = find_all_covering_paths_zone(all_paths, span_paths, n, m)
    cover_list = find_all_covering_paths_zone(all_paths, span_paths, n, m)
    print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

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
