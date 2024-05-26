import time
from collections import Counter, defaultdict
import json
from pathlib import Path

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

        if current_cover == full_cover:
            all_solutions.append(selected_paths[:])
            if len(all_solutions) % 10 == 0:
                print(len(all_solutions))

            return

        for i in range(path_index, len(combined_bitmasks)):
            if current_cover & combined_bitmasks[i] == 0:  # No overlap
                selected_paths.append(i)
                backtrack(current_cover | combined_bitmasks[i], i + 1, selected_paths)
                selected_paths.pop()

    # Try each spangram path as the starting point
    for i, span_bitmask in enumerate(spanmask_list):
        backtrack(span_bitmask, num_spans, [i])

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

        if current_combination in explored_combination:
            return
        # Mark the combination as explored
        explored_combination.add(current_combination)

        if current_cover == full_cover:
            all_solutions.append(current_combination)

            if len(all_solutions) % 10 == 0:
                print(len(all_solutions))
            return

        for i in range(num_spans, len(combined_bitmasks)):
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
def get_all_words(matrix, word_list):

    word_list = optimize_word_list(matrix, word_list)

    n = len(matrix)
    m = len(matrix[0])

    longest_len = len(max(word_list, key=len))

    start_time = time.time()
    print(f"building trie: {len(word_list)} words")
    trie = build_trie(word_list)
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
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
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

    print(
        f"found {len(all_paths)} words and {len(span_paths)} spangrams: {time.time() - start_time:.4f}s"
    )

    return all_paths, span_paths


def cover_to_word(cover, matrix):
    return [[path_to_word(path, matrix) for path in path_list] for path_list in cover]


def __main__():

    resource_dir = "resources"
    word_file = "wordlist-20210729.txt"
    # word_file = "wordlist-v2.txt"
    input_file = "strands_input.json"

    word_path = Path(resource_dir, word_file)
    input_path = Path(resource_dir, input_file)

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

    n = len(matrix)
    m = len(matrix[0])

    start_time = time.time()
    cover_list = find_all_covering_paths(all_paths, span_paths, n, m)
    print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

    cover_list[0]
    cover_to_word(cover_list[37], matrix)

    start_time = time.time()
    cover_list = find_all_covering_paths_track(all_paths, span_paths, n, m)
    print(f"found {len(cover_list)} covering paths: {time.time() - start_time:.4f}s")

    cover_list[0]
    cover_to_word(cover_list[37], matrix)

    """
    Word List Length: 198422
    Optimized Word List Length: 602
    building trie: 602 words
    trie built: 0.0005s
    found 417 words and 87 spangrams: 0.0020s
    found 298 covering paths: 0.5721s
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
