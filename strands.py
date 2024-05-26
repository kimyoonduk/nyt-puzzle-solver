import time
from collections import Counter
import json
from pathlib import Path

resource_dir = "resources"
word_file = "wordlist-20210729.txt"
input_file = "strands_input.json"

word_path = Path(resource_dir, word_file)
input_path = Path(resource_dir, input_file)

with open(word_path, "r") as f:
    word_list = f.read().splitlines()

with open(input_path, "r") as f:
    input_data = json.load(f)

big_input = input_data["big"]
small_input = input_data["small"]


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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(n):
        for j in range(m):
            for dx, dy in directions:
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


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def search_word(self, word):
        node = self.search_prefix(word)
        return node is not None and node.is_end_of_word


def build_trie(word_list):
    trie = Trie()
    for word in word_list:
        trie.insert(word)
    return trie


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


def find_all_covering_paths(bitmasks, span_bitmasks, n, m):
    full_cover = (1 << (n * m)) - 1  # Bitmask with all nodes covered
    all_solutions = []

    def backtrack(current_cover, path_index, selected_paths):
        if current_cover == full_cover:
            all_solutions.append(selected_paths[:])
            return

        for i in range(path_index, len(bitmasks)):
            if current_cover & bitmasks[i] == 0:  # No overlap
                selected_paths.append(i)
                backtrack(current_cover | bitmasks[i], i + 1, selected_paths)
                selected_paths.pop()

    # Try each spangram path as the starting point
    for i, span_bitmask in enumerate(span_bitmasks):
        backtrack(span_bitmask, 0, [i])

    return all_solutions


def get_all_words(matrix, word_list):

    word_list = optimize_word_list(matrix, word_list)

    n = len(matrix)
    m = len(matrix[0])

    longest_len = len(max(word_list, key=len))

    start_time = time.time()
    print(f"building trie: {len(word_list)} words")
    trie = build_trie(word_list)
    print(f"trie built: {time.time() - start_time:.4f}s")
    all_words = []
    all_paths = []
    span_words = []
    span_paths = []
    visited = [[False for _ in range(m)] for _ in range(n)]

    def dfs(x, y, path, current_word, current_node):
        if len(current_word) >= 4 and current_node.is_end_of_word:
            all_words.append(current_word)
            all_paths.append(path)

            if is_spangram(path, n, m):
                span_words.append(current_word)
                span_paths.append(path)

        if len(current_word) > longest_len:
            return

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        for dx, dy in directions:
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
        f"found {len(all_words)} words and {len(span_words)} spangrams: {time.time() - start_time:.4f}s"
    )

    # Convert paths to bitmasks
    bitmasks = [path_to_bitmask(path, n, m) for path in all_paths]
    span_bitmasks = [path_to_bitmask(path, n, m) for path in span_paths]

    solutions = []

    cover_list = find_all_covering_paths(bitmasks, span_bitmasks, n, m)
    if len(cover_list) > 0:

        for covering_idx in cover_list:
            covering_words = [span_words[covering_idx[0]]]
            covering_paths = [span_paths[covering_idx[0]]]

            covering_words += [all_words[i] for i in covering_idx[1:]]
            covering_paths += [all_paths[i] for i in covering_idx[1:]]

            solutions.append((covering_words, covering_paths))

        print(f"found {len(solutions)} covering paths: {time.time() - start_time:.4f}s")

    else:
        print(f"no covering paths: {time.time() - start_time:.4f}s")

    return solutions


def __main__():

    solutions = get_all_words(small_input, word_list)

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
