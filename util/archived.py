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
