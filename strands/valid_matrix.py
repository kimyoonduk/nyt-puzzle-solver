import numpy as np

from .strands_helpers import path_to_word, check_crossing_path, is_trapped
from .paths import get_crossing_matrix


# precomputed matrix for checking valid path pairings
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

    return valid


# precomputed matrix for checking valid path pairings
def init_valid_matrix_v2(combined_bitmasks, combined_path_dict, board, min_zone_size=4):

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

            # if the sets words_i and words_j are exactly the same, then invalid
            if words_i == words_j:
                valid[i, j] = False
                valid[j, i] = False

    crossing_matrix = get_crossing_matrix(combined_bitmasks, combined_path_dict)

    valid = np.logical_and(valid, crossing_matrix)

    return valid


# update validity matrix for pair of bitmasks i and j, given a spangram
# if any cluster is smaller than min_zone_size, pair i, j is invalid
def update_valid_matrix(valid, span_bm, bitmask_list, n, m, min_zone_size=4):
    k = valid.shape[0]

    full_cover = (1 << (n * m)) - 1

    for i in range(k):
        bm_i = bitmask_list[i]

        for j in range(i + 1, k):
            bm_j = bitmask_list[j]

            available = full_cover & ~span_bm & ~bm_i & ~bm_j

            # check for trapped zones. if any cluster is less than min_zone_size, invalid
            invalid = is_trapped(available, n, m, min_zone_size)

            if invalid:
                valid[i, j] = False
                valid[j, i] = False

    return valid
