import numpy as np


def check_crossing_path_numpy(l1, l2, get_coords=False):
    # Convert lists to numpy arrays

    # len(l1) = m, len(l2) = n
    l1 = np.array(l1)
    l2 = np.array(l2)

    # get segments of l1 and l2
    v1 = np.diff(l1, axis=0)
    v2 = np.diff(l2, axis=0)

    # slope of v1 (convert inf or -inf to 0 for convenience)
    # dims: m-1
    m1 = np.divide(v1[:, 1], v1[:, 0], where=(v1[:, 0] != 0))

    # slope of v2
    # dims: n-1
    m2 = np.divide(v2[:, 1], v2[:, 0], where=(v2[:, 0] != 0))

    # get boolean mask for m1 * m2 = -1
    # this indicates that the two segments cross diagonally
    # dims: (m-1)x(n-1)
    cross = m1[:, np.newaxis] * m2[np.newaxis, :] == -1

    # get start points of l1 and l2
    # dims: (m-1)x2, (n-1)x2
    l1_start = l1[:-1]
    l2_start = l2[:-1]

    # get matrix of diff between l1_start and l2_start
    # dims: (m-1)x(n-1)x2
    diff = l1_start[:, np.newaxis, :] - l2_start[np.newaxis, :, :]

    # get sums to calculate adjacency and relative position
    # dims: (m-1)x(n-1)
    diff_sum = diff.sum(axis=2)
    diff_sum_abs = np.abs(diff).sum(axis=2)

    # diff_sum_abs == 1 indicates the two points are adjacent by 1 unit
    adjacent = diff_sum_abs == 1

    # diff_sum == -1: l1 is left of or below l2
    # m1 == 1 results in cross
    # dims: (m-1)x(n-1)
    l1_left = adjacent & (diff_sum == -1)

    # get all points where m2 == 1 results in a cross
    # diff_sum == 1: l1 is right of or above l2
    # dims: (m-1)x(n-1)
    l1_right = adjacent & (diff_sum == 1)

    # get all crossing points
    # dims: (m-1)x(n-1)
    all_cross = cross & ((l1_left & (m2 == -1)) | (l1_right & (m2 == 1)))

    if get_coords:
        # return coordinates of all crossing points
        return np.where(all_cross)
    else:
        # cross exists
        return np.any(all_cross)


def get_crossing_matrix(combined_bitmasks, combined_path_dict):

    k = len(combined_bitmasks)

    flattened_paths = []
    path_index_array = []
    end_index_array = []

    for i in range(k):
        bitmask = combined_bitmasks[i]

        for path in combined_path_dict[bitmask]:
            flattened_paths.extend(path)
            path_index_array.extend([i] * len(path))

            # denote end index of each path to separate paths
            end_index_array.append(len(flattened_paths) - 1)

    flattened_paths = np.array(flattened_paths)

    crossing_coords = check_crossing_path_numpy(flattened_paths, flattened_paths, True)

    crossing_matrix = np.ones((k, k), dtype=bool)

    for i, j in zip(*crossing_coords):
        # if the paths are from different clusters
        cluster_i = path_index_array[i]
        cluster_j = path_index_array[j]

        if (
            cluster_i != cluster_j
            and i not in end_index_array
            and j not in end_index_array
        ):
            crossing_matrix[cluster_i, cluster_j] = False
            crossing_matrix[cluster_j, cluster_i] = False

    return crossing_matrix


def test():

    l1 = [(0, 0), (1, 1), (1, 2), (2, 2), (3, 1), (4, 1)]
    l2 = [(0, 1), (1, 0), (2, 1), (3, 2), (3, 3)]
    print(check_crossing_path_numpy(l1, l2))  # Outputs: True

    coords = check_crossing_path_numpy(l1, l2, True)
    print(coords)  # Outputs: [[0, 3], [0, 2]]

    for i, j in zip(*coords):
        print(i, j)
