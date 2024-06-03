from collections import defaultdict

from util.strands_helpers import check_crossing_path

# cardinal directions only
DIRECTIONS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# includes diagonals
DIRECTIONS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# converts a path of coordinates in a n x m matrix to a bitmask
def path_to_bitmask(path, n, m):
    bitmask = 0
    for x, y in path:
        bitmask |= 1 << (x * m + y)
    return bitmask


# given a list of paths, returns a list of bitmasks and a dictionary mapping each bitmask to the paths that correspond to it
def get_bitmask_list(path_list, n, m):

    bitmask_path_dict = defaultdict(list)
    bitmasks = set()

    for path in path_list:

        bitmask = path_to_bitmask(path, n, m)
        bitmasks.add(bitmask)
        bitmask_path_dict[bitmask].append(path)

    bitmask_list = list(bitmasks)

    return bitmask_list, bitmask_path_dict


# count number of 1s in bitmask
def count_set_bits(x):
    count = 0
    while x:
        x &= x - 1  # Remove the rightmost set bit
        count += 1
    return count


# bitmask version of dfs to find connected components in the cardinal direction
def dfs_bitmask(input_bm, n, m):
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
            clusters.append(cluster)

    return clusters


# check for "trapped" connected components of size less than min_zone_size
def is_trapped(input_bm, n, m, min_zone_size=4):
    # bitmask to track visited nodes
    visited = 0
    clusters = []

    def dfs(start, visited):
        invalid = False
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
                for dx, dy in DIRECTIONS_8:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m:
                        npos = nx * m + ny
                        if (input_bm & (1 << npos)) and not (visited & (1 << npos)):
                            stack.append(npos)

        if count_set_bits(cluster) < min_zone_size:
            invalid = True

        return cluster, visited, invalid

    for i in range(n * m):
        # if node is part of input_bm and not visited
        if (input_bm & (1 << i)) and not (visited & (1 << i)):
            cluster, visited, invalid = dfs(i, visited)

            if invalid:
                return True

    return False


def divide_board_into_zones_bm(span_bitmask, n, m, min_zone_size=4):
    full_cover = (1 << (n * m)) - 1  # Bitmask with all nodes covered
    available = full_cover & ~span_bitmask

    clusters = dfs_bitmask(available, n, m)

    if len(clusters) == 0:
        return []

    # check if any clusters are too small
    for cluster in clusters:
        if count_set_bits(cluster) < min_zone_size:
            print(
                f"Cluster too small: {count_set_bits(cluster)}. Returning empty zones"
            )
            return []

    return clusters


# for debugging
def bitmask_to_coordinates(bitmask, n, m):
    coordinates = []
    for i in range(n):
        for j in range(m):
            if bitmask & (1 << (i * m + j)):
                coordinates.append((i, j))
    return coordinates


def get_connection(cluster, clusters, span_path, span_bitmask, n, m):
    expanded_nodes = 0

    coordinates = bitmask_to_coordinates(cluster, n, m)

    # iterate through coordinates and expand in 8 directions
    for x, y in coordinates:
        for dx, dy in DIRECTIONS_8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m:
                npos = nx * m + ny
                # check new node does not cross any span path
                new_seg = [(x, y), (nx, ny)]
                cross_span = check_crossing_path(span_path, new_seg)
                overlap_span = (1 << npos) & span_bitmask

                if not cluster & (1 << npos) and not overlap_span and not cross_span:
                    expanded_nodes |= 1 << npos

    # check if any expanded nodes are part of other clusters
    for i, other_cluster in enumerate(clusters):
        # assume only one cluster will have a connection
        if other_cluster & expanded_nodes and other_cluster != cluster:
            return i

    return -1


def divide_board_into_zones_with_merge(spangram_path, n, m, min_zone_size=4):
    span_bitmask = path_to_bitmask(spangram_path, n, m)
    full_cover = (1 << (n * m)) - 1  # Bitmask with all nodes covered
    available = full_cover & ~span_bitmask

    clusters = dfs_bitmask(available, n, m)

    new_clusters = []

    # union find probably overkill?
    while clusters:
        merged_this_iteration = False
        i = 0
        while i < len(clusters):
            cluster = clusters[i]
            if count_set_bits(cluster) < min_zone_size:
                connect_idx = get_connection(
                    cluster, clusters, spangram_path, span_bitmask, n, m
                )
                if connect_idx >= 0:
                    # Merge clusters and add to new_clusters
                    new_clusters.append(cluster | clusters[connect_idx])
                    # Remove both clusters from the list to avoid reprocessing
                    clusters.pop(max(i, connect_idx))
                    clusters.pop(min(i, connect_idx))
                    merged_this_iteration = True
                    # Start over as the list has changed
                    continue
                else:
                    print(f"Can't merge cluster of size: {count_set_bits(cluster)}")
                    return []
            i += 1  # Only increment if no merge happened

        if not merged_this_iteration:
            # No more merges possible, add remaining clusters to new_clusters
            new_clusters.extend(clusters)
            break  # Exit the loop as no more action is possible

    return new_clusters
