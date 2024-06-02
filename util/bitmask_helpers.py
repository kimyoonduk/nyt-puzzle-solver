from collections import defaultdict


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
