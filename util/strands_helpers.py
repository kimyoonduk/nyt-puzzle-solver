def is_spangram(path, n, m):
    touches_top = any(x == 0 for x, y in path)
    touches_bottom = any(x == n - 1 for x, y in path)
    touches_left = any(y == 0 for x, y in path)
    touches_right = any(y == m - 1 for x, y in path)
    return (touches_top and touches_bottom) or (touches_left and touches_right)


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


# given two paths, check if the lines cross (not overlap)
def check_crossing_path(path1, path2):
    for i in range(len(path1) - 1):
        for j in range(len(path2) - 1):
            if check_crossing_seg((path1[i], path1[i + 1]), (path2[j], path2[j + 1])):
                return True

    return False


def path_to_word(path, matrix):
    return "".join(matrix[x][y] for x, y in path)


def cover_to_word(cover, matrix):
    return [[path_to_word(path, matrix) for path in path_list] for path_list in cover]


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
