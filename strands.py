import time
import json
from pathlib import Path
import numpy as np
import datetime
import itertools

from lexicon import get_game

from util.trie import build_trie
from util.word_helpers import optimize_word_list
from util.strands_helpers import (
    is_spangram,
    check_crossing_path,
    path_to_word,
    cover_to_word,
    get_bitmask_list,
    divide_board_into_zones_with_merge,
    is_trapped,
)

# from util.integer_trie import build_integer_trie

# cardinal directions only
DIRECTIONS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# includes diagonals
DIRECTIONS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# big integer for placeholder solution size
BIG_NUMBER = 99999999


def get_local_word_list(word_file):
    parent_dir = Path(__file__).parent
    resource_dir = "resources"

    word_path = Path(parent_dir, resource_dir, word_file)
    with open(word_path, "r") as f:
        word_list = f.read().splitlines()

    return word_list


def save_results(results, filename):
    parent_dir = Path(__file__).parent
    resource_dir = "resources"

    with open(Path(parent_dir, resource_dir, filename), "w") as f:
        json.dump(results, f)


# given a matrix and a lexicon, return a list of all words and spangrams that can be formed
# includes filters for minimum word length, character counts, and valid sequences
def get_all_words(matrix, word_list, verbose=False):

    word_list = optimize_word_list(matrix, word_list)

    n = len(matrix)
    m = len(matrix[0])

    longest_len = len(max(word_list, key=len))

    start_time = time.time()
    if verbose:
        print(f"building trie: {len(word_list)} words")

    # build a trie from the filtered word list for efficient word search
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

            # word path cannot cross itself
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

    # run dfs on all possible starting points in the matrix
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


# given a validity matrix, a list of indices to check, and an optional valid index filter, return a list of valid indices
def get_valid_bm_idx(valid, idx_list, valid_idx_filter=None):
    # Convert idx_list to a numpy array for efficient operations
    idx_list = np.atleast_1d(idx_list)

    # If filtering index is empty, all indices in validity matrix are valid
    if len(idx_list) == 0:
        valid_idx = np.arange(valid.shape[0])
    else:
        # Get the AND of valid paths for all indices in idx_list
        valid_paths = np.logical_and.reduce(valid[idx_list])
        valid_idx = np.where(valid_paths)[0]

    # If valid_idx_filter is provided, filter valid_idx with only those that are in valid_idx_filter
    if valid_idx_filter is not None:
        valid_idx = np.intersect1d(valid_idx, valid_idx_filter, assume_unique=True)

    return valid_idx


"""
Version 5

Inputs
- all_paths: list of all word paths
- span_paths: list of all spangram paths
- matrix: game board of size n x m
- solution_size: number of words in the solution set
- timeout: time limit for each spangram search (time limit for entire game is 2x timeout)

Logic
- Convert all paths to bitmasks
- Precompute validity matrix for all bitmask pairs i and j, based on overlap, crossing, and word equivalence

- For each spangram, divide board into zones based on spangram paths
- Sort spangrams in ascending order of their max zone size
- Sort zones in ascending order of zone size

- For each zone, backtrack to find all valid solutions
- If zone is too large, resize and update validity matrix for further prune based on connected component size
- backtrack optimizations
    - tracks visited subsets to avoid redundant searches
    - early exit if possible solution size is exceeded
    - exit if zone times out
- Combine zone permutations to get spangram solutions

"""


def find_all_covering_paths_v5(
    all_paths, span_paths, matrix, solution_size=None, timeout=300
):

    n, m = len(matrix), len(matrix[0])

    # Convert paths to bitmasks
    bitmask_list, bitmask_path_dict = get_bitmask_list(all_paths, n, m)
    spanmask_list, spanmask_path_dict = get_bitmask_list(span_paths, n, m)
    combined_bitmasks = spanmask_list + bitmask_list
    combined_path_dict = {**bitmask_path_dict, **spanmask_path_dict}

    num_spans = len(spanmask_list)
    k = len(combined_bitmasks)

    if solution_size:
        filter_solutions_by_size = True
        print(f"searching solution of size: {solution_size}")
    else:
        filter_solutions_by_size = False
        solution_size = BIG_NUMBER
    print(f"span bitmasks: {num_spans}")
    print(f"all bitmasks: {k}")

    start = time.time()
    game_deadline = start + timeout * 2
    game_timed_out = False
    print(f"initializing validity matrix")
    # initialize matrix
    valid = init_valid_matrix(combined_bitmasks, combined_path_dict, matrix)
    print(f"matrix initialized: {time.time() - start:.4f}s")

    all_candidates = []
    all_solutions = []

    zones_by_span = []

    timed_out_span = -1
    time_to_first_solution = -1

    for span_idx, span_bitmask in enumerate(spanmask_list):

        spangram_path = combined_path_dict[span_bitmask][0]
        span_word = path_to_word(spangram_path, matrix)

        # get list of valid bitmasks
        valid_idx_list = get_valid_bm_idx(valid, span_idx)

        # divide board into two or more zones
        zonemask_list = divide_board_into_zones_with_merge(spangram_path, n, m)
        print(f"Span {span_idx}-{span_word}: {len(zonemask_list)} zones")

        if len(zonemask_list) == 0:
            print(f"No valid zones for span: {span_idx}-{span_word}")
            continue

        zone_bitmasks = [[] for _ in range(len(zonemask_list))]

        # assign path bitmasks to zones
        for bm_idx in valid_idx_list:
            bitmask = combined_bitmasks[bm_idx]

            for zone_idx, zonemask in enumerate(zonemask_list):

                if bitmask & zonemask:
                    zone_bitmasks[zone_idx].append(bm_idx)
                    break

        zones_by_span.append((span_idx, zone_bitmasks, zonemask_list))

    # sort spanmask_list by ascending order of max(zone sizes)
    sorted_spanmask_list = sorted(
        zones_by_span, key=lambda pair: max(len(zone) for zone in pair[1])
    )
    print(f"Searching {len(sorted_spanmask_list)} spans, sorted by max zone size")

    for span_idx, zone_bitmasks, zonemask_list in sorted_spanmask_list:

        if game_timed_out:
            print(f"Game timed out. Stopping search regardless of solutions.")
            break

        if (timed_out_span >= 0) and len(all_solutions) > 0:
            print(
                f"Span {timed_out_span} timed out. Stopping search and returning available solutions."
            )
            break

        span_start = time.time()
        span_deadline = span_start + timeout

        span_bitmask = combined_bitmasks[span_idx]
        spangram_path = combined_path_dict[span_bitmask][0]
        span_word = path_to_word(spangram_path, matrix)
        print(f"\n[{span_idx}]{span_word}: {len(zone_bitmasks)} zones")

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

        available_zone_count = solution_size - 1  # subtract 1 for the spangram

        for zone_idx, path_bitmask_list in enumerate(zone_bitmasks):
            print(f"Zone {span_idx}-{zone_idx}: {len(path_bitmask_list)} paths")

            def backtrack_zone(
                current_cover,
                current_path,
                solution_set,
                solution_mask,
                valid_idx_list,
                valid_matrix,
                bitmask_list,
            ):

                current_subset = frozenset(current_path)

                if current_cover == solution_mask:

                    solution_set.add(current_subset)

                    return

                if len(current_path) >= available_zone_count:
                    # print(f"Exceeded zone count: {len(current_path)} / {available_zone_count}")
                    visited_subsets.add(current_subset)
                    return

                if time.time() > span_deadline:
                    visited_subsets.add(current_subset)
                    return

                if current_subset in visited_subsets:
                    return
                visited_subsets.add(current_subset)

                new_valid_idx_list = get_valid_bm_idx(
                    valid_matrix, current_path, valid_idx_list
                )
                for i in new_valid_idx_list:
                    if current_cover & bitmask_list[i] == 0:
                        current_path.append(i)
                        backtrack_zone(
                            current_cover | bitmask_list[i],
                            current_path,
                            solution_set,
                            solution_mask,
                            new_valid_idx_list,
                            valid_matrix,
                            bitmask_list,
                        )
                        current_path.pop()

            # if too many paths in zone, resize matrix and filter further
            if len(path_bitmask_list) > 300:

                # create new valid matrix, combined_bitmasks, and combined_path_dict based on valid_idx_list
                zone_valid = valid[path_bitmask_list][:, path_bitmask_list]

                print(f"resizing validity matrix: {zone_valid.shape}")

                # combined_bitmasks are reindexed based on zone_valid
                zone_valid_bitmasks = [combined_bitmasks[i] for i in path_bitmask_list]
                new_bm_index = range(len(path_bitmask_list))

                # update valid span
                zone_valid = update_valid_matrix(
                    zone_valid, span_bitmask, zone_valid_bitmasks, n, m
                )

                # maintain a dict from new index to original index
                resized_to_orig = {i: path_bitmask_list[i] for i in new_bm_index}

                zone_solution_resized = set()

                for i in new_bm_index:
                    backtrack_zone(
                        zone_valid_bitmasks[i],
                        [i],
                        zone_solution_resized,
                        zonemask_list[zone_idx],
                        new_bm_index,
                        zone_valid,
                        zone_valid_bitmasks,
                    )

                # convert zone_solution_resized (set of frozensets) to original indices
                zone_solution_resized = {
                    frozenset(resized_to_orig[i] for i in sol)
                    for sol in zone_solution_resized
                }

                zone_solutions[zone_idx] = zone_solution_resized

            else:
                for i in path_bitmask_list:
                    backtrack_zone(
                        combined_bitmasks[i],
                        [i],
                        zone_solutions[zone_idx],
                        zonemask_list[zone_idx],
                        path_bitmask_list,
                        valid,
                        combined_bitmasks,
                    )

            # if zone timed out, skip this spangram
            if time.time() > span_deadline:
                print(f"Zone {span_idx}-{zone_idx} timed out. Skipping spangram.")
                timed_out_span = span_idx
                span_has_solution = False
                break

            # if zone does not have solution, skip this spangram
            if len(zone_solutions[zone_idx]) == 0:
                print(f"Zone {span_idx}-{zone_idx} has no solution. Skipping spangram.")

                span_has_solution = False
                break

            # update available zone count
            # get unique solutions
            zone_solution_list = list(zone_solutions[zone_idx])
            # subtract size of smallest zone_solution

            min_zone_size = len(min(zone_solution_list, key=len))
            available_zone_count -= min_zone_size
            print(
                f"Zone {span_idx}-{zone_idx}: {len(zone_solution_list)} solutions with min size {min_zone_size}."
            )

        print(f"span finished: {time.time() - span_start:.4f}s")

        if not span_has_solution:
            continue

        # combine zone solutions
        # TODO: refactor with list(itertools.product(*solution))
        span_solutions = []

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
            if (len(all_solutions) == 0) and (time_to_first_solution < 0):
                time_to_first_solution = time.time() - start
                print(f"First solution found in {time_to_first_solution:.4f}s")
            for zone_idx in range(len(zone_solutions)):
                print(f"Zone {span_idx}-{zone_idx}: {len(zone_solutions[zone_idx])}")

            if filter_solutions_by_size:
                # filter solutions by solution size
                filtered_solutions = [
                    sol for sol in span_solutions if len(sol) == solution_size
                ]
                print(
                    f"[{span_idx}]{span_word}: {len(span_solutions)} candidate solutions found. {len(filtered_solutions)} solutions of size {solution_size}"
                )
            else:
                filtered_solutions = span_solutions
                print(
                    f"[{span_idx}]{span_word}: {len(span_solutions)} candidate solutions found."
                )

            all_candidates.extend(span_solutions)
            all_solutions.extend(filtered_solutions)

        # update game timeout flag
        game_timed_out = time.time() > game_deadline

    solution_path_list = []
    candidate_path_list = []

    for solution in all_solutions:
        solution_path = [combined_path_dict[combined_bitmasks[idx]] for idx in solution]
        solution_path_list.append(solution_path)

    for candidate in all_candidates:
        candidate_path = [
            combined_path_dict[combined_bitmasks[idx]] for idx in candidate
        ]
        candidate_path_list.append(candidate_path)

    timed_out = (
        True if (timed_out_span >= 0 and len(solution_path_list) == 0) else False
    )

    elapsed_time = time.time() - start

    solution_object = {
        "solution_path_list": solution_path_list,
        "candidate_path_list": candidate_path_list,
        "timed_out": timed_out,
        "elapsed_time": elapsed_time,
        "time_to_first_solution": time_to_first_solution,
    }

    return solution_object


def test_published_games(game_date=None, timeout=300):

    word_list = get_local_word_list("wordlist-v4.txt")

    start_date = datetime.date(2024, 3, 4)
    today = datetime.date.today()

    candidate_found_count = 0
    solved_count = 0
    correct_solve_count = 0
    timeout_count = 0

    results = {}

    if game_date:
        start_date = datetime.datetime.strptime(game_date, "%Y-%m-%d").date()
        today = start_date

    while start_date <= today:
        date_str = start_date.strftime("%Y-%m-%d")

        game = get_game(date_str)

        matrix = game["matrix"]
        clue = game["clue"]
        solution_count = game["solution_count"]
        given_solution = game["solution_words"]
        solution_paths = game["solution_paths"]
        spangram = game["spangram"]
        game_date = game["date"]
        print(f"Starting game: {game_date} - {clue}")

        n = len(matrix)
        m = len(matrix[0])

        all_paths, span_paths = get_all_words(matrix, word_list)
        all_words = [path_to_word(path, matrix) for path in all_paths]
        span_words = [path_to_word(path, matrix) for path in span_paths]
        print(f"Word paths found: {len(all_words)}")
        print(f"Span paths found: {len(span_words)}")

        for row in matrix:
            print("  ".join(row))

        start_time = time.time()
        solution_object = find_all_covering_paths_v5(
            all_paths, span_paths, matrix, solution_count, timeout
        )
        print(f"Search completed")
        print(f"Given solution: {[spangram] + given_solution}")

        solution_path_list = solution_object["solution_path_list"]
        candidate_path_list = solution_object["candidate_path_list"]
        timed_out = solution_object["timed_out"]
        elapsed_time = solution_object["elapsed_time"]
        time_to_first_solution = solution_object["time_to_first_solution"]

        solutions_in_words = [
            cover_to_word(cover, matrix) for cover in solution_path_list
        ]

        all_solution_permuations = set()

        for solution in solutions_in_words:
            print(solution)
            solution_perms = list(itertools.product(*solution))

            for perm in solution_perms:
                all_solution_permuations.add(frozenset(perm))

        print(
            f"found {len(all_solution_permuations)} solutions: {time.time() - start_time:.4f}s"
        )

        given_solution_set = frozenset([spangram] + given_solution)

        candidate_found = len(candidate_path_list) > 0
        solved = len(solution_path_list) > 0
        correct_solution = given_solution_set in all_solution_permuations

        # convert all_solution_permuations to a list of lists
        all_solution_permuations = [list(sol) for sol in all_solution_permuations]

        if candidate_found:
            candidate_found_count += 1

        if solved:
            solved_count += 1

        if correct_solution:
            print(f"Correct solution found")
            correct_solve_count += 1
        else:
            print(f"Correct solution not found")

        if timed_out:
            timeout_count += 1

        results[game_date] = {
            "solutions": all_solution_permuations,
            "solution_paths": solution_path_list,
            "candidate_paths": candidate_path_list,
            "solution_words": solutions_in_words,
            "candidate_words": [
                cover_to_word(cover, matrix) for cover in candidate_path_list
            ],
            "timed_out": timed_out,
            "candidate_found": candidate_found,
            "solved": solved,
            "correct_solution": correct_solution,
            "elapsed_time": elapsed_time,
            "time_to_first_solution": time_to_first_solution,
        }

        start_date += datetime.timedelta(days=1)
        print(f"Game {game_date} Completed in {elapsed_time:.4f}s.\n")

    stats = {
        "candidate_found": candidate_found_count,
        "solved": solved_count,
        "correct_solve": correct_solve_count,
        "timeout": timeout_count,
    }

    return results, stats


def solve(matrix, word_list, solution_count=None, timeout=300):

    n = len(matrix)
    m = len(matrix[0])

    all_paths, span_paths = get_all_words(matrix, word_list)
    all_words = [path_to_word(path, matrix) for path in all_paths]
    span_words = [path_to_word(path, matrix) for path in span_paths]
    print(f"Word paths found: {len(all_words)}")
    print(f"Span paths found: {len(span_words)}")

    for row in matrix:
        print("  ".join(row))

    start_time = time.time()
    solution_object = find_all_covering_paths_v5(
        all_paths, span_paths, matrix, solution_count, timeout
    )
    print(f"Search completed")

    solution_path_list = solution_object["solution_path_list"]
    candidate_path_list = solution_object["candidate_path_list"]
    timed_out = solution_object["timed_out"]
    elapsed_time = solution_object["elapsed_time"]
    time_to_first_solution = solution_object["time_to_first_solution"]

    solutions_in_words = [cover_to_word(cover, matrix) for cover in solution_path_list]

    all_solution_permuations = set()

    for solution in solutions_in_words:
        print(solution)
        solution_perms = list(itertools.product(*solution))

        for perm in solution_perms:
            all_solution_permuations.add(frozenset(perm))

    print(
        f"found {len(all_solution_permuations)} solutions: {time.time() - start_time:.4f}s"
    )

    # convert all_solution_permuations to a list of lists
    all_solution_permuations = [list(sol) for sol in all_solution_permuations]

    if timed_out:
        print(f"Search timed out: {elapsed_time:.4f}s / {timeout}s")

    result = {
        "solutions": all_solution_permuations,
        "solution_paths": solution_path_list,
        "candidate_paths": candidate_path_list,
        "solution_words": solutions_in_words,
        "candidate_words": [
            cover_to_word(cover, matrix) for cover in candidate_path_list
        ],
        "timed_out": timed_out,
        "elapsed_time": elapsed_time,
        "time_to_first_solution": time_to_first_solution,
    }

    print(f"Game Completed in {elapsed_time:.4f}s.\n")

    return result


def __main__():

    results, stats = test_published_games(game_date="2024-06-04", timeout=180)

    today = datetime.date.today()
    save_results(results, f"results-{today}-1.json")

    # get average time_to_first_solution, where time_to_first_solution > 0
    time_to_first_solution_list = [
        result["time_to_first_solution"]
        for result in results.values()
        if result["time_to_first_solution"] > 0
    ]
    np.median(time_to_first_solution_list)

    # get number of correct solutions
    correct_solution_count = len(
        [result for result in results.values() if result["correct_solution"]]
    )

    stats
    len(results)

    word_list = get_local_word_list("wordlist-v4.txt")
    result = solve(
        [
            ["B", "A", "N", "A"],
            ["N", "A", "I", "T"],
            ["F", "R", "U", "L"],
            ["E", "I", "E", "P"],
            ["M", "L", "A", "P"],
        ],
        word_list,
    )
    print(result)
