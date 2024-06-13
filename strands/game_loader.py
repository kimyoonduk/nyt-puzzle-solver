from pathlib import Path
import random
import json

import datetime
import requests

import time
import random

from .strands_helpers import get_all_words, path_to_word

resource_dir = Path(__file__).parent.parent / "resources"
strands_dir = resource_dir / "strands"
sb_dir = resource_dir / "spellingbee"

json_list = list(strands_dir.glob("*.json"))


def run_v1():

    wl_path = resource_dir / "wordlist-20210729.txt"

    with open(wl_path, "r") as f:
        word_list = f.read().splitlines()

    word_list = [word.upper() for word in word_list]
    word_list = [word for word in word_list if len(word) >= 4]

    word_set = set(word_list)

    for json_path in json_list:
        with open(json_path, "r") as f:
            game_data = json.load(f)

        board = game_data["startingBoard"]
        matrix = [[char for char in row] for row in board]

        all_paths, span_paths = get_all_words(matrix, word_list)
        combined_paths = span_paths + all_paths
        all_words = [path_to_word(path, matrix) for path in combined_paths]
        all_words = list(set(all_words))

        solutions = game_data["solutions"]

        drop_counter = 0
        # for word in all_words but not in solutions, remove from word_set

        for word in all_words:
            if word not in solutions:
                drop_counter += 1
                word_set.discard(word)

        add_counter = 0
        # for word in solutions but not in all_words, add to word_set
        for word in solutions:
            if word not in all_words:
                add_counter += 1
                # print(f"added {word} to word_set")
                word_set.add(word)
        print(f"all_words length: {len(all_words)}")
        print(f"solutions length: {len(solutions)}")
        print(f"dropped {drop_counter} words and added {add_counter} words")
        print(f"word_set length: {len(word_set)}")

    # save word_set to \n separated txt file
    word_set = sorted(list(word_set))
    with open(resource_dir / "wordlist-v3.txt", "w") as f:
        f.write("\n".join(word_set))


def run_v2():

    wl_path = resource_dir / "wordlist-v3.txt"

    with open(wl_path, "r") as f:
        word_list = f.read().splitlines()

    word_list = [word.upper() for word in word_list]
    word_list = [word for word in word_list if len(word) >= 4]

    word_set = set(word_list)
    print(len(word_set))

    for json_path in json_list:
        with open(json_path, "r") as f:
            word_data = json.load(f)

        # four letter string of unique characters
        stem = json_path.stem

        # find all words in the word_set that contain all chars in stem in any order
        matching_words = [
            word for word in word_set if all(char in word for char in stem.upper())
        ]

        drop_counter = 0
        # if word in matching_word is not in word_data, drop from word_set
        for word in matching_words:
            if word not in word_data:
                # word_set.discard(word)
                drop_counter += 1

        # if word in word_data is not in matching_words, add to word_set
        add_counter = 0
        for word in word_data:
            if word not in matching_words:
                word_set.add(word)
                add_counter += 1

        print(f"dropped {drop_counter} words and added {add_counter} words")

    len(word_set)

    # save word_set to \n separated txt file
    word_set = sorted(list(word_set))
    with open(resource_dir / "wordlist-v4.txt", "w") as f:
        f.write("\n".join(word_set))


games = {}

for json_path in json_list:
    with open(json_path, "r") as f:
        game_data = json.load(f)

        # print(game_data)

        print_date = game_data["printDate"]
        board_in_string = game_data["startingBoard"]
        matrix = [[char for char in row] for row in board_in_string]

        # solution_words = game_data["themeWords"]

        solution_dicts = game_data["themeCoords"]
        solution_paths = [val for _, val in solution_dicts.items()]
        solution_words = [key for key, _ in solution_dicts.items()]
        solution_count = len(solution_paths) + 1

        games[print_date] = {
            "date": print_date,
            "matrix": matrix,
            "clue": game_data["clue"],
            "solution_count": solution_count,
            "solution_words": solution_words,
            "solution_paths": solution_paths,
            "spangram": game_data["spangram"],
        }


def get_game(print_date=None):

    if print_date:
        try:
            return games[print_date]
        except KeyError:
            print(f"print date {print_date} not found")
            return None

    # return random date from games
    else:
        random_item_idx = random.choice(list(games.keys()))
        return games[random_item_idx]


def collect_connections(start_date=None, end_date=None):

    if not start_date:
        start_date = datetime.date(2023, 6, 12)

    if not end_date:
        end_date = datetime.date.today()

    save_dir = resource_dir / "connections"

    while start_date <= end_date:
        date_str = start_date.strftime("%Y-%m-%d")

        file_path = save_dir / f"{date_str}.json"

        if file_path.exists():
            start_date += datetime.timedelta(days=1)
            continue

        url = f"https://www.nytimes.com/svc/connections/v2/{date_str}.json"

        res = requests.get(url)

        json_data = res.json()

        if json_data["status"] == "OK":
            with open(file_path, "w") as f:
                json.dump(json_data, f)

        # randomly sleep between 0.3-0.7 seconds
        sleep_time = random.uniform(0.3, 0.7)
        time.sleep(sleep_time)

        start_date += datetime.timedelta(days=1)
