from collections import Counter

# cardinal directions only
DIRECTIONS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# includes diagonals
DIRECTIONS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# given a matrix and a list of words, filter words based on character counts and valid sequences
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

    # word_list = filter_words_by_valid_sequences(word_list, matrix)

    print(f"Optimized Word List Length: {len(word_list)}")

    return word_list


# given a matrix and a list of words, return a list of words that can be formed using the characters in the matrix
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


# find all pairs of adjacent characters in the matrix
def get_adjacent_pairs(matrix):
    n = len(matrix)
    m = len(matrix[0])
    pairs = set()

    for i in range(n):
        for j in range(m):
            for dx, dy in DIRECTIONS_8:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < m:
                    pairs.add((matrix[i][j], matrix[ni][nj]))
                    pairs.add(
                        (matrix[ni][nj], matrix[i][j])
                    )  # Add reverse pair as well

    return pairs


# given a list of words and a list of valid pairs of characters, return a list of words that do not contain any invalid pairs of characters
def filter_words_by_adjacent_pairs(word_list, valid_pairs):
    def has_invalid_pairs(word):
        for i in range(len(word) - 1):
            if (word[i], word[i + 1]) not in valid_pairs:
                return True
        return False

    return [word for word in word_list if not has_invalid_pairs(word)]
