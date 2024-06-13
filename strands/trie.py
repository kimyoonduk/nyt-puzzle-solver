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

    def size(self):
        return self._get_size_recursive(self.root)

    def _get_size_recursive(self, node):
        size = 0
        for child in node.children.values():
            size += self._get_size_recursive(child)
        return size + 1


def build_trie(word_list):
    trie = Trie()
    for word in word_list:
        trie.insert(word)
    return trie
