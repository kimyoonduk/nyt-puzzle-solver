class IntegerTrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class IntegerTrie:
    def __init__(self):
        self.root = IntegerTrieNode()

    def insert(self, elements):
        """Inserts a set of integers into the trie."""
        node = self.root
        for element in sorted(elements):  # Sort elements to maintain consistent order
            if element not in node.children:
                node.children[element] = IntegerTrieNode()
            node = node.children[element]
        node.is_end_of_word = True

    def search_subset(self, elements):
        """Checks if there is any subset of the given set in the trie."""
        elements = sorted(elements)  # Sort to use ordered elements for searching
        return self._search_recursive(self.root, elements, 0)

    def _search_recursive(self, node, elements, index):
        """Recursive helper function to search subsets in the trie."""
        if node.is_end_of_word:
            return True
        if index >= len(elements):
            return False
        # Try to find a deeper subset with the current element
        if elements[index] in node.children:
            if self._search_recursive(
                node.children[elements[index]], elements, index + 1
            ):
                return True
        # Skip the current element and try the next
        return self._search_recursive(node, elements, index + 1)


def build_integer_trie(set_list):
    """Utility function to build the trie from a list of integer sets."""
    trie = IntegerTrie()
    for elements in set_list:
        trie.insert(elements)
    return trie
