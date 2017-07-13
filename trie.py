## Moddified barebones trie for storing all kmers of a read
## implementation based on  https://nickstanisha.github.io/2015/11/21/a-good-trie-implementation-in-python.html
class Node:

    def __init__(self, label=None, data=None):
        self.label = label
        self.data = data
        self.children = dict()
        self.count = 0

    def add_child(self, key, data=None):
        if not isinstance(key, Node):
            self.children[key] = Node(key,data)
        else:
            self.children[key.label] = key

    def get_item(self, key):
        return self.children[key]

class Trie:
    def __init__(self):
        self.head = Node()

    def get_item(self, key):
        return self.head.children[key]

    def add(self, word):
        current_node = self.head
        word_finished = True

        for i in range(len(word)):
            if word[i] in current_node.children:
                current_node = current_node.children[word[i]]
            else:
                word_finished = False
                break

        if not word_finished:
            while i < len(word):
                current_node.addChild(word[i])
                current_node = current_node.children[word[i]]
                i += 1

        current_node.data = word
        current_node.count += 1

        def getCount(self, word):
            if not self.has_word(word):
                raise ValueError('{} not found in trie'.format(word))
            
            current_node = self.head
            for letter in word:
                current_node = current_node[letter]

            return current_node.count


