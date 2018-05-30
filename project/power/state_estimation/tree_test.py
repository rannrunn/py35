class Tree(object):
    "Generic Tree Node."
    def __init__(self, name='temp', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                print(child)
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)


t = Tree('*', [Tree('1'),
               Tree('2'),
               Tree('+', [Tree('3', [Tree('5'), Tree('6')]),
                          Tree('4')])])


def treeSearch(tree):
    print(tree.name)
    children = tree.children
    if tree.children is not None:
        for child in children:
            treeSearch(child)


treeSearch(t)


