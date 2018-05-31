class Tree(object):
    def __init__(self, sw_flag=None, sw_id=None, sw_loc=None, dl_id=None, children=None):
        self.sw_flag = sw_flag
        self.sw_id = sw_id
        self.sw_loc = sw_loc
        self.dl_id = dl_id
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)


t = Tree('*','','','', [Tree('1','','',''),
               Tree('2','','',''),
               Tree('+','','','', [Tree('3','','','', [Tree('5','','',''), Tree('6','','','')]),
                          Tree('4','','','')])])


def treeSearch(tree):
    print(tree.sw_flag)
    children = tree.children
    if tree.children is not None:
        for child in children:
            treeSearch(child)


treeSearch(t)


