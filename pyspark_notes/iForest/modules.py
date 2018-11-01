# -*- coding: UTF-8 -*-

import math

class ITree:
    def __init__(self):
        self._left_tree = None
        self._right_tree = None
        self._type = 0
        self._size = 0

    def gettype(self):
        return self._type

    def getsize(self):
        return self._size

    def get_left_tree(self):
        return self._left_tree

    def get_right_tree(self):
        return self._right_tree

class ITreeBranch(ITree):
    def __init__(self, left, right, split_column, split_value):
        self._left_tree = left
        self._right_tree = right
        self._split_column = split_column
        self._split_value = split_value
        self._type = 2

    def get_split_column(self):
        return self._split_column

    def get_split_value(self):
        return self._split_value

class ITreeLeaf(ITree):
    def __init__(self, size):
        self._size = size
        self._type = 1

class IsolationForest:
    def __init__(self, num_samples, trees):
        self._num_samples = num_samples
        self._trees = trees

    def predict(self, x):
        column_num = len(x.__fields__)
        value_array = []
        for i in range(column_num):
            value_array.append(x[i])
        predictions = []
        for i in range(10):
            tree = self._trees[i]
            pre = pathlength(value_array, tree, 0)
            predictions.append(pre)
        score = pow(2, -(sum(predictions)/len(predictions))/cost(self._num_samples))
        return score

def cost(num_items):
    return 2 * (math.log(num_items - 1) + 0.5772156649) - (2 * (num_items - 1) / num_items)

def pathlength(x, tree, current_score):
    score = current_score
    if tree.gettype() == 1: # leaf
        tree_size = tree.getsize()
        if tree.getsize() > 1:
            score += cost(tree_size)
        else:
            score += 1
        # print(score)
    elif tree.gettype() == 2: # branch
        sample_value = x[tree.get_split_column()]
        if sample_value < tree.get_split_value():
             return pathlength(x, tree.get_left_tree(), score + 1)
        else:
             return pathlength(x, tree.get_right_tree(), score + 1)
    return score
