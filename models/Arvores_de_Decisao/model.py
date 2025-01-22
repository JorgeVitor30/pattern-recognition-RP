import numpy as np

class DecisionTreeGini:
    def __init__(self):
        self.tree = None
        self.max_depth = None
        self.min_samples_split = None

    def gini_index(self, groups, classes):
        n_instances = sum([len(group) for group in groups])
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                proportion = [row[-1] for row in group].count(class_val) / size
                score += proportion ** 2
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def best_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def terminal_node(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split_node(self, node, depth):
        left, right = node['groups']
        del(node['groups'])

        if not left or not right:
            node['left'] = node['right'] = self.terminal_node(left + right)
            return

        if depth >= self.max_depth:
            node['left'], node['right'] = self.terminal_node(left), self.terminal_node(right)
            return

        if len(left) <= self.min_samples_split:
            node['left'] = self.terminal_node(left)
        else:
            node['left'] = self.best_split(left)
            self.split_node(node['left'], depth + 1)

        if len(right) <= self.min_samples_split:
            node['right'] = self.terminal_node(right)
        else:
            node['right'] = self.best_split(right)
            self.split_node(node['right'], depth + 1)

    def fit(self, X, y, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        dataset = np.column_stack((X, y))
        self.tree = self.best_split(dataset)
        self.split_node(self.tree, 1)

    def predict(self, X):
        return np.array([self.predict_row(self.tree, row) for row in X])

    def predict_row(self, node, row):
        if isinstance(node, int):
            return node  

        if row[node['index']] < node['value']:
            return self.predict_row(node['left'], row)
        else:
            return self.predict_row(node['right'], row)