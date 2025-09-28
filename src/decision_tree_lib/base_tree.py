import numpy as np
import pandas as pd
from collections import Counter

'''
    Class to represent a Node within a Decision Tree
'''
class Node:
    def __init__(self, feature=None, threshold=None, branches=None, value=None, is_leaf=False):
        self.feature = feature       # Attribute used for comparison
        self.threshold = threshold   # Threshold to continuous values
        self.branches = branches     # Dictionary of sub-trees (branches) for categorical values
        self.value = value           # Class value, if it is a leaf
        self.is_leaf = is_leaf

'''
    Generic class to represent all Decision Tree algorithms
'''
class BaseDecisionTree:
    '''
        Class constructor
        @param max_depth - Maximum depth of the Decision Tree
        @param min_samples_split - Minimum number of samples required to split an internal node
    '''
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = Node()

    '''
        Start the tree construction
        @param X - Training features
        @param y - Corresponding target labels

    '''
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.root = self._build_tree(X, y)

    '''
        Recursive method to build the Decision Tree
        @param X - DataFrame with all instances and all independent features
        @param y - Corresponding target labels
        @param depth - Current depth of each recursion
        @return None - Root of the built sub-tree
    '''
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # If any of the following conditions are true, a leaf node is created and the recursion in this branch is interrupted 

        # Condition 1: The node is pure (all samples belong to the same class)
        if n_labels == 1:
            leaf_value = y.iloc[0]
            return Node(value=leaf_value, is_leaf=True)

        # Condition 2: The maximum depth was reached
        if depth >= self.max_depth:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True)

        # Condition 3: The node's number os samples is smaller then the min_samples_split
        if n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True)

        # Find the best split with the abstract method implemented by the child classes (ID3, C4.5 and CART)
        best_split = self._find_best_split(X, y)

        # If there is no division that enhances the model, the recursion is interrupted in this branch
        if not best_split or best_split.get('gain', -1) <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True)
            
        # If there is a good division, the Decision Tree continues to be created
        feature_name = best_split["feature"]
        branches = {}

        # If the division is for a categorical feature
        if "threshold" not in best_split:
            # For each unique value of the chosen feature, a branch is created
            for value in X[feature_name].unique():
                # Filter the data to create the branch's subset
                subset_mask = X[feature_name] == value
                X_subset, y_subset = X[subset_mask], y[subset_mask]

                # If the subset is empty, a leaf node is created with the majority class of the father
                if X_subset.empty:
                    branches[value] = Node(value=self._most_common_label(y), is_leaf=True)
                else:
                    # Driver of recursion to create the sub-tree
                    branches[value] = self._build_tree(X_subset, y_subset, depth + 1)
            
            # Return a decision node with categorical branches
            return Node(feature=feature_name, branches=branches)

        # If the division is for a continuous feature
        else:
            # Identify the encountered threshold
            threshold = best_split["threshold"]

            # Divide the data in two subsets: left (<= threshold) and right (> threshold)
            left_mask = X[feature_name] <= threshold
            X_left, y_left = X[left_mask], y[left_mask]

            right_mask = X[feature_name] > threshold
            X_right, y_right = X[right_mask], y[right_mask]

            # Driver of recursion to create the left branch
            left_child = self._build_tree(X_left, y_left, depth + 1)

            # Driver of recursion to create the right branch
            right_child = self._build_tree(X_right, y_right, depth + 1)
            
            # Store both children in a dictionary for consistency
            branches = {"left": left_child, "right": right_child}

            # Return a decision node with the threshold information and the binary branches
            return Node(feature=feature_name, threshold=threshold, branches=branches)

    '''
        Find the current data's best division using different criteria

        Iterate over each attribute, calculate the criteria that would result from dividing by that attribute, 
        and returns the attribute that maximizes that criteria

        @param X - DataFrame with all instances and all independent features
        @param y - Corresponding target labels
        @return best_split_info - Dictionary with the best division of samples in order to maximize the criteria
    '''
    def _find_best_split(self, X, y):
        raise NotImplementedError("This method must be implemented by the subclass.")

    '''
        Make predictions for a novel datasaet
        @param X - Datasets with independent features to be predicted
    '''
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for _, x in X.iterrows()])

    '''
        Traverse through the Decision Tree to classify one sample
        @param x - Unique sample (row) of DataFrame, represented as a series
        @return O valor da classe predita quando um nó folha é alcançado.
    '''
    def _traverse_tree(self, x: pd.Series, node: Node):
        # If the current node is a leaf node, then it signals the end of the trasversing
        if node.is_leaf:
            return node.value

        # Idenfify the feature value used to divide the data
        feature_value = x[node.feature]

        # If the division is categorical (based on unique values)
        if node.threshold is None:
            # The current feature value is used as a 'key' to find the next branch in the node's branch dictionary
            next_node = node.branches.get(feature_value)

            # If the sample 'x' has a catagorical value that do not exists in the train dataset, return None
            if next_node is None:
                return None
            
            # If a branch was found, the trasverse continue through it
            return self._traverse_tree(x, next_node)

        # If the division is continuous (based on a threshold)
        else:
            # Compate the current sample value with the node's threshold
            if feature_value <= node.threshold:
                # If it is smaller or equal, the left branch is followed
                return self._traverse_tree(x, node.branches['left'])
            else:
                # If it is bigger, the right branch is followed
                return self._traverse_tree(x, node.branches['right'])

    '''
        Find the most common label predicted when the Decision Tree can't be divided anymore
        @param y - Predictions of a certain dataset
        @return str - Most common label
    '''
    def _most_common_label(self, y: pd.Series):
        counter = Counter(y)
        return counter.most_common(1)[0][0]