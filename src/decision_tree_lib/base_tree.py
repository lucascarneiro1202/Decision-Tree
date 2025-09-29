import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

'''
    Try to import dependency to print a Decision Tree
'''
try:
    import graphviz
except ImportError:
    print("Graphviz nnot found. To use graph visualization, install it with: pip install graphviz")
    graphviz = None

'''
    Class to represent a Node within a Decision Tree
'''
class Node:
    def __init__(self, feature=None, threshold=None, left_values=None,branches=None, value=None, majority_class=None, is_leaf=False, samples=None, class_dist=None, criterion=None):
        self.feature = feature               # Attribute used for comparison
        self.threshold = threshold           # Threshold to continuous values
        self.left_values = left_values       # Dictionary of left and right sub-trees (for CART)
        self.branches = branches             # Dictionary of sub-trees (branches) for categorical values
        self.value = value                   # Class value, if it is a leaf
        self.majority_class = majority_class # Majority class present in the node
        self.is_leaf = is_leaf
        self.samples = samples               # Number on samples of the node
        self.class_dist = class_dist         # Class distribution (ex: {0: 80, 1: 20})
        self.criterion = criterion           # Criterio of node comparison

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
        self.classes_ = sorted(y.unique())
        self.n_classes_ = len(self.classes_)
        self.root = self._build_tree(X, y)

    '''
        Recursive method to build the Decision Tree
        @param X - DataFrame with all instances and all independent features
        @param y - Corresponding target labels
        @param depth - Current depth of each recursion
        @return Node - Root of the built sub-tree
    '''
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        n_samples = len(y)
        if n_samples == 0:
            return None 
            
        n_labels = len(y.unique())
        class_dist = y.value_counts().to_dict()
        maj_class = self._most_common_label(y)
        current_impurity = self._calculate_impurity(y)

        # If any of the following conditions are true, a leaf node is created and the recursion in this branch is interrupted 

        # Condition 1: The node is pure (all samples belong to the same class)
        if n_labels == 1:
            leaf_value = y.iloc[0]
            return Node(value=leaf_value, is_leaf=True, samples=n_samples, class_dist=class_dist, impurity=current_impurity)

        # Condition 2: The maximum depth was reached
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True, samples=n_samples, class_dist=class_dist, impurity=current_impurity)

        # Condition 3: The node's number os samples is smaller then the min_samples_split
        if self.min_samples_split is not None and n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True, samples=n_samples, class_dist=class_dist, impurity=current_impurity)

        # Find the best split with the abstract method implemented by the child classes (ID3, C4.5 and CART)
        best_split = self._find_best_split(X, y)

        # If there is no division that enhances the model, the recursion is interrupted in this branch
        if not best_split or best_split.get('gain', -1) <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True, samples=n_samples, class_dist=class_dist, impurity=current_impurity)
            
        # If there is a good division, the Decision Tree continues to be created
        feature_name = best_split["feature"]
        branches = {}

        # If the division is for a continuous feature
        if "threshold" in best_split:
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
            return Node(feature=feature_name, threshold=threshold, branches=branches, majority_class=maj_class, samples=n_samples, class_dist=class_dist, impurity=current_impurity)
        
        # If the division is for a categorical feature with only two values (CART)  
        elif "left_values" in best_split:
            # Identify the values from the left branch
            left_values = best_split["left_values"]

            # Create a mask to find instances with the left values identified
            left_mask = X[feature_name].isin(left_values)

            # Filter dataset based on the values identified
            X_left, y_left = X[left_mask], y[left_mask]

            # Create a mask to find instances with the right values identified
            right_mask = ~left_mask

            # Filter dataset based on the values identified
            X_right, y_right = X[right_mask], y[right_mask]

            # Driver of recursion for both branches
            left_child = self._build_tree(X_left, y_left, depth + 1)
            right_child = self._build_tree(X_right, y_right, depth + 1)

            branches = {"left": left_child, "right": right_child}

            return Node(feature=feature_name, left_values=left_values, branches=branches, majority_class=maj_class, samples=n_samples, class_dist=class_dist, impurity=current_impurity)
                    
        # If the division is for a categorical feature with multiple values (ID3 and C4.5)
        else:
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
            return Node(feature=feature_name, branches=branches, majority_class=maj_class, samples=n_samples, class_dist=class_dist, impurity=current_impurity)

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

        # If the division is continuous (based on a threshold)
        if node.threshold is not None:
            if feature_value <= node.threshold:
                return self._traverse_tree(x, node.branches['left'])
            else:
                return self._traverse_tree(x, node.branches['right'])
            
        # If the division is categorical with two values (CART)
        elif node.left_values is not None:
            if feature_value in node.left_values:
                return self._traverse_tree(x, node.branches['left'])
            else:
                return self._traverse_tree(x, node.branches['right'])

        # If the division is categorical with multiple values (ID3 and C4.5)
        else:
            next_node = node.branches.get(feature_value)
            if next_node is None:
                return node.majority_class
            return self._traverse_tree(x, next_node)

    '''
        Find the most common label predicted when the Decision Tree can't be divided anymore
        @param y - Predictions of a certain dataset
        @return str - Most common label
    '''
    def _most_common_label(self, y: pd.Series):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    '''
        Start printing a Decision Tree from the root
    '''
    def print_tree(self):
        if not self.root:
            print("Decision tree was still not fitted.")
            return
        
        # Drive of recursion
        self._print_recursive(self.root)

    '''
        Auxiliary function to print a Decision Tree recursively
        @param node - Node to be printed
    '''
    def _print_recursive(self, node, prefix=""):
        if node is None:
            return

        # If the node is a leaf node, print only its predicted value
        if node.is_leaf:
            print(f"{prefix}└── Predict: {node.value}")
            return

        # If the division is continuous or categorical with only two values (CART)
        if node.threshold is not None or node.left_values is not None:            
            # Logic fot the left branch

            # If the division is continuous
            if node.threshold is not None:
                rule_left = f"IF {node.feature} <= {node.threshold:.2f}"
            
            # If the division if categorical with only two values (CART)
            else:
                rule_left = f"IF {node.feature} in {node.left_values}"
            
            # Print the left branch
            print(f"{prefix}├── {rule_left}")
            self._print_recursive(node.branches['left'], prefix + "│   ")

            # Logic for the right branch

            # If the division is continuous
            if node.threshold is not None: 
                rule_right = f"ELSE ({node.feature} > {node.threshold:.2f})"

            # If the division if categorical with only two values (CART)
            else:
                rule_right = f"ELSE (not in {node.left_values})"

            # Print the right branch
            print(f"{prefix}└── {rule_right}")
            self._print_recursive(node.branches['right'], prefix + "│   ")

        # If the division if categorical with more than two branches (ID3 and C4.5)
        else:
            print(f"{prefix}├── IF {node.feature} is:")
            
            # Iterate over each branch (value) of the division
            branches = list(node.branches.items())
            for i, (value, child_node) in enumerate(branches):
                is_last = (i == len(branches) - 1)
                connector = "└──" if is_last else "├──"
                
                # Print the node
                print(f"{prefix}│   {connector} {value}")
                new_prefix = prefix + "    " if is_last else prefix + "│   "
                self._print_recursive(child_node, new_prefix)

    '''
        Generate a Decision Tree visualization using Graphviz library
        @param - orientation: 'TB' (Top-to-Bottom, standard) or 'LR' (Left-to-Right)
        @return dot - graphviz.Digraph object that can be rendered within a notebook
    '''
    def export_graphviz(self, orientation='TB'):
        if graphviz is None:
            print("Graphviz is not installed. It is not possible to generate visualization.")
            return

        if not self.root:
            print("The Decision Tree was not fitted.")
            return
        
        # Use a color map to show different colors
        cmap = plt.get_cmap('rainbow', self.n_classes_)
        colors_hex = [mcolors.to_hex(cmap(i)) for i in range(self.n_classes_)]

        dot = graphviz.Digraph(comment='Decision Tree', graph_attr={'rankdir': orientation}, format='png')
        
        dot.attr('node', shape='box', style='filled, rounded', fontname='helvetica')
        dot.attr('edge', fontname='helvetica')

        self._add_nodes_edges(self.root, dot, colors=colors_hex)
        return dot

    '''
        Auxiliary function to add vexes and edged to the graph recursively
        @param node - Current node of a Decision Tree
        @param dot - graphviz.Digraph object that can be rendered within a notebook
        @param colors - List of colors to fill the nodes
        @param parent_id - Unique ID for each node
        @param branch_label - Label printed as a branch label
    '''
    def _add_nodes_edges(self, node, dot, colors, parent_id=None, branch_label=""):
        # If the node is not valid, interrupt the recursion in this branch
        if node is None:
            return

        # Create a unique ID for each node to avoid collision
        node_id = id(node)

        # Create a node label
        if node.is_leaf:    
            label = (f"value = {node.class_dist}\nsamples = {node.samples}\nclass = {node.value}")
        else:
            if node.threshold is not None:
                rule = f"{node.feature} <= {node.threshold:.2f}" # For continuous attributes, the feature is a threshold
            else:
                rule = node.feature # For ID3/C4.5, the rule is the attribute in itself

            label = (f"{rule}\n"
                     f"{self.criterion} = {node.impurity:.3f}\n"
                     f"samples = {node.samples}\n"
                     f"value = {node.class_dist}\n"
                     f"class = {node.majority_class}")

        majority_class = node.majority_class if not node.is_leaf else node.value
        
        # Find majority class' index in the model's list of classes
        if majority_class in self.classes_:
            class_index = self.classes_.index(majority_class)
            node_color = colors[class_index]
        else:
            node_color = '#FFFFFF' # Standard color (white) if something goes wrong

        # Create new vex
        dot.node(str(node_id), label, fillcolor=node_color)

        # Create new edge, if parent exists
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id), label=str(branch_label))

        if not node.is_leaf:
            # If the division is binary (continuous or categorical from CART)
            if node.threshold is not None or node.left_values is not None:
                # Labels for continuous split
                if node.threshold is not None:
                    label_left = f"<= {node.threshold:.2f}"
                    label_right = f"> {node.threshold:.2f}"
                # Labels for categorical split (CART)
                else: 
                    # Convert the set of values ​​to a string
                    values_str = str(node.left_values)

                    # Truncate the string if it is too long
                    if len(values_str) > 20:
                        values_str = values_str[:17] + "..."
                    
                    label_left = f"in {values_str}"
                    label_right = f"not in {values_str}"
                
                self._add_nodes_edges(node.branches['left'], dot, colors, node_id, label_left)
                self._add_nodes_edges(node.branches['right'], dot, node_id, label_right)
            
            # If the division is categorical with multi-values (ID3 and C4.5)
            else:
                for value, child_node in node.branches.items():
                    self._add_nodes_edges(child_node, dot, colors, node_id, value)