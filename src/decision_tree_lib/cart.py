import pandas as pd
from itertools import combinations
from .base_tree import BaseDecisionTree, Node
from . import utils

'''
    Decision Tree using CART algorithm
    - Division Criteria: Gini Impurity Reduction
    - Attribute Type: Strictly binary for all features
'''
class CARTDecisionTree(BaseDecisionTree):
    def __init__(self, max_depth=100, min_samples_split=2):
        super().__init__(max_depth, min_samples_split)
        self.criterion = 'gini'

    def _calculate_impurity(self, y: pd.Series) -> float:
        return utils.gini_impurity(y)
    
    '''
        Find the current data's best division using Gini Impurity Reduction

        Iterate over each attribute, treating them as categorical or continuous
        - For the categorial ones, find the best binary division of the values that maximize the Gini Impurity Reduction
        - For the continuous ones, find the best binarization threshold that maximixe the Gini Impurity Reduction

        @param X - DataFrame with all instances and all independent features
        @param y - Corresponding target labels
        @return best_split_information - Dictionary with the best division of samples in order to maximize the Gini Impurity Reduction
    '''
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_gini_reduction = -1
        best_split_information = {}
        n_samples = len(y)

        # Parent node's gini (befote the split) - it is used to calculate the Gini Impurity Reduction
        parent_gini = utils.gini_impurity(y)

        # Iterate over all available columns (attributes)
        for feature in X.columns:
            # If the feature is categorical
            if not pd.api.types.is_numeric_dtype(X[feature]):
                # Find the unique values of the feature
                unique_values = list(X[feature].unique())

                # If there is less than 2 unique values, then further comparison is useless
                if len(unique_values) < 2:
                    continue

                # Generate all possible binary divisions
                for i in range(1, len(unique_values) // 2 + 1):
                    # Calculate the combinations of unique values
                    for left_values_tuple in combinations(unique_values, i):
                        left_values = set(left_values_tuple)
                        
                        # Split data based on the current partition
                        left_mask = X[feature].isin(left_values)
                        y_left, y_right = y[left_mask], y[~left_mask]

                        # Calculate the Gini Impurity for each split
                        gini_left = utils.gini_impurity(y_left)
                        gini_right = utils.gini_impurity(y_right)

                        # Calculate the Weighed Gini Impurity for the division 
                        weighted_gini = (len(y_left) / n_samples) * gini_left + (len(y_right) / n_samples) * gini_right

                        # Calculate the Gini Impurity Reduction
                        current_reduction = parent_gini - weighted_gini

                        # Compare the best Gini Impurity Reduction found and update it if it is bigger
                        if current_reduction > best_gini_reduction:
                            best_gini_reduction = current_reduction
                            best_split_information = {
                                'feature': feature,
                                'left_values': left_values,  # Store the values set of the left branch
                                'gain': best_gini_reduction
                            }

            # If the feature is continuous
            else:
                # Sort the unique values to find the best candidates for a threshold
                unique_values = sorted(X[feature].unique())

                # If there is one unique value or less, then it is not need to calculate all options
                if len(unique_values) <= 1:
                    continue

                # Calculate the cut points (mean between two adjacent values)
                thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]

                # Test all thresholsd in order to find the best one for this feature
                for threshold in thresholds:
                    # Split the data in two subsets based on the current threshold
                    left_mask = X[feature] <= threshold
                    y_left, y_right = y[left_mask], y[~left_mask]

                    # Calculate the Gini Impurity for each split
                    gini_left = utils.gini_impurity(y_left)
                    gini_right = utils.gini_impurity(y_right)

                    # Calculate the Weighed Gini Impurity for the division 
                    weighted_gini = (len(y_left) / n_samples) * gini_left + (len(y_right) / n_samples) * gini_right
                    
                    # Calculate the Gini Impurity Reduction
                    current_reduction = parent_gini - weighted_gini

                    # Compare the best Gini Impurity Reduction found and update it if it is bigger
                    if current_reduction > best_gini_reduction:
                        best_gini_reduction = current_reduction
                        best_split_information = {
                            'feature': feature,
                            'threshold': threshold,
                            'gain': best_gini_reduction
                        }
                            
        return best_split_information