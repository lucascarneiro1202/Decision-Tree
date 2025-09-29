import pandas as pd
import numpy as np
from .base_tree import BaseDecisionTree, Node
from . import utils

'''
    Decision Tree using C4.5 algorithm
    - Division Criteria: Gain Ratio
    - Attribute Type: Categorical and continuous features
'''
class C45DecisionTree(BaseDecisionTree):
    def __init__(self, max_depth=100, min_samples_split=2):
        super().__init__(self, max_depth=100, min_samples_split=2)
        self.criterion = 'gain-ratio'

    def _calculate_gain_ratio(self, y: pd.Series) -> float:
        return utils.gain_ratio(y)
    
    '''
        Find the current data's best division using Gain Ratio

        Iterate over each attribute, treating them as categorical or continuous
        - For the categorial ones, calculate the gain ratio that would result from dividing by that attribute, 
        and returns the attribute that maximizes that gain
        - For the continuous ones, find the best binarization threshold that maximizes the Gain Ratio

        @param X - DataFrame with all instances and all independent features
        @param y - Corresponding target labels
        @return best_split_information - Dictionary with the best division of samples in order to maximize the gain ratio
    '''
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_gain_ratio = -1
        best_split_information = {}

        # Iterate over all available columns (features)
        for feature in X.columns:
            # If the feature is categorical
            if not pd.api.types.is_numeric_dtype(X[feature]):

                # For the current feature, the label set 'y' is divided in subsets, one for each unique value of this feature
                subsets_y = []
                
                # Obtain the unique values for the current feature
                unique_values = X[feature].unique()
                
                # Create a label subset for each unique value
                for value in unique_values:
                    # Find indexes where the feature has this value
                    mask = X[feature] == value

                    # Use this indexes to filter 'y' labels
                    y_subset = y[mask]

                    # Append this label subset to the list of subsets of the feature
                    subsets_y.append(y_subset)
                
                # If the division result in only one subset, then it is useless
                if len(subsets_y) <= 1:
                    continue

                # Calculate the gain ratio for the current division
                current_gain_ratio = utils.gain_ratio(y, subsets_y)

                # Compare the best gain ratio found and update it if it is bigger
                if current_gain_ratio > best_gain_ratio:
                    best_gain_ratio = current_gain_ratio
                    best_split_information = {
                        'feature': feature,
                        'gain': current_gain_ratio
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
                    right_mask = X[feature] > threshold
                    
                    subsets_y = [y[left_mask], y[right_mask]]
                    
                    # Calculate the gain ratio for this binary division
                    current_gain_ratio = utils.gain_ratio(y, subsets_y)

                    # Compare the best gain ratio found (of all features and thresholds) and update it if it is bigger
                    if current_gain_ratio > best_gain_ratio:
                        best_gain_ratio = current_gain_ratio
                        best_split_information = {
                            'feature': feature,
                            'threshold': threshold,
                            'gain': current_gain_ratio
                        }
        
        return best_split_information