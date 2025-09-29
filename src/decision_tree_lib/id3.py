import pandas as pd
from .base_tree import BaseDecisionTree, Node
from . import utils

'''
    Decision Tree using ID3 algorithm
    - Division Criteria: Information Gain
    - Attribute Type: Only categorical features
'''
class ID3DecisionTree(BaseDecisionTree):
    def __init__(self):
        super().__init__(self, max_depth=100, min_samples_split=2)
        self.criterion = 'entropy'

    def _calculate_impurity(self, y: pd.Series) -> float:
        return utils.entropy(y)

    '''
        Find the current data's best division using Information Gain

        Iterate over each attribute, calculate the information gain that would result from dividing by that attribute, 
        and returns the attribute that maximizes that gain

        @param X - DataFrame with all instances and all independent features
        @param y - Corresponding target labels
        @return best_split_information - Dictionary with the best division of samples in order to maximize the information gain
    '''
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_gain = -1
        best_feature = None
        
        # Iterate over all available columns (features) in the current dataset
        for feature in X.columns:
            
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

            # Calculate the information gain for the current division
            current_gain = utils.information_gain(y, subsets_y)
            
            # Compare the best informagion gain found and update it if it is bigger
            if current_gain > best_gain:
                best_gain = current_gain
                best_feature = feature
                
        # If no feature resulted in a positive gain, then there is no good division to be done
        if best_feature is None:
            return {}

        # Return the information about the best division found
        best_split_information = {
            'feature': best_feature,
            'gain': best_gain
        }
        
        return best_split_information