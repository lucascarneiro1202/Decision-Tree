import numpy as np
import pandas as pd
import math
from collections import Counter

'''
    Calculate the entropy of a given label set
    @param y - Series of all instances' labels for a certain feature
    @return entropy_value
'''
def entropy(y: pd.Series) -> float:
    # Calculate the number of occurrences of each distinct value of a series
    value_counts = y.value_counts()

    # Encounter the total number of elements of a series
    total_elements = len(y)

    # Calcualte the probability of each distinct value of a series
    probabilities = value_counts / total_elements

    # Apply the entropy formula
    entropy_value = 0
    for p in probabilities:
        if p > 0:  # Avoid log(0)
            entropy_value -= p * math.log(p, 2) # Base 2 for bits

    return entropy_value

'''
    Calculate the Gini Impurity of a given label set
    @param y - Series of all instances' labels for a certain feature
    @return gini_impurity_value
'''
def gini_impurity(y: pd.Series) -> float:
    # Calculate the number of occurrences of each distinct value of a series
    value_counts = y.value_counts()

    # Encounter the total number of elements of a series
    total_elements = len(y)

    # Calculate the probability of each distinct value of a series
    probabilities = value_counts / total_elements

    # Apply the Gini Impurity formula
    probabilities_sum_square = (probabilities ** 2).sum()
    gini_impurity_value = 1 - probabilities_sum_square

    return gini_impurity_value

'''
    Calculate the information gain of a given label set
    @param y - Series of all instances' labels for a certain feature
    @param subsets - List of series in which each series contains the labels of the subset related to one possible value of a certain feature
    @return info_gain_value
'''
def information_gain(y: pd.Series, subsets: list[pd.Series]) -> float:
    # Calculate the entropy of all instances' labels for a certain feature
    parent_entropy = entropy(y)
    
    # Calculate the wheighed mean of each subset related to one possible value of a certain feature
    total_n = len(y)
    if total_n == 0:
        return 0

    weighted_child_entropy = 0
    for subset in subsets:
        n_subset = len(subset)
        if n_subset == 0:
            continue
            
        # Subset weight
        weight = n_subset / total_n
        
        # Add the child's weighed entropy to the sum
        weighted_child_entropy += weight * entropy(subset)
        
    # Apply the Information Gain formula
    info_gain_value = parent_entropy - weighted_child_entropy
    
    return info_gain_value

'''
    Calculate the gain ratio of a given label set
    @param y - Series of all instances' labels for a certain feature
    @param subsets - List of series in which each series contains the labels of the subset related to one possible value of a certain feature
'''
def gain_ratio(y: pd.Series, subsets: list[pd.Series]) -> float:
    # Calculate the information gain of a given label set
    info_gain_value = information_gain(y, subsets)
    
    # If the information gain is 0, its gain ratio will also be 0
    if info_gain_value == 0:
        return 0
        
    # Calculate the Split Info of the subsets
    total_n = len(y)
    split_info = 0
    for subset in subsets:
        n_subset = len(subset)
        if n_subset == 0:
            continue
        
        proportion = n_subset / total_n
        split_info -= proportion * math.log2(proportion)
        
    # If spit info is 0, then all labels are in one branch
    if split_info == 0:
        return 0
        
    # Apply the Gain Ratio formula
    gain_ratio_value = info_gain_value / split_info

    return gain_ratio_value