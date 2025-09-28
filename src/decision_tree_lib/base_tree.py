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
        @param X - Training features
        @param y - Corresponding target labels
        @param depth - Current depth of each recursion
    '''
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # TODO: Implementar critérios de parada (base da recursão)
        # 1. Se todos os exemplos pertencem à mesma classe
        # 2. Se a profundidade máxima foi atingida
        # 3. Se o número de amostras for menor que min_samples_split
        # 4. Se não houver mais atributos para dividir

        # TODO: Encontrar a melhor divisão (lógica específica de cada subclasse)
        best_split = self._find_best_split(X, y)

        # TODO: Se a melhor divisão não oferecer ganho, criar um nó folha
        
        # TODO: Criar a sub-árvore recursivamente
        pass

    def _find_best_split(self, X, y):
        """Encontra a melhor divisão (a ser implementada pelas subclasses)."""
        raise NotImplementedError("Este método deve ser implementado pela subclasse.")

    '''
        Make predictions for a novel datasaet
        @param X - Datasets with independent features to be predicted
    '''
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for _, x in X.iterrows()])

    '''
        Traverse through the Decision Tree to classify one sample
        @param x - 
    '''
    def _traverse_tree(self, x: pd.Series, node: Node):
        """Navega na árvore para classificar uma única amostra."""
        if node.is_leaf:
            return node.value
        
        # TODO: Implementar a lógica de navegação
        # Se o nó divide por atributo contínuo (usar threshold)
        # Se o nó divide por atributo categórico (usar branches)
        pass

    '''
        Find the most common label predicted
        @param y - Predictions of a certain dataset
        @return str - Most common label
    '''
    def _most_common_label(self, y: pd.Series):
        """Retorna o rótulo mais comum em um conjunto."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]