import pandas as pd
from .base_tree import BaseDecisionTree, Node
from . import utils

class CARTDecisionTree(BaseDecisionTree):
    """Árvore de Decisão usando o algoritmo CART."""

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series):
        """Encontra a melhor divisão binária usando o índice Gini."""
        best_gini_reduction = -1
        best_split = {}

        # TODO: Iterar sobre todos os atributos
        # Se o atributo for CONTÍNUO:
        #   Encontrar o melhor limiar que minimiza a impureza de Gini ponderada
        # Se o atributo for CATEGÓRICO:
        #   Encontrar a melhor partição binária dos valores que minimiza a impureza de Gini ponderada
        #   (Ex: {A, B} vs {C} para valores {A, B, C})
        
        return best_split