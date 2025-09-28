import pandas as pd
from .base_tree import BaseDecisionTree, Node
from . import utils

class ID3DecisionTree(BaseDecisionTree):
    """Árvore de Decisão usando o algoritmo ID3."""

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series):
        """Encontra a melhor divisão usando o ganho de informação."""
        best_gain = -1
        best_feature = None

        # TODO: Iterar sobre todas as colunas (atributos) de X
        # Para cada atributo, calcular o ganho de informação (usando utils.information_gain)
        # Armazenar o atributo com o maior ganho
        
        return best_feature