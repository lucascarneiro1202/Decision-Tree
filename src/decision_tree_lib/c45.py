import pandas as pd
from .base_tree import BaseDecisionTree, Node
from . import utils

class C45DecisionTree(BaseDecisionTree):
    """Árvore de Decisão usando o algoritmo C4.5."""

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series):
        """Encontra a melhor divisão usando a razão de ganho."""
        best_gain_ratio = -1
        best_split = {} # Dicionário para guardar a melhor divisão

        # TODO: Iterar sobre todos os atributos
        # Se o atributo for CATEGÓRICO:
        #   Calcular a razão de ganho (utils.gain_ratio)
        # Se o atributo for CONTÍNUO:
        #   Encontrar o melhor limiar (threshold) de divisão
        #   Calcular a razão de ganho para esse limiar
        #   Atualizar best_split se a razão de ganho for maior
        
        return best_split