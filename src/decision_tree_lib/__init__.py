'''
    Biblioteca de Árvore de Decisão: Implementações dos algoritmos ID3, C4.5 e CART
'''
from .id3 import ID3DecisionTree
from .c45 import C45DecisionTree
from .cart import CARTDecisionTree

__all__ = ['ID3DecisionTree', 'C45DecisionTree', 'CARTDecisionTree']