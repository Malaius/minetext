import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from treeLSTM import treeLSTM
from Vector2Classifier import Vector2Classifier

#
# Torch module to apply a sentence encoder (treeLSTM)  to a pair of dependency trees and use the output is
# a classifier (Vector2Classifier)
#


class treeLSTM_pairclassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.0, num_layers=1, dtype=torch.float32, labels=list(range(1, 6)), combination_mode="sum"):
        super(treeLSTM_pairclassifier, self).__init__()
        self.tree_lstm = treeLSTM(
            input_dim, hidden_dim, dropout, num_layers, dtype, combination_mode)
        self.label_idx = dict([(label, idx)
                               for idx, label in enumerate(labels)])
        self.classifier = Vector2Classifier(hidden_dim, len(labels), dtype)

    #Scores are expected to be Python float
    def forward(self, trees_A, trees_B, list_of_labels, hiddens_A=None, hiddens_B=None, cells_A=None, cells_B=None):
        updated_trees_A, root_hidden_A, root_cell_A = self.tree_lstm(
            trees_A, hiddens_A, cells_A)
        updated_trees_B, root_hidden_B, root_cell_B = self.tree_lstm(
            trees_B, hiddens_B, cells_B)
        target_idx = [self.label_idx[l] for l in list_of_labels]
        logprob, pred_label_idx, crossentropy = self.classifier(
            root_hidden_A, root_hidden_B, target_idx)
        prec, recall, f1, support = precision_recall_fscore_support(
            target_idx, pred_label_idx, labels=None)
        return crossentropy, logprob, prec, recall, f1, support
