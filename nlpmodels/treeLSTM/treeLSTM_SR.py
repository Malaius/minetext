import torch 
import torch.nn as nn
from treeLSTM import treeLSTM
from SRFeedForward1L import SRFeedForward1L


class treeLSTM_SR(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, dropout=0.1, num_layers=1, dtype=torch.float32, scores=list(range(1,6)), attention=False, generate_query=False):
        super(treeLSTM_SR,self).__init__()
        self.generate_query=generate_query
        self.tree_lstm = treeLSTM(
            input_dim, hidden_dim, attention_dim, dropout, num_layers, dtype, attention, generate_query)
        self.SRFF=SRFeedForward1L(hidden_dim, dropout,dtype,scores)

    #Scores are expected to be Python float
    def forward(self, trees_A, trees_B, scores, hiddens_A=None, hiddens_B=None, cells_A = None, cells_B = None ):
        if self.generate_query:
            query_A=self.tree_lstm.generate_batch_query(trees_A)
            query_B=self.tree_lstm.generate_batch_query(trees_B)
        else:
            query_A=None
            query_B=None
        updated_trees_A, root_hidden_A, root_cell_A= self.tree_lstm(
            trees_A, query_B)
        updated_trees_B, root_hidden_B, root_cell_B = self.tree_lstm(
            trees_B, query_A)
        loss, rank=self.SRFF(root_hidden_A, root_hidden_B, torch.tensor(scores))
        return loss, rank, root_hidden_A, root_hidden_B, root_cell_A, root_cell_B
    #
    # Expect tensors [batch_size]
    #
    def pearson_corr_loss(self, scores_true, scores_pred):
        #
        # Pearson correlation coefficient
        #
        batch_size=scores_true.shape[0]
        true_mean=torch.mean(scores_true)*torch.ones(batch_size)
        pred_mean=torch.mean(scores_pred)*torch.ones(batch_size)
        d_true_pred=torch.matmul((scores_true -true_mean),(scores_pred - pred_mean))
        d_true = torch.matmul((scores_true -true_mean),(scores_true - true_mean))
        d_pred=torch.matmul((scores_pred -pred_mean),(scores_pred - pred_mean))
        return d_true_pred/(torch.sqrt(d_pred)*torch.sqrt(d_true))