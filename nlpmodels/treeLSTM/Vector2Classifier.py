import torch
import torch.nn as nn

#
# Semantic relatedeness model from  https://arxiv.org/pdf/1503.00075.pdf
#


class Vector2Classifier(nn.Module):
    #Labels have to be numbers in range(numlabels)
    def __init__(self, vector_dim, numlabels, dropout=0.0, dtype=torch.float32):
        super(Vector2Classifier, self).__init__()
        if vector_dim < 1:
            raise ValueError("Hidden_dim cannot be 0 or negative")
        #
        self.vector_dim = vector_dim
        self.numlabels = numlabels
        self.labels = range(numlabels)
        self.dropout = dropout
        self.dtype = dtype
        self.rand_factor = torch.sqrt(torch.tensor(
            1.0/self.vector_dim, dtype=self.dtype))
        self.linear_layer = nn.Linear(2*self.vector_dim, self.numlabels)
        self.relu_layer = nn.ReLU()
        self.logprob_label = nn.LogSoftmax()
        self.loss = nn.NLLLoss(reduction="mean")

    #
    # h_1,2 [batch size, hidden dim]
    # target_scores [bath_size]
    # target_label_idx - list of class indices
    def forward(self, v_1, v_2, target_label_idx):
        batch_size = v_1.shape[0]
        #v_cross = v_1*v_2
        #v_plus = torch.abs(v_1-v_2)
        #v = torch.cat([v_cross, v_plus], axis=1)
        v = torch.cat([v_1, v_2], axis=1)
        logprob = self.logprob_label(self.relu_layer(self.linear_layer(v)))
        pred_label_idx = [idx.item() for idx in torch.argmax(logprob, axis=1)]
        crossentropy = self.loss(logprob, torch.tensor(target_label_idx))
        return logprob, pred_label_idx, crossentropy
