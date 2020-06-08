import torch
import torch.nn as nn

#
# Semantic relatedeness model from  https://arxiv.org/pdf/1503.00075.pdf
#
class SRFeedForward1L(nn.Module):
    #By default, the SICK scores, 1 to 5
    def __init__(self, hidden_dim, dropout=0.0, dtype=torch.float32, scores=list(range(1,6))):
        super(SRFeedForward1L,self).__init__()
        if hidden_dim < 1:
            raise ValueError("Hidden_dim cannot be 0 or negative")
        #
        self.hidden_dim=hidden_dim
        self.num_scores=len(scores)
        self.dropout=dropout
        self.dtype = dtype
        self.rand_factor=torch.sqrt(torch.tensor(1.0/self.hidden_dim, dtype= self.dtype))
        self.score_dict=dict([(s,i) for i, s in enumerate(scores)])

        self.W_cross=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim, self.hidden_dim,dtype=self.dtype)-self.rand_factor,requires_grad=True)
        self.W_plus=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim, self.hidden_dim,dtype=self.dtype)-self.rand_factor,requires_grad=True)
        self.b_h=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,dtype=self.dtype)-self.rand_factor,requires_grad=True)
        self.W_p=nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_scores, self.hidden_dim,dtype=self.dtype)-self.rand_factor,requires_grad=True)
        self.b_p=nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_scores,dtype=self.dtype)-self.rand_factor,requires_grad=True)
        self.r=torch.tensor(scores,dtype=dtype)
        self.KLDivergence=nn.KLDivLoss(reduction="batchmean")

    #
    # h_1,2 [batch size, hidden dim]
    # target_scores [bath_size]
    #
    def forward(self, h_1, h_2, target_scores):
        batch_size=h_1.shape[0]
        h_cross = h_1*h_2
        h_plus=torch.abs(h_1-h_2)
        #Product on a batched vector
        h_s=torch.sigmoid(torch.einsum('ij,bj->bi',(self.W_cross, h_cross))+torch.einsum('ij,bj->bi',(self.W_plus, h_plus)) + torch.stack([self.b_h for b in range(batch_size)],axis=0))
        # [batch_size, K] logsoftmax in dim 1 (so normalization is on the p for every k)
        p_theta=torch.log_softmax(torch.einsum('ij,bj->bi',(self.W_p,h_s))+torch.stack([self.b_p for b in range(batch_size)],axis=0),dim=1)
        y=torch.einsum('i,bi->b',self.r,torch.exp(p_theta)) #predicted score
        # p_theta - loprobabilities
        # p_target - Probability distribution
        # target -scores: 1....5
        floor_target_scores_plus_one=[(idx,self.score_dict[int(f.item()) + 1]) for idx, f in enumerate(torch.floor(target_scores)) if int(f.item()) + 1 < self.num_scores]
        floor_target_scores=[(idx,self.score_dict[int(f.item())]) for idx, f in enumerate(torch.floor(target_scores)) if int(f.item()) < self.num_scores]
        p_target=torch.zeros(batch_size, self.num_scores)
        bidx_plus_one=[p[0] for p in floor_target_scores_plus_one]
        bidx=[p[0] for p in floor_target_scores]
        p_target[bidx_plus_one,[p[1] for p in floor_target_scores_plus_one]]=target_scores[bidx_plus_one]-torch.floor(target_scores[bidx_plus_one])
        p_target[bidx,[p[1] for p in floor_target_scores]]=torch.floor(target_scores[bidx]) - target_scores[bidx] + torch.ones(len(bidx),dtype=self.dtype)
        #First parameter (p_theta) must be a log probability
        loss=self.KLDivergence(p_theta, p_target) ## p_target*log p_target -p_theta  = KL(target||p_theta) by definition
        return loss, y # Average_on_minibatch(KL(target||p_theta)). Use a cuadratic regularization when optimizing