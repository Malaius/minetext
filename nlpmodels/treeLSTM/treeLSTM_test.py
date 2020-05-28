import unittest
from treeLSTM import treeLSTM
import torch

class treeLSTM_test(unittest.TestCase):
    def setUp(self):
        self.model_att_sentence=treeLSTM(50,25,20,dropout=0,num_layers=1, dtype=torch.float32, attention=True, generate_query=True)

    def test_attention_vector(self):
        #9 hidden vectors
        #
        with torch.no_grad():
            hiddens=[torch.rand(25) for i in range(9)]
            #
            M_q=torch.rand(25)
            M_k=torch.stack(hiddens,axis=1)
            query=torch.mv(self.model_att_sentence.W_query, M_q) # [att_dim, hidden_dim] x [hidden_dim,#children] ->[att_dim,#children]
            key=torch.mm(self.model_att_sentence.W_key, M_k) #[att_dim,#children]
            #[#children, #children]
            align=torch.mv(torch.transpose(key,0,1),query)/torch.sqrt(torch.tensor(25., dtype=torch.float32))
            #
            # Each row (dim=0) is the attention probability for every children according to the owrd in that row
            # Normalize on columns
            alpha=torch.nn.functional.softmax(align, dim=0) ##[#children, #children]
            pre_h=torch.mv(M_k,alpha) #Child hidden vectors weighted by their attention#[hidden_dim, #children]
            h=torch.tanh(torch.mv(self.model_att_sentence.W_att,pre_h) + self.model_att_sentence.b_att)
            #
            h_model=self.model_att_sentence._attention(hiddens, M_q)

            
            result= torch.sum(torch.abs(h - h_model)).item()
            self.assertEqual(result < 0.00001,True)

    def test_attention_matrix(self):
        #9 hidden vectors
        #
        with torch.no_grad():
            hiddens=[torch.rand(25) for i in range(9)]
            #
            M_q=torch.rand(25,9)
            M_k=torch.stack(hiddens,axis=1)
            query=torch.mm(self.model_att_sentence.W_query, M_q)
            key=torch.mm(self.model_att_sentence.W_key, M_k)
            align=torch.mm(torch.transpose(query,0,1),key)/torch.sqrt(torch.tensor(25., dtype=torch.float32))
            #
            # Each row (dim=0) is the attention probability for every children according to the owrd in that row
            # Normalize on columns
            alpha=torch.nn.functional.softmax(align, dim=0) #
            
            pre_h=torch.mm(M_k,alpha) #Weighted sum for every children [hidden, #children]
            h=torch.sum(torch.tanh(torch.mm(self.model_att_sentence.W_att,pre_h)+ torch.stack([self.model_att_sentence.b_att for i in range(pre_h.shape[1])],axis=1)),dim=1)
            #
            h_model=self.model_att_sentence._attention(hiddens, M_q)

            result= torch.sum(torch.abs(h - h_model)).item()
            self.assertEqual(result < 0.00001,True)
    def test_dropout_copy(self):
        p=0.1
        d=torch.bernoulli((1.0-p)*torch.ones(25))
        d1=d
        d2=d
        result = d1 == d
        self.assertEqual(torch.min(result),True)
        self.assertEqual(torch.max(result),True)
if __name__ == '__main__':
    unittest.main()
