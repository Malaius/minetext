import unittest
from crf import CRF
import torch
import itertools
import sys
#
# This test class requires some serious cleanup
#
class TestCRFClass(unittest.TestCase):
    #transition[batch, k , T, S_-T-k, S_T]
    def logprob_k(self, emission, transitions_start, transitions, N, nneigh, z):
        S=emission.shape[1]
        p=sum([emission[t,z[t]] for t in range(N)])
        for k in range(nneigh):
            if k < len(z):
                p=p + transitions_start[k,z[k]]
            for t in range(1,N):
                if t - k - 1 >= 0 and t-k-1 < N:
                    p=p + transitions[k, t, z[t-k -1],z[t]] 
        return p

    def viterbitest(self,  nneigh, lengths, transitions_start, transitions_k, emission):
        batch_size=emission.shape[0]
        chunks = [self.crfnn._get_chunks(list(range(length)), nneigh)[0] for length in lengths]
        z_dec=self.crfnn._viterbi_decode_k(transitions_start, transitions_k, emission, lengths, chunks)
        #emission_A, transitions_A, tail_transitions_A, lengths_A = self.crfnn._consolidate_emission_transition(emission, transitions_k, lengths)
        #z=self.crfnn._viterbi_decode(transitions_A, emission_A, lengths_A, tail_transitions_A)
        #z_dec=self.crfnn._segregate_sequence(z, lengths)
        score=self.crfnn._seq_score_k(transitions_start, transitions_k, emission,lengths,z_dec)
        for batch in range(batch_size):
            manual_score=self.logprob_k(emission[batch,:,:], transitions_start[batch,:,:], transitions_k[batch,:, :,:,:],lengths[batch], nneigh, z_dec[batch,:])
            print("Scores are equals", score[batch], manual_score)
            self.assertEqual(score[batch], manual_score) 
        for batch in range(batch_size):
            zs=[(z_,self.logprob_k(emission[batch,:,:], transitions_start[batch,:,:],transitions_k[batch,:,:,:,:],lengths[batch], nneigh, z_)) for z_ in itertools.product(range(self.S),repeat=lengths[batch])]
            zmax, pmax=max(zs, key=lambda p: p[1])
            print("Maxima are equal, nneigh, batch",nneigh, batch,z_dec[batch,0:lengths[batch]], zmax)
            for i in range(lengths[batch]):
                self.assertEqual(z_dec[batch,i], zmax[i])
            print("Maximum scores are equal",score[batch], pmax)
            self.assertEqual(score[batch], pmax)

    def setUp(self):
        torch.manual_seed(100)
        self.S=3
        
    def test_one_neighbor(self):
        nneigh=1
        self.crfnn=CRF(["A","B","C"], 50, 20, 20, num_neighbors= nneigh)
        lengths=[10,10,10,10,10]
        T=max(lengths)
        batch_size=len(lengths)
        transitions_start=torch.log(torch.rand(batch_size, nneigh,self.S))
        transitions_k=torch.log(torch.rand(batch_size, nneigh, 10, self.S,self.S))
        emission=torch.log(torch.rand(batch_size, T,self.S))
        self.viterbitest(nneigh, lengths, transitions_start, transitions_k, emission)

    def test_two_neighbor(self):
        nneigh=2
        self.crfnn=CRF(["A","B","C"], 50, 20, 20, num_neighbors= nneigh)
        lengths=[10,10,10,10,10]
        T=max(lengths)
        batch_size=len(lengths)
        transitions_start=torch.log(torch.rand(batch_size, nneigh,self.S))
        transitions_k=torch.log(torch.rand(batch_size, nneigh, 10, self.S,self.S))
        emission=torch.log(torch.rand(batch_size, T,self.S))
        self.viterbitest(nneigh, lengths, transitions_start, transitions_k, emission)

    def test_three_neighbor(self):
        nneigh=3
        self.crfnn=CRF(["A","B","C"], 50, 20, 20, num_neighbors= nneigh)
        lengths=[10,10,10,10,10]
        T=max(lengths)
        batch_size=len(lengths)
        transitions_start=torch.log(torch.rand(batch_size, nneigh,self.S))
        transitions_k=torch.log(torch.rand(batch_size, nneigh, 10, self.S,self.S))
        emission=torch.log(torch.rand(batch_size, T,self.S))
        self.viterbitest(nneigh, lengths, transitions_start, transitions_k, emission)

    def test_one_neighbor_different_lengths(self):
        nneigh=1
        self.crfnn=CRF(["A","B","C"], 50, 20, 20, num_neighbors= nneigh)
        lengths=[9,10,7,8,5]
        T=max(lengths)
        batch_size=len(lengths)
        transitions_start=torch.log(torch.rand(batch_size, nneigh,self.S))
        transitions_k=torch.log(torch.rand(batch_size, nneigh, 10, self.S,self.S))
        emission=torch.log(torch.rand(batch_size, T,self.S))
        self.viterbitest(nneigh, lengths, transitions_start, transitions_k, emission)

    def test_two_neighbor_different_lengths(self):
        nneigh=2
        self.crfnn=CRF(["A","B","C"], 50, 20, 20, num_neighbors= nneigh)
        lengths=[9,10,7,8,5]
        T=max(lengths)
        batch_size=len(lengths)
        transitions_start=torch.log(torch.rand(batch_size, nneigh,self.S))
        transitions_k=torch.log(torch.rand(batch_size, nneigh, 10, self.S,self.S))
        emission=torch.log(torch.rand(batch_size, T,self.S))
        self.viterbitest(nneigh, lengths, transitions_start, transitions_k, emission)

    def test_three_neighbor_different_lengths(self):
        nneigh=3
        self.crfnn=CRF(["A","B","C"], 50, 20, 20, num_neighbors= nneigh)
        lengths=[9,10,7,8,5]
        T=max(lengths)
        batch_size=len(lengths)
        transitions_start=torch.log(torch.rand(batch_size, nneigh,self.S))
        transitions_k=torch.log(torch.rand(batch_size, nneigh, 10, self.S,self.S))
        emission=torch.log(torch.rand(batch_size, T,self.S))
        self.viterbitest(nneigh, lengths, transitions_start, transitions_k, emission)

if __name__ == '__main__':
    unittest.main()

