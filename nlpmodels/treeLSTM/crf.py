# Starting from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
#
# Followo¡ing http://www.cs.columbia.edu/~mcollins/crf.pdf
# We have a graph of labels s_g=(s_1, S_2... \in S). We have input data x for every node (x_g=(x_1...)). We aim to build a  model
# p(s_g|x_g) as a "giant" log linear model, with parameters w_g=(w_1,w_2...) for every node
#  
#
#
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools

# Compute log sum exp in a numerically stable way for the forward algorithm
#
# Generic viterbi decoding. S is the number of (hidden labels). For a sequence of T observations (x)
# b_s (x) emission probability of x if in state s and a_s1s2 the transition probability between inner states
# it finds the more likely sequence of hidden states
# transition (s+2xs+2) is the transition matrix, including an initial en end state. emission (sx T) encodes the 
# probability of emission of the observed value at position t
#transition has the start-end states in positions S and S+1
#
# transition is a feature that can depend on x
#
#
class CRF(nn.Module):
    #tag_to_ix: ordered list of tags
    # neighs: define transitions to consider in a linear chain: transition logP (i, i + neigh) for neigh in neighs 
    def __init__(self, tag_to_ix, embedding_dim, hidden_dim, hidden_dim_trans, num_neighbors=1, dropout=0.0, transition_type="constant"):
        super(CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_trans = hidden_dim_trans
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self._tag_encoder=dict([(tag_to_ix[i],i) for i in range(self.tagset_size)])
        self.LOW=-1.0e20
        self.num_neigbors=num_neighbors
        #
        # Checking if tags are BIO style, and create a transition exclusion list:
        #  B_... -> O, B_label1 -> I_label2 (2<>1)
        #
        bios=[t == "O" or t[0:2] == "B-" or t[0:2] == "I-" for t in tag_to_ix]
        self.forbidden_transitions=[]
        self.forbidden_starts=[]
        #Forbid transitions at first neighbors only
        if len(tag_to_ix) == len([b for b in bios if b]) and "O" in tag_to_ix:
            for t in tag_to_ix:
                # Forbid B-label1 -> I-label2. We MUST allow B-label-> O
                if t[0:2] == "B-":
                    for t1 in tag_to_ix:
                        if t1[0:2] == "I-" and t1[2:] != t[2:]:
                            self.forbidden_transitions.append((self.tag_encoder(t), self.tag_encoder(t1)))
                elif t[0:2] == "I-":
                    # Forbid Start  -> I-label2
                    self.forbidden_starts.append(self.tag_encoder(t))
                    # Forbid O -> I-label2
                    self.forbidden_transitions.append((self.tag_encoder("O"), self.tag_encoder(t)))
                    # Forbid I-label1 -> I-label2
                    for t1 in tag_to_ix:
                        if t1[0:2] == "I-" and t1[2:] != t[2:]:
                            self.forbidden_transitions.append((self.tag_encoder(t), self.tag_encoder(t1)))
        self.new_label_idxs=list(itertools.product(range(self.tagset_size),repeat=self.num_neigbors))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, #// Floor division by 2, because output dim= hidden* num directions
                            num_layers=1, bidirectional=True, bias=True, batch_first=True,
                            dropout=dropout)
        
        # Maps the output of the LSTM into tag space.
        # input=(batch,...,input_dim)
        # output=(batch,...,output_dim)
        self.lstm_to_emission=nn.Linear(hidden_dim, self.tagset_size, bias=True)
        self.logsoftmax=torch.nn.LogSoftmax(dim=-1)
        #if potential_type = "linear"
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        if transition_type == "constant":
            self.transitions_k_start=nn.Parameter(torch.randn(self.num_neigbors,self.tagset_size), requires_grad=True)
            self.transitions_k = nn.Parameter(torch.randn(self.num_neigbors,self.tagset_size,self.tagset_size), requires_grad=True)
        else:
            for k in range(num_neighbors):
                #This is to access parameters in self.named_parameters
                self.add_module("lstm_trans_k" + str(k),
                                    nn.LSTM(embedding_dim, hidden_dim_trans // 2, #// Floor division by 2, because output dim= hidden* num directions
                                        num_layers=1, bidirectional=True, bias=True, batch_first=True,
                                        dropout=dropout))
                self.add_module("lstm_to_transition_k" + str(k),
                                    nn.Linear(hidden_dim_trans, self.tagset_size**2, bias=True))
                self.add_module("vector_to_relu_k" + str(k),
                                nn.Linear(embedding_dim, hidden_dim_trans, bias=True))
                self.add_module("relu_trans_start_k" + str(k),nn.ReLU(inplace=False))
                self.add_module("relu_to_transition_start_k" + str(k),
                                nn.Linear(hidden_dim_trans, self.tagset_size, bias=True))
            self.dict_layers=dict([m for m in self.named_modules()])
            self.hidden_layer_trans =[None for k in range(num_neighbors)]
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        #We use the self.tagset_size position for transitions from the start (sequence: START - 0,1...)
        #self.hidden = self._init_hidden()
        self.transition_type=transition_type
        self.emissions=None
        self.logemissions=None
        #transition[batch, k , T, S_-T-k, S_T]
        self.logtransitions_k = None
        self.logtransitions_k_start = None
    #
    def aggregate_gradients(self):
        out={}
        for name, parameter in self.named_parameters():
            if not parameter.grad is None:
                out[name]= torch.max(parameter.grad.view(-1).data)
        return out
    #Business rules
    def _blacklist_transitions(self):
        #transition[batch, k , T, S_-T-k, S_T]
        if self.transition_type == "constant":
            for idx0, idx1 in self.forbidden_transitions:
                self.transitions_k[0,idx0,idx1] = self.LOW
            #
            for idx0 in self.forbidden_starts:
                self.transitions_k_start[0,idx0] = self.LOW
        else:
            for idx0, idx1 in self.forbidden_transitions:
                self.logtransitions_k[:,0,:,idx0,idx1] = self.LOW
            #
            for idx0 in self.forbidden_starts:
                self.logtransitions_k_start[:,0,idx0] = self.LOW
        
    #
    # To expose gradients easily for monitoring purposes
    #
    def tag_encoder(self,tag):
        if tag in self._tag_encoder.keys():
            return self._tag_encoder[tag]
        else:
            return -1 
    #
    # Emissions is batched. Transitions it is not
    #https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    #
    def _get_chunks(self,old_steps, chunk_size):
        if chunk_size < 1:
            raise AttributeError("_right_chunks: chunk_size must be > 0")
        out=[]
        out_relative=[]
        for i in range(0,len(old_steps),chunk_size):
            out.append(old_steps[i:i+chunk_size])
            out_relative.append(list(range(len(out[-1]))))
        return out, out_relative
    #
    # From a secuence of new labl idx, return the equivalent sequence of label idx
    # z[batch, index]
    def _segregate_sequence(self, z, lengths):
        out=[]
        max_len=max(lengths)
        for batch in range(z.shape[0]):
            row=[]
            for col in range(z.shape[1]):
                row=row + list(self.new_label_idxs[z[batch,col]])
            out.append(torch.tensor(row[0:max_len]))
        return torch.stack(out, axis=0)

    def _init_hidden(self, hidden_dim, batch_size=1):
        #Even if batch_first = True, hiddn and cell state tensors batch dimension is for dim=1
        return (torch.randn(2, batch_size, hidden_dim // 2),
                torch.randn(2, batch_size, hidden_dim // 2))
    # 
    # Execute once before training to allocate the required tensors
    # transition_start[batch, k, S_k]
    #
        
    # Bishop pattern Recognition, pp 411. Solves the decoding problem x^*=arg max_x p(x), for P the 
    # probability distributions of graphic models. A generalization of the Viterbi algorithm
    # Defined for trees (exact in this case). A generalization for other type graphs would be the junction tree
    #
    # This works with logarithms of probability to prevent under/overflows
    # It is message passing in a factor graph
    # The CRF model is: p(y|wx)=exp(Sum_n w_n*phi_n(x,y_n) + SumG_n1n2 w_n*phi_n(x,y_n1,y_n2) )/Z with graph SumG a sum
    # over the nodes of a graph. This distribution can be represented as a factor graph
    # We will assume linear graphs with k neighbors so we can use the Viterbi algorithm for
    # decoding (y*=arg max_y P(y|wx)) anf the forward algorithm to evaluate the partition function
    # Z=Sum_y p(y|wx)
    # Viterbi is a special case of the max sum algorithm when transitions is only between first neighbors
    # For general graphs, we either do brute forcce calculation or apply approxiations such as loopy belief propagation
    # or AD3 (see pystruct)
    #
    # We consider always linear graphs with K neighbors by aggregating neighboring nodes and applying brute force in the 
    # aggregated nodes and viterbi in the aggregated graph
    # The first ad last nodes of the chain are "virtual" start and end nodes. This way we can model specifically
    # the probbaility of being at the end or the beginning of the sequence
    # S-> Number of states for y
    # T -> Length of chain (without including special nodes for start and end)
    # (log)transitions -> (batch, S+1,S) Depends implicitely on all  observed values
    #   T(x)_s0s1=log Prob of going from state 0 in x_i-1 to s1 in x_i
    # (log)emissions -> (batch, T,S)
    # This implementation assumes a minibatch dimension in the emission/transition tensors  
    def _viterbi_decode(self,transitions, emissions, lengths, tail_transition=None):
        #Initialization
        # [0...,S-1] states, S-> Start label,
        #Checking dtypes is critical when adding a tensor to tensor.slices
        if emissions.dtype != transitions.dtype:
            raise AttributeError("Emissions and Transition tensors have different dtypes")
        S=emissions.shape[2]
        T=emissions.shape[1]
        batch_size=emissions.shape[0]
        T1=torch.zeros(batch_size, S,T, dtype=transitions.dtype)
        T2=torch.zeros(batch_size, S,T,dtype=torch.int64)
        z=torch.zeros(batch_size, T,dtype=torch.int64)
        T1[:,:,0]=transitions[:,S,0:S] + emissions[:,0,0:S]
        T2[:,:,0]=S #torch.tensor([S in range(batch_size)]) #The previous label
        for n in range(batch_size): 
            for t in range(1,lengths[n]): ##Here
                a=torch.stack([T1[n,:,t-1] for i in range(S)], dim=1)
                b=torch.stack([emissions[n,t,:] for i in range(S)], dim=0) #Stack as rows
                if t == lengths[n] -1 and not tail_transition is None:
                    B=a +tail_transition[n,0:S,0:S] + b
                else:
                    B=a +transitions[n,0:S,0:S] + b
                T1[n,:,t], T2[n,:, t] = torch.max(B,0)
            _, z[n,lengths[n]-1] = torch.max(T1[n,:,lengths[n]-1],0)
            for t in range(lengths[n]-1,0,-1):
                z[n,t-1]=T2[n,z[n,t],t]
                
        return z
    # transition_start[batch, k, S_k] < Probability of having label S:k in the kth position
    # transition[batch, k , T, S_-T-k, S_T]
    # emissions[batch, T, S_T]
    # chunks[batch] -> list of chunks with the real indices, not relative
    def _viterbi_decode_k(self,transitions_start, transitions, emissions, lengths, chunks):
        #Initialization
        # [0...,S-1] states, S-> Start label,
        #Checking dtypes is critical when adding a tensor to tensor.slices
        if emissions.dtype != transitions.dtype:
            raise AttributeError("Emissions and Transition tensors have different dtypes")
        S=emissions.shape[2]
        T=emissions.shape[1]
        batch_size=emissions.shape[0]
        lengths_red=[len(chunks[n]) for n in range(batch_size)]
        T_red=max(lengths_red)
        #Check all batches have  at least neigh nodes
        if min([len(chunks[n][0]) for n in range(batch_size)]) < self.num_neigbors:
            raise ValueError("A record in the batch is too short")
        #self.num_neigbors=num_neighbors
        #self.new_label_idxs=list(itertools.product(range(self.tagset_size),repeat=self.num_neigbors))
        #chunks -> indices
        nneigh=transitions.shape[1]
        T1=self.LOW*torch.ones(batch_size, len(self.new_label_idxs),T_red, dtype=transitions.dtype)
        T2=torch.zeros(batch_size, len(self.new_label_idxs),T_red,dtype=torch.int64)
        z=torch.zeros(batch_size, T_red,dtype=torch.int64)
        #Initialize
        for i, s in enumerate(self.new_label_idxs):
            A_=torch.zeros(batch_size, dtype=transitions.dtype)
            for j, c in enumerate(chunks[0][0]): #We assume the first chunk is the same in the full batch 
                A_ = A_ + transitions_start[:,c,s[j]]+ emissions[:,c,s[j]]
                for k in range(j):#Internal transitions
                    A_ = A_ +  transitions[:,j-k-1,c,s[k],s[j]]
            T1[:,i,0] = A_
        T2[:,:,0]=S #torch.tensor([S in range(batch_size)]) #The previous label
        # Loop through remaining positions
        for n in range(batch_size): 
            for t in range(1,lengths_red[n]): ##t is the position in the "reduced lattice"
                a=torch.stack([T1[n,:,t-1] for i in range(len(self.new_label_idxs))], dim=1)
                b_=torch.zeros(len(self.new_label_idxs))
                for i, s in enumerate(self.new_label_idxs):
                    for j, c in enumerate(chunks[n][t]): #We assume the first chunk is the same in the full batch 
                        b_[i] = b_[i] + emissions[n,c,s[j]]
                        for k in range(j):#Internal transitions
                            b_[i] = b_[i] +  transitions[n,j-k-1,c,s[k],s[j]]
                B=a + torch.stack([b_ for j in range(len(self.new_label_idxs))], dim=0)
                #B=a +transitions[n,0:S,0:S] + b
                for i, s in enumerate(self.new_label_idxs):
                    for i1, s1 in enumerate(self.new_label_idxs):
                        for j_p, c_p in enumerate(chunks[n][t-1]):
                            #0 1  2 3   2-> (0,1) (1,0)  3-> (1,1) (2,0)
                            for k in range(min(chunks[n][t][0] -c_p -1,nneigh-1), min(chunks[n][t][-1] -c_p,nneigh)):
                                I=c_p + k + 1 -chunks[n][t][0]
                                B[i1, i] = B[i1, i] +  transitions[n,k,c_p + k + 1,s1[j_p],s[I]]    
                T1[n,:,t], T2[n,:, t] = torch.max(B,0)
            #Recovering reduced indices                
            _, z[n,lengths_red[n]-1] = torch.max(T1[n,:,lengths_red[n]-1],0)
            for t in range(lengths_red[n]-1,0,-1):
                z[n,t-1]=T2[n,z[n,t],t]
        return self._segregate_sequence(z, lengths)
    
    #
    # Calculate log Z (log of partition function)
    #
    # vec is a [*,:] dimensional array. This function mitigates possibility of numerical overflow
    #Evaluate log_sum_exp of a tensor on a given dimension (gneralization of log_sum_exp)
    def _log_sum_exp2(self, vec, dim=0):
        max_score, idx = torch.max(vec, dim)
        max_score_broadcast=torch.stack([max_score for i in range(vec.shape[dim])], dim=dim)
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=dim))

    #
    # Transitions and emissions are log probabilities. This is to calculate the partition function
    # of the CRF model given the observed values x
    # Implemented for a minibatch
    ## S-> Number of states for y
    # T -> Length of chain (without including special nodes for start and end)
    # (log)transitions -> (batch, S+1,S) Depends implicitely on all  observed values
    #   T(x)_s0s1=log Prob of going from state 0 in x_i-1 to s1 in x_i
    # (log)emissions -> (batch, T,S)

    def _forward_algorithm(self, transitions, emissions, lengths):
        S=emissions.shape[2]
        T=emissions.shape[1]
        batch_size=emissions.shape[0]
        logalpha=torch.zeros(batch_size, S,T, dtype=emissions.dtype)
        logZ=torch.zeros(batch_size, dtype=emissions.dtype)
        logalpha[:,:,0]=emissions[:,0,:] + transitions[:,S,0:S]
        for n in range(batch_size):
            for t in range(1,lengths[n]):
                a=torch.stack([logalpha[n,:,t-1] for s in range(S)], dim=1)
                b=a+ transitions[n,0:S,0:S] #Index runs the labels in the previous position
                logalpha[n,:,t]=emissions[n,t,:] + self._log_sum_exp2(b, dim=0)
            logZ[n]=self._log_sum_exp2(logalpha[n,:,lengths[n] - 1], dim=0)
        return logZ

    #
    # log P + log Z for  a batch
    # This is the log likeliehood  plust the log partition function
    # def _seq_score(self, transitions, emissions, lengths, seq):
    #     S=emissions.shape[2]
    #     T=emissions.shape[1]
    #     batch_size=emissions.shape[0]
    #     logp=torch.zeros(batch_size,dtype=emissions.dtype)
    #     for n in range(batch_size):
    #         logp[n]=emissions[n,0,seq[n,0]] + transitions[n,S,seq[n,0]]
    #         for t in range(1,lengths[n]):
    #             logp[n]=logp[n] + emissions[n,t,seq[n,t]] + transitions[n,seq[n,t-1],seq[n,t]]
    #     return logp
    # # Score of a batch of sequences
    #Transitions are batched
    def _seq_score_k(self, transitions_start, transitions_k, emissions, lengths, seq):
        nneigh=transitions_k.shape[1]
        S=emissions.shape[2]
        T=emissions.shape[1]
        batch_size=emissions.shape[0]
        logp=torch.zeros(batch_size,dtype=emissions.dtype)
        for n in range(batch_size):
            logp[n]=torch.sum(torch.tensor([emissions[n,t,seq[n,t]] for t in range(lengths[n])]))
            for k in range(nneigh):
                if k < lengths[n]:
                    logp[n]=logp[n] + transitions_start[n, k,seq[n,k]]
                for t in range(1,lengths[n]):
                    if t -k -1 >= 0 and t -k -1 < lengths[n]:
                        logp[n]=logp[n] + transitions_k[n,k, t,seq[n,t -k -1],seq[n,t]]
        return logp

    #input is a tensor (batch_size,max seq size,embedding)
    #lengths is a tensor (batch_size,length of sequence)
    def forward(self, input_sequence, lengths): 
        # Get the emission scores from the BiLSTM
        #Cleanup always the hidden state of the lSTM
        batch_size=input_sequence.shape[0]
        max_seq_size=input_sequence.shape[1]
        #Initializing here give us flexibility to call the model on different batch sizes during inference
        self.logtransitions_k = torch.randn(batch_size, self.num_neigbors,max_seq_size,self.tagset_size, self.tagset_size)
        self.logtransitions_k_start = torch.randn(batch_size, self.num_neigbors,self.tagset_size)
        
        self.hidden=self._init_hidden(self.hidden_dim, batch_size=batch_size)
        #pre-emissions is the hidden layer, a product of a sigmoid (in [0,1)]*tanh (in [-1,1])
        #
        pre_emissions, self.hidden=self.lstm(input_sequence,self.hidden)
        #pre_emissions is (batch, seq, 2*hidden size), emissions is batch, seq, tagset_size
        self.emissions=self.lstm_to_emission(pre_emissions)
        # Create transition matrix
        # 
        if  self.transition_type == "constant":
            #transition_k [batch , k , T, label, label]
            self._blacklist_transitions()
            self.logtransitions_k=self.logsoftmax(self.transitions_k.reshape(self.num_neigbors, self.tagset_size* self.tagset_size)).reshape(self.num_neigbors, self.tagset_size, self.tagset_size).expand(batch_size, max_seq_size,-1,-1,-1).transpose(1,2)
            self.logtransitions_k_start=self.logsoftmax(self.transitions_k_start).expand(batch_size,-1,-1)
        else:
            for i in range(self.num_neigbors):
                self.hidden_layer_trans[i]=self._init_hidden(self.hidden_dim_trans, batch_size=batch_size)
                #Apply biLSTM
                ltran = self.dict_layers["lstm_trans_k" + str(i)]
                lstm_to_transition = self.dict_layers["lstm_to_transition_k" + str(i)]
                relu_trans_start = self.dict_layers["relu_trans_start_k" + str(i)]
                vector_to_relu = self.dict_layers["vector_to_relu_k" + str(i)]
                relu_to_transition_start= self.dict_layers["relu_to_transition_start_k" + str(i)]
                pre_flat_trans, self.hidden_layer_trans[i] = ltran(input_sequence,self.hidden_layer_trans[i])
                #Apply RELU to first vector in sequence
                pre_flat_trans_start = relu_trans_start(vector_to_relu(input_sequence[:,i,:]))
                self.logtransitions_k_start[:,i,:]=self.logsoftmax(relu_to_transition_start(pre_flat_trans_start).reshape(batch_size,self.tagset_size))
                #Apply linear transformations and logsoftmax
                self.logtransitions_k[:,i,:,:,:]=self.logsoftmax(lstm_to_transition(pre_flat_trans).reshape(batch_size,max_seq_size, self.tagset_size, self.tagset_size))
                self._blacklist_transitions()
                # Batch the transition matrix
        self.logemissions=self.logsoftmax(self.emissions)
        chunks=[self._get_chunks(list(range(length)), self.num_neigbors)[0] for length in lengths]
        best_sequence= self._viterbi_decode_k(self.logtransitions_k_start,
                                                self.logtransitions_k, 
                                                self.logemissions, 
                                                lengths,
                                                chunks)
        best_score=self._seq_score_k(self.logtransitions_k_start,
                                                self.logtransitions_k, 
                                                self.logemissions, 
                                                lengths,
                                                best_sequence)
        return best_sequence, best_score
    
    # target_seq is already using the label indices
    def loss_structural(self, input_data, input_lengths, target_seq):
        best_sequence, best_score= self.forward(input_data, input_lengths)
        t_seq=torch.tensor(target_seq)
        target_score=self._seq_score_k(self.logtransitions_k_start,
                                                self.logtransitions_k, 
                                                self.logemissions, 
                                                input_lengths,
                                                t_seq)
        #Average on the minibath. Torch mean requires a float datatype 
        return torch.mean(best_score - target_score) # >= 0 yes, best score is arg max by construction