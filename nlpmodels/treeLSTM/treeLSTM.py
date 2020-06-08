import torch
import torch.nn as nn
import networkx as nx
#
#
# Implementation of Sum-child (dependency) tree LST, per Kai Sheng Tai,Richard Socher,Christopher D. Manning, "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks" https://arxiv.org/pdf/1503.00075.pdf
#
# Implements an attention mechanism when aggregating children vectors: Uses model 2 in Mahtab Ahmed, Muhammad Rifayat Samee, Robert E. Mercer  "Improving Tree-LSTM with self-attention"
#

class treeLSTM(nn.Module):
    #
    # input_dim is the final embedding dimension of the input vectors. If input_dim < actual vector dimension, we clip
    #
    def __init__(self, input_dim, hidden_dim,attention_dim,dropout=0.0, num_layers=1, dtype=torch.float32, attention=False, generate_query=False):
        super(treeLSTM, self).__init__()
        #
        if hidden_dim < 1:
            raise ValueError("Hidden_dim cannot be 0 or negative")
        #
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim=attention_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.dtype = dtype
        self.rand_factor = torch.sqrt(torch.tensor(
            1.0/self.hidden_dim, dtype=self.dtype, requires_grad=False))
        self.attention = attention
        self.generate_query=generate_query
        #Initialization as Uniform(1/sqrt(hidden_dim),-1/sqrt(hidden_dim))
        self.W_i_0 = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype), requires_grad=True)
        if self.num_layers > 1:
            self.W_i = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)
        self.U_i = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype), requires_grad=True)
        self.b_i = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype), requires_grad=True)

        self.W_f_0 = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype), requires_grad=True)
        if self.num_layers > 1:
            self.W_f = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)
        self.U_f = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)
        self.b_f = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers, 
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)

        self.W_o_0 = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype), requires_grad=True)
        if self.num_layers > 1:
            self.W_o = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)
        self.U_o = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype), requires_grad=True)
        self.b_o = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype), requires_grad=True)

        self.W_u_0 = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype), requires_grad=True)
        if self.num_layers > 1:
            self.W_u = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)
        self.U_u = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype), requires_grad=True)
        self.b_u = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype), requires_grad=True)
    # Defining parameters for the attention. Attention only places in the last hidden states from children
    # (for the multilayer case)
        if self.attention:
            self.W_key=nn.Parameter(2.0*self.rand_factor*torch.rand(self.attention_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.attention_dim,
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)
            self.W_query=nn.Parameter(2.0*self.rand_factor*torch.rand(self.attention_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.attention_dim,
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)
            #self.W_value=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
            #                                                        self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.hidden_dim,
            #                                                        self.hidden_dim, dtype=self.dtype), requires_grad=True)
            self.W_att=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype), requires_grad=True)
            self.b_att = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim, dtype=self.dtype)-self.rand_factor*torch.ones(self.hidden_dim, dtype=self.dtype), requires_grad=True)
            
        if self.generate_query:
            self.build_query=nn.LSTM(self.input_dim,self.hidden_dim,self.num_layers,bias=True,batch_first=True, dropout=self.dropout, bidirectional=False)
        
    def _initialize_c(self):
        return torch.zeros(self.hidden_dim, dtype=self.dtype)
        #return 2.0*self.rand_factor*torch.rand(self.hidden_dim, dtype=self.dtype)-self.rand_factor

    def _initialize_h(self):
        return torch.zeros(self.hidden_dim, dtype=self.dtype)
        #return 2.0*self.rand_factor*torch.rand(self.hidden_dim, dtype=self.dtype)-self.rand_factor
    #
    # hiddens_ -> All descendants of the node of interest [hidden_dim] - Not only children!
    # query is either a [hidden_dim] tensor generated from a different sequence we want to compare 
    # If None, we use as query_ the column-stacked children hidden vectors [hidden_dim, num_children]
    # 
    # 
    def _attention(self, hiddens_, query_=None):
        M_key=torch.stack(hiddens_,axis=1)
        if query_ is None:
            M_query=torch.stack(hiddens_,axis=1)
        else:
            M_query = query_
        if len(M_query.shape) == 2:
            query=torch.mm(self.W_query, M_query)
            key=torch.mm(self.W_key, M_key)
            align=torch.mm(torch.transpose(query,0,1),key)*self.rand_factor
            #
            # Each row (dim=0) is the attention probability for every children according to the owrd in that row
            # Normalize on columns
            alpha=torch.nn.functional.softmax(align, dim=0) #
            pre_h=torch.mm(M_key,alpha) #Weighted sum for every children [hidden, #children]
            h_attentive=torch.sum(torch.tanh(torch.mm(self.W_att,pre_h)+ torch.stack([self.b_att for i in range(pre_h.shape[1])],axis=1)),dim=1)
        elif len(M_query.shape) == 1:
            query=torch.mv(self.W_query, M_query)
            key=torch.mm(self.W_key, M_key)
            align=torch.mv(torch.transpose(key,0,1),query)*self.rand_factor
            #
            # Each row (dim=0) is the attention probability for every children according to the owrd in that row
            # Normalize on columns
            alpha=torch.nn.functional.softmax(align, dim=0) #
            pre_h=torch.mv(M_key,alpha) #Expected value of the attention given to every child
            h_attentive=torch.tanh(torch.mv(self.W_att,pre_h) + self.b_att)
        else:
            raise ValueError("Attention query needs to be a vector or a matrixs")
        return h_attentive
    # Initialized list of hidden and cell states (typically, from children)
    # Output: Updated Node hidden and cell state
    #
    # hiddens_ list of hidden vectors from children node
    # all_hiddens -> List of hidden vectors from all descendants
    #
    def _combine_children_hidden(self, hiddens_, all_hiddens_, query=None):
        if self.attention:
            return self._attention(all_hiddens_, query)
        else:  # Standard child-sum combination fr dependency-based treeLSTM
            return torch.sum(torch.stack([h_ for h_ in hiddens_], axis=0), axis=0)

    #
    def _cell_forward(self, x_, hiddens_0, cells_0, all_hiddens_0, query=None):
        x = x_[0:self.input_dim]  # Clipping by default. Moved to forward
        #Summing the children contribution
        hiddens_ = hiddens_0
        cells_ = cells_0
        h_sum = self._combine_children_hidden(hiddens_, all_hiddens_0, query)
        #h_sum=torch.sum(torch.stack([h_ for h_ in hiddens_],axis=0),axis=0)
        #c_sum=torch.sum(torch.stack([c_ for c_ in cells_],axis=0),axis=0)
        #Node input gate
        i = torch.sigmoid(torch.matmul(self.W_i_0, x) +
                          torch.matmul(self.U_i[0, :, :], h_sum) + self.b_i[0, :])
        #Link forget gate
        f_pc = []
        #torch.ger is a tensor product. Not needed
        for h_ in hiddens_:
            f_pc.append(torch.sigmoid(torch.matmul(self.W_f_0, x) +
                                      torch.matmul(self.U_f[0, :, :], h_) + self.b_f[0, :]))
        #Node output gate
        o = torch.sigmoid(torch.matmul(self.W_o_0, x) +
                          torch.matmul(self.U_o[0, :, :], h_sum) + self.b_o[0, :])
        #Auxiliary
        u = torch.tanh(torch.matmul(self.W_u_0, x) +
                       torch.matmul(self.U_u[0, :, :], h_sum) + self.b_u[0, :])
        #Hadamard (element wise products)
        #Cell gate
        c = i*u + \
            torch.sum(torch.stack(
                [f*c_ for f, c_ in zip(f_pc, cells_)], axis=0), axis=0)
        #Hidden state
        h = o * torch.tanh(c)
        # Additional layers
        for l in range(1, self.num_layers):
            x_1 = h
            hiddens_ = hiddens_0
            cells_ = cells_0
            h_sum = torch.sum(torch.stack(
                [h_ for h_ in hiddens_], axis=0), axis=0)
            i = torch.sigmoid(torch.matmul(
                self.W_i[l-1, :, :], x_1)+torch.matmul(self.U_i[l, :, :], h_sum) + self.b_i[l, :])
            #Link forget gate
            f_pc = []
            #torch.ger is a tensor product. Not needed
            for h_ in hiddens_:
                f_pc.append(torch.sigmoid(torch.matmul(
                    self.W_f[l-1, :, :], x_1)+torch.matmul(self.U_f[l, :, :], h_) + self.b_f[l, :]))
            #Node output gate
            o = torch.sigmoid(torch.matmul(
                self.W_o[l-1, :, :], x_1)+torch.matmul(self.U_o[l, :, :], h_sum) + self.b_o[l, :])
            #Auxiliary
            u = torch.tanh(torch.matmul(
                self.W_u[l-1, :, :], x_1)+torch.matmul(self.U_u[l, :, :], h_sum) + self.b_u[l, :])
            #Hadamard (element wise products)
            #Cell gate
            c = i*u + \
                torch.sum(torch.stack(
                    [f*c_ for f, c_ in zip(f_pc, cells_)], axis=0), axis=0)
            #Hidden state
            h = o * torch.tanh(c)
        return h, c

    def _parent_covered(self, tree, set_covered_children, parent):
        schildren = set(tree.successors(parent))
        return len(schildren.difference(set_covered_children)) == 0, schildren
    #
    # Calculates hidden, output and cell states for a batch of dependency trees
    # A tree is an instance of nx.DiGraph, with a "vector" property to store  the word embedding
    # The relationship goes from parent to child
    # 
    #
    def generate_batch_query(self,trees):
        # We do not batch the execution. A hit in performance, but this is just an initial version
        query_batch_=[]
        for tree in trees:
            #Node ID is the position in the sentence
            x_=torch.stack([v[1] for v in sorted([(n, tree.nodes[n]["dict"]["vector"][0:self.input_dim]) for n in tree], key=lambda p: p[0])],dim=0)
            x=x_.reshape(1,x_.shape[0], x_.shape[1])
            h_0_=self._initialize_h()
            h_0=h_0_.reshape(1,1,h_0_.shape[0])
            c_0_=self._initialize_c()
            c_0=c_0_.reshape(1,1,c_0_.shape[0])
            _,(h_n,_) = self.build_query(x,(h_0,c_0))
            query_batch_.append(h_n.reshape(h_0_.shape[0]))
        return torch.stack(query_batch_,dim=0) #[batch_size, hidden_size]

    def forward(self, trees, query=None):
        #
        # Always initialize the first hidden and cell state vectors (zero)
        #
        root_hidden = []
        root_cell = []
        for tree_idx, tree in enumerate(trees):
            #
            # Droput mask vectors. We will apply the same mask during all steps for a single tree
            # (trying to follow the tied weights method described in Yarin Gal, Zoubin Ghahramani "A Theoretically Grounded Application of Dropout in Recurrent Neural Network")
            #
            z_x=torch.bernoulli((1.0 -self.dropout)*torch.ones(self.input_dim))
            z_h=torch.bernoulli((1.0 -self.dropout)*torch.ones(self.hidden_dim))
            #Initialize on leaves
            children = [n for n in tree if tree.out_degree(
                n) == 0 and tree.in_degree(n) > 0]
            processed_nodes = set()
            for sc in children:  # Apply cell update on leaves
                h_0 = self._initialize_h()
                c_0 = self._initialize_c()
                h, c = self._cell_forward(z_x*tree.nodes[sc]["dict"]["vector"][0:self.input_dim],
                                          [z_h*h_0],
                                          [c_0],
                                          [z_h*h_0],
                                          None if query is None else query[tree_idx,:])
                tree.nodes[sc]["dict"]["h"] = h
                tree.nodes[sc]["dict"]["c"] = c
                processed_nodes.add(sc)
            go_ahead = True
            while go_ahead:
                parents = []
                children_local = []
                schildren = set(children)
                for sc in schildren:
                    for n in set(tree.predecessors(sc)).difference(schildren):
                        is_ok, schildren_local = self._parent_covered(
                            tree, schildren, n)
                        if is_ok and not n in parents:  # Prevent duplicates
                            parents.append(n)
                            children_local.append(schildren_local)
                for p, c_local in zip(parents, children_local):
                    h, c = self._cell_forward(z_x*tree.nodes[p]["dict"]["vector"][0:self.input_dim],
                                              [z_h*tree.nodes[c_]["dict"]["h"]
                                                  for c_ in c_local],
                                              [tree.nodes[c_]["dict"]["c"]
                                                  for c_ in c_local],
                                               [z_h*tree.nodes[c_]["dict"]["h"] for c_ in nx.descendants(tree,p)],
                                               None if query is None else query[tree_idx,:])
                    tree.nodes[p]["dict"]["h"] = h
                    tree.nodes[p]["dict"]["c"] = c
                    processed_nodes.add(p)
                children = list(set(children + parents))
                go_ahead = len(parents) > 0
            #Validation:
            all_covered = len(
                [n for n in tree if not n in processed_nodes]) > 0
            if all_covered:
                raise Warning("Not all nodes covered")
            roots = [n for n in tree if tree.out_degree(
                n) > 0 and tree.in_degree(n) == 0]
            root_hidden.append(torch.mean(torch.stack(
                [tree.nodes[n]["dict"]["h"] for n in roots], axis=0), axis=0))
            root_cell.append(torch.mean(torch.stack(
                [tree.nodes[n]["dict"]["c"] for n in roots], axis=0), axis=0))
        return trees, torch.stack(root_hidden, axis=0), torch.stack(root_cell, axis=0)
