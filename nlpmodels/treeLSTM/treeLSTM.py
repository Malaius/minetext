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
    def __init__(self, input_dim, hidden_dim, dropout=0.0, num_layers=1, dtype=torch.float32, combination_mode="sum", attention=False):
        super(treeLSTM, self).__init__()
        #
        if hidden_dim < 1:
            raise ValueError("Hidden_dim cannot be 0 or negative")
        #
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.dtype = dtype
        self.rand_factor = torch.sqrt(torch.tensor(
            1.0/self.hidden_dim, dtype=self.dtype))
        self.combination_mode = combination_mode
        self.attention = attention
        #Initialization as Uniform(1/sqrt(hidden_dim),-1/sqrt(hidden_dim))
        self.W_i_0 = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        if self.num_layers > 1:
            self.W_i = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.U_i = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.b_i = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)

        self.W_f_0 = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        if self.num_layers > 1:
            self.W_f = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.U_f = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.b_f = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)

        self.W_o_0 = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        if self.num_layers > 1:
            self.W_o = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.U_o = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.b_o = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)

        self.W_u_0 = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                  self.input_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        if self.num_layers > 1:
            self.W_u = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers-1, self.hidden_dim,
                                                                    self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.U_u = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers, self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.b_u = nn.Parameter(2.0*self.rand_factor*torch.rand(self.num_layers,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
    # Defining parameters for the attention. Attention only places in the last hidden states from children
    # (for the multilayer case)
        self.W_key=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.W_query=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.W_value=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.W_att=nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim,
                                                                self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
        self.b_att = nn.Parameter(2.0*self.rand_factor*torch.rand(self.hidden_dim, dtype=self.dtype)-self.rand_factor, requires_grad=True)
    
    def _initialize_c(self, cell_initial=None):
        if cell_initial is None:
            return torch.zeros(self.hidden_dim, dtype=self.dtype)
        else:
            return cell_initial
        #return 2.0*self.rand_factor*torch.rand(self.hidden_dim, dtype=self.dtype)-self.rand_factor

    def _initialize_h(self, hidden_initial=None):
        if hidden_initial is None:
            return torch.zeros(self.hidden_dim, dtype=self.dtype)
        else:
            return hidden_initial
        #return 2.0*self.rand_factor*torch.rand(self.hidden_dim, dtype=self.dtype)-self.rand_factor
    #
    # hiddens_ -> All descendants of the node of interest [hidden_dim] - Not only children!
    # query is either a [hidden_dim] tensor generated from a different sequence we want to compare 
    # If None, we use as query_ the column-stacked children hidden vectors [hidden_dim, num_children]
    # 
    # 
    def _attention(self, hiddens_, query_=None):
        M_key=torch.stack(hiddens_,axis=1)
        key=torch.mm(self.W_key,M_key) #[hidden_dim, #children]
        if query_ is None:
            M_query=torch.stack(hiddens_,axis=1)
        else:
            M_query = query_
        if len(M_query.shape) == 2:
            query=torch.mm(self.W_query,M_query) #[hidden_dim, #children] or [hidden_dim]
            align=torch.mm(torch.transpose(query,0,1),key)*self.rand_factor #[#children, #children]
            #Attention probability (for the word corresponding to row n, the different values in that row
            #  are the attention that needs ot be given to each word in the subtree)
            alpha=torch.nn.functional.softmax(align, dim=1) # 
            h_pre_attentive=torch.sum(torch.mm(self.W_att, torch.mm(alpha, M_k),dim=0) # [hidden_dim]
        elif len(M_query.shape) == 2:
            query=self.W_query @ M_query
            align = (query @ key).reshape(1,-1)*self.rand_factor #[1, #children]
            alpha=torch.nn.functional.softmax(align, dim=1) # 
            h_pre_attentive = self.W_att @ (alpha @ M_k) # [hidden_dim]
        else:
            raise ValueError("Attention query needs to be a vector or a matrixs")
        h_attentive=nn.functional.tanh(h_pre_attentive + self.b_att)
        return h_attentive
    # Initialized list of hidden and cell states (typically, from children)
    # Output: Updated Node hidden and cell state
    #
    # hiddens_ list of hidden vectors from children node
    #

    def _combine_children_hidden(self, hiddens_, children_x_, children_idx_):
        if self.combination_mode == "sum":
            return torch.sum(torch.stack([h_ for h_ in hiddens_], axis=0), axis=0)
        else:  # Placeholder for future children combination modes
            return torch.sum(torch.stack([h_ for h_ in hiddens_], axis=0), axis=0)

    #
    def _cell_forward(self, x_, hiddens_0, cells_0, children_x_, children_idx_):
        x = x_[0:self.input_dim]  # Clipping by default
        #Summing the children contribution
        hiddens_ = hiddens_0
        cells_ = cells_0
        h_sum = self._combine_children_hidden(
            hiddens_, children_x_, children_idx_)
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
    # hiddens
    #

    def forward(self, trees, hiddens_initial=None, cells_initial=None):
        #
        # Always initialize the first hidden and cell state vectors (zero)
        #
        root_hidden = []
        root_cell = []
        for tree_idx, tree in enumerate(trees):
            #Initialize on leaves
            children = [n for n in tree if tree.out_degree(
                n) == 0 and tree.in_degree(n) > 0]
            processed_nodes = set()
            for sc in children:  # Apply cell update on leaves
                if hiddens_initial is None:
                    h_0 = self._initialize_h()
                else:
                    h_0 = self._initialize_h(hiddens_initial[tree_idx, :])
                if cells_initial is None:
                    c_0 = self._initialize_c(None)
                else:
                    c_0 = self._initialize_c(cells_initial[tree_idx, :])
                h, c = self._cell_forward(tree.nodes[sc]["dict"]["vector"],
                                          [h_0],
                                          [c_0],
                                          [],
                                          children)
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
                    h, c = self._cell_forward(tree.nodes[p]["dict"]["vector"],
                                              [tree.nodes[c_]["dict"]["h"]
                                                  for c_ in c_local],
                                              [tree.nodes[c_]["dict"]["c"]
                                                  for c_ in c_local],
                                              [tree.nodes[c_]["dict"]["vector"]
                                                  for c_ in c_local],
                                              c_local)
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
