import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import networkx as nx
#
# Builds a torch dataset of networkx dependency trees from a list of phrases.
# It relies on a spacy model for tokenization, parsing and word embedding
#


class SentencePairDataset(data.Dataset):
    #
    # spacy vectors are numpy float32 (https://spacy.io/api/token)
    #
    def __init__(self, file_name, format_="SICK", nlp=None, dtype=torch.float32):
        if format_ == "SICK":
            df = pd.read_csv(filepath_or_buffer=file_name,
                             header=0, delimiter="\t")
        else:
            df = None
        self.tag_labels = dict([(l, i) for i, l in enumerate(
            [m for m in nlp.pipeline if m[0] == "tagger"][0][1].labels)])
        self.num_tag_labels = len(
            [m for m in nlp.pipeline if m[0] == "tagger"][0][1].labels)
        self.dep_labels = dict([(l, i) for i, l in enumerate(
            [m for m in nlp.pipeline if m[0] == "parser"][0][1].labels)])
        self.num_dep_labels = len(
            [m for m in nlp.pipeline if m[0] == "parser"][0][1].labels)
        self.dtype = dtype
        self.out = []
        self.embed_dim = 0
        for row in df.iterrows():
            sentence_A = row[1]["sentence_A"]
            sentence_B = row[1]["sentence_B"]
            tree_A = self._get_tree(sentence_A, nlp)
            if len(self.out) == 1:
                self.embed_dim = tree_A.nodes[0]["dict"]["vector"].shape[0]
            tree_B = self._get_tree(sentence_B, nlp)
            if not tree_A is None and not tree_B is None:
                self.out.append([tree_A, tree_B, float(row[1]["relatedness_score"]), row[1]
                                 ["entailment_label"], row[1]["entailment_AB"], row[1]["SemEval_set"]])
                print("Processed", row[1]["pair_ID"])
            else:
                print("Excluding sentence pair, too many sentences",
                      row[1]["pair_ID"])

    #
    # Returns tree from parent to child
    #
    def _label_encoder(self, dict_labels, label):
        out = torch.zeros(len(dict_labels), dtype=self.dtype)
        if label in dict_labels.keys():
            out[dict_labels[label]] = 1.0
        return out

    def _get_tree(self, sentence, nlp):
        doc = nlp(sentence)
        if not doc.has_vector:
            return None
        if len(list(doc.sents)) > 1:
            return None
        average_vector = torch.from_numpy(doc.vector)
        G = nx.DiGraph()
        for token in doc:
            v_pos = self._label_encoder(self.tag_labels, token.tag_)
            v_dep = self._label_encoder(self.dep_labels, token.dep_)
            if token.has_vector:
                v = torch.cat(
                    [torch.from_numpy(token.vector), v_pos, v_dep], axis=0)
            else:
                v = torch.cat([average_vector, v_pos, v_dep], axis=0)
            #Root
            if token.head == token:
                G.add_node(token.i, dict={"vector": v})
            else:
                G.add_edge(token.head.i, token.i)
                G.add_node(token.i, dict={"vector": v})
        return G

    def __len__(self):
        return len(self.out)

    def __getitem__(self, idx):
        return self.out[idx]
