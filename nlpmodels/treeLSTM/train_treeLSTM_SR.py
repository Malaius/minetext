import torch.optim
from treeLSTM_SR import treeLSTM_SR
from torch.utils.data import DataLoader
from SentencePairDataset import SentencePairDataset
import pickle
import random
import pandas as pd

def _batch_list(list_, batch_size):
    list__=random.sample(list_, k=len(list_)) #To avoid shuffling in place
    out=[]
    j=0
    while j < len(list__):
        out.append(list__[j:j+batch_size])
        j += batch_size
    return out

with open(r"C:\Users\unhel\OneDrive\source\repos\treeLSTM\SICK_dataset.pickle",mode="br") as inmodel:
    ds=pickle.load(inmodel)
#Manual batching: DataLoader does not work with networkx objeccts
data_size=len(ds)
tag_dim=ds.num_tag_labels + ds.num_dep_labels
#n_train=int(0.8*data_size)
#idx_train=random.sample(list(range(data_size)), k=n_train)
#sidx_train=set(idx_train)
#Separate data for test
#idx_test=[i for i in list(range(data_size)) if not i in sidx_train]
scores_test=[ds[i][2] for i in range(data_size) if ds[i][5] == "TEST"]
trees_A_test = [ds[i][0] for i in range(data_size) if ds[i][5] == "TEST"]
trees_B_test = [ds[i][1]  for i in range(data_size) if ds[i][5] == "TEST"]
ds_train=[ds[i]  for i in range(data_size) if ds[i][5] == "TRAIN"]
print("Train dataset size", len(ds_train))
print("Test dataset size", len(scores_test))
idx_train=list(range(len(ds_train)))
#input_dim, hidden_dim, dropout=0.0, num_layers=1, dtype=torch.float32, scores=list(range(1,6)), attention=False, generate_query=False):
model = treeLSTM_SR(300,50, 150, dropout=0.1, num_layers=1, dtype=torch.float32, scores=list(range(1,6)), attention=False, generate_query=False)
for p in model.parameters():
    print("Parameter",p.shape)
#Weight decay is the L2 penalty, advisable in this case
optimizer = torch.optim.Adagrad(model.parameters(),lr=0.05, weight_decay=1.0e-4)
#optimizer = torch.optim.Adam(model.parameters(),lr=0.05, weight_decay=1.0e-4)
NEPOCHS=15
BATCH_SIZE=25

with torch.no_grad():
    #Stateless prediction
    print("Initial validation")
    loss_test, ranks_test,_,_,_,_=model(trees_A_test, trees_B_test, scores_test)
    df_scores=pd.DataFrame(data=zip([s for s in scores_test], [r.item() for r in ranks_test]), columns=["true","pred"])
    print("Pearson correlation.",df_scores.corr("pearson"))
    print("Spearman correlation.",df_scores.corr("spearman"))
    print("Kendall tau.",df_scores.corr("kendall"))
for epoch in range(NEPOCHS):
    batches=_batch_list(idx_train,BATCH_SIZE)
    for bidx, batch in enumerate(batches):
        trees_A= [ds_train[b][0] for b in batch]
        trees_B= [ds_train[b][1] for b in batch]
        scores= [ds_train[b][2] for b in batch]
        #if len(batch) == BATCH_SIZE: #Discard last batch for statefull mode...
            #Scores are expected to be Python float
        optimizer.zero_grad()
        loss, ranks, _, _, _, _ =model(trees_A, trees_B, scores) #Stateless
        loss.backward() #retain_graph=True) #Calculate gradients
        optimizer.step() #Optimizer step
        print("Epoch",epoch,"Batch",bidx, "loss", loss.item()) # "Pearson corr.",1.0 - antipearson.item())
    with torch.no_grad():
        #Statepless prediction
        loss_test, ranks_test,_,_,_,_=model(trees_A_test, trees_B_test, scores_test)
        df_scores=pd.DataFrame(data=zip([s for s in scores_test], [r.item() for r in ranks_test]), columns=["true","pred"])
        print("Pearson correlation.",df_scores.corr("pearson"))
        print("Spearman correlation.",df_scores.corr("spearman"))
        print("Kendall tau.",df_scores.corr("kendall"))
torch.save(model,"treeLSTM_NOAttention_SR_SICK_v3_epochs15_dropout01.pickle")