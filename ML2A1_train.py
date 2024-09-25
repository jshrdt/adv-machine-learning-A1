## model definition & training script ###
import torch
import torch.nn as nn
import torch.optim as optim
#import numpy as np
from tqdm import tqdm
from ML2A1_2_helper import * 

   
# class Batcher: ##?? replace with func
#     def __init__(self, data, device, batch_size=8, max_iter=None):
#         self.X = data['imgs']
#         self.y = data['labels']
#         self.device = device
#         self.batch_size = batch_size
#         self.max_iter = max_iter
#         self.curr_iter = 0
    
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.curr_iter == self.max_iter:
#             raise StopIteration
#         ## rethink ur sructure, this would need a tensor already
#         # either: shuffle when still as filename (ez shuffle, but have to restructure 
#         # OOP; or see that u get it as torch here? )
#         permutation = torch.randperm(self.X.size()[0], device=self.device)
#         print(permutation)
#         splitX = torch.split(self.X[permutation], self.batch_size)
#         splity = torch.split(self.y[permutation], self.batch_size)
#         self.curr_iter += 1
        
#         return splitX, splity

## define model structure ##
class CNN(nn.Module):
    def __init__(self, n_classes, img_dims, idx_to_char):
        super(CNN, self).__init__()
        # Initialise model params
        self.input_size = img_dims[0]*img_dims[1]
        self.hsize_1 = 200
        self.hsize_2 = 150
        self.output_size = n_classes
        self.img_dims = img_dims
        self.idx_to_char = idx_to_char
        #self.device = device #?
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1), #padding changes size! need to account for after flatten
            nn.Flatten())
        self.net2 = nn.Sequential(
            nn.Linear(self.input_size, self.hsize_1),
            nn.Tanh(),
            nn.Linear(self.hsize_1, self.hsize_2),
            nn.Tanh(),
            nn.Linear(self.hsize_2, self.output_size)
            )
        
    def forward(self, input_x, mode=None):
        output = self.net1(torch.Tensor(input_x).reshape(1, input_x.shape[0],
                                        input_x.shape[1]*input_x.shape[2]))
        preds = self.net2(output.reshape(-1, self.input_size))
        #raise ValueError
        # from vid: 
        # current_matrix = batch.permute(0,3,1,2) #need permute??
        # current_matrix = self.layers1(current_matrix)
        # output_matrix = self.layers2(current_matrix.reshape(-1, input_size))
        if mode=='train':
            return preds
        else:
            return self.idx_to_char(int(preds.argmax()))
        
## training loop ##
def train(data, epochs, device, batch_size, pre_model=None, savefile=False):
    ## TBD load data in here?
    
    print('Start training')
    
    train_batches = MyBatcher(data.train, batch_size, device).batches
    # # in training: add image flips???
    n_classes = data.n_classes
    
    if not pre_model:
        model = CNN(n_classes, data.avg_size, data.idx_to_char)
    else:
        model = pre_model
        
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()  #try with NLLLoss for logsoftmax?
    #loss_func = nn.NLLLoss()
    for e in range(epochs):
        print(f'\nepoch {e+1}')
        e_loss = 0
                
        for batch in tqdm(train_batches):
                    
            optimizer.zero_grad()
            
            # make prediction
            pred = (model(batch[0], mode='train')).double()
            
            # get gold label (tensor of len n_classes, one-hot for gold label idx)
            # gold = np.zeros(n_classes)            
            # gold[batch[1]] = float(1)
            # gold = torch.tensor(gold).reshape(1, n_classes)
            gold = batch[1]
            
            # calculate/log loss, backpropagate, update gradient
            loss = loss_func(pred, gold)
            e_loss += loss
            # total_loss += loss
            loss.backward()
            optimizer.step()
            
        print(f'loss {e_loss}')
    
    if savefile:
        torch.save(model, savefile)
       # torch.save(model.state_dict(), savefile)    
    
    return model
