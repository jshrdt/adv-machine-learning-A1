## model definition & training script ###
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader_ML2A1 import * 

parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)

# must have
parser.add_argument('-lg', '--languages', nargs='+', required=True,
                    help='Languages to train on. English | Thai')
parser.add_argument('-dpi', '--dpis', nargs='+', required=True,
                    help='DPI formats to train on. 200 | 300 | 400')
parser.add_argument('-ft', '--fonts', nargs='+', required=False,
                    help='Fonts to train on. normal|bold|italic|bold_italic')
# optional
parser.add_argument('-ep', '--epochs', type=int, default=5,
                    help='Number of training epochs. Any.')
parser.add_argument('-bs', '--batch_size', type=int, default=32,
                    help='Size of training data batches. Any.')
parser.add_argument('-s', '--savefile', default=None,
                    help='Enable saving of model, specify filename/path.')
parser.add_argument('-srcd', '--source_dir',
                    default='/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/',
                    help='Pass a custom source directory pointing to image data.')

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
        
        # define net structure
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Flatten())
        
        self.net2 = nn.Sequential(
            nn.Linear(self.input_size, self.hsize_1),
            nn.Tanh(),
            nn.Linear(self.hsize_1, self.hsize_2),
            nn.Tanh(),
            nn.Linear(self.hsize_2, self.output_size))
        
    def forward(self, input_x, mode=None):
        output = self.net1(torch.Tensor(input_x).reshape(1, input_x.shape[0],
                                        input_x.shape[1]*input_x.shape[2]))
        preds = self.net2(output.reshape(-1, self.input_size))

        if mode=='train':
            return preds
        else:
            return self.idx_to_char(int(preds.argmax()))
        
## training loop ##
def train(data, device, epochs, batch_size):    
    print('Start training...')
    
    train_batches = MyBatcher(data.train, batch_size).batches
    # in training: add image flips???
    n_classes = data.n_classes

    # initialise model and TODO
    model = CNN(n_classes, data.avg_size, data.idx_to_char).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        print(f'\nepoch {epoch+1}')
        # Reset loss for epoch
        total_loss = 0
                
        for batch in tqdm(train_batches):
            # Reset gradient.
            optimizer.zero_grad()
            # Make prediction
            pred = (model(batch[0], mode='train')).double()
            # Get gold label.
            gold = batch[1].to(device)
            # Calculate+log loss, backpropagate, and update gradient.
            loss = loss_func(pred, gold)
            total_loss += loss
            loss.backward()
            optimizer.step()
            
        print(f'loss {total_loss}')
        
    print('Training complete.')
    return model

def init_train(src_dir, specs, device, eps=5, b_s=32, savefile=None):
    # Read data & process data accordings to specs from source directory
    print('Selecting files for training:', specs)
    data = DataLoader(src_dir, specs, device)

    # Train model
    m = train(data, device, eps, b_s)
    
    if savefile:
        torch.save(m, savefile)
        print('Model saved to ', savefile)
    
    return m
    

if __name__=="__main__":
    # defaults
    src_dir_cpu = '../ThaiOCR/ThaiOCR-TrainigSet/'
    src_dir_gpu = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    src_dir = src_dir_gpu if torch.cuda.is_available() else src_dir_cpu

    # Get specifcations for training data from argparse
    args = parser.parse_args()
    specs = {'languages': args.languages, 'dpis': args.dpis, 'fonts': args.fonts}
    
    # begin training
    model = init_train(src_dir, specs, device, args.epochs, args.batch_size,
                       args.savefile)
    

