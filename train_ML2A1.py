## Model definition & training script ###
# Imports
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloader_ML2A1 import * 

# Initialise argument parser and define command line arguments.
parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)

# Must-have
parser.add_argument('-lg', '--languages', nargs='+', required=True,
                    help='Languages to train on. English | Thai')
parser.add_argument('-dpi', '--dpis', nargs='+', required=True,
                    help='DPI formats to train on. 200 | 300 | 400')
parser.add_argument('-ft', '--fonts', nargs='+', required=False,
                    help='Fonts to train on. normal|bold|italic|bold_italic')
# Optional
parser.add_argument('-ep', '--epochs', type=int, default=5,
                    help='Number of training epochs. Any.')
parser.add_argument('-bs', '--batch_size', type=int, default=32,
                    help='Size of training data batches. Any.')
parser.add_argument('-s', '--savefile', default=None,
                    help='Enable saving of model, specify filename/path.')
parser.add_argument('-srcd', '--source_dir',
                    default='/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/',
                    help='Pass a custom source directory pointing to image data.')

class CNN(nn.Module):
    def __init__(self, n_classes: int, img_dims: tuple[int, int], idx_to_char):
        super(CNN, self).__init__()
        # Initialise model params.
        self.input_size = img_dims[0]*img_dims[1]
        self.hsize_1 = int(self.input_size/2)
        self.output_size = n_classes
        self.idx_to_char = idx_to_char
        self.img_dims = img_dims  # used to resize test data
        
        # Define net structure.
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Flatten()
            )
        
        self.net2 = nn.Sequential(
            nn.Linear(self.input_size, self.hsize_1),
            #nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(self.hsize_1, self.output_size),
            nn.LogSoftmax(dim=1)
            )
        
    def forward(self, X: torch.Tensor, mode: str=None) -> torch.Tensor|str:
        # Reshape batched input and pass through for convolutional layer.
        net1_output = self.net1(X.reshape(1, X.shape[0], X.shape[1]*X.shape[2]))
        # Send output through classifier.
        preds = self.net2(net1_output.reshape(-1, self.input_size))

        if mode=='train':
            # Return LogSoftMax distribution over classes.
            return preds
        else:
            # Return predicted character.
            return self.idx_to_char(int(preds.argmax()))
        
def batch_data(data: OCRData, batch_size: int) -> list:
    """Shuffle training data again (unseeded) and create batches."""
    # Shuffle data.
    permutation = torch.randperm(data['imgs'].size()[0])
    permX = data['imgs'][permutation]
    permy = data['labels'][permutation]
    # Extract batches.
    batches = [(permX[i*batch_size:(i+1)*batch_size],
                permy[i*batch_size:(i+1)*batch_size])
            for i in range(int(data['imgs'].size()[0]/batch_size))]

    return batches
        
def train(data: DataLoader, device: torch.device, epochs: int,
          batch_size: int) -> CNN:
    """Train and return a CNN model."""
    # Transform & batch training data.
    print('Transforming data...')        
    train_data = OCRData(data.train, device, size_to=data.avg_size).transformed
    print('Batching data...')
    train_batches = batch_data(train_data, batch_size)

    # Batch training data, initialise model, optimizer, and loss function.
    model = CNN(data.n_classes, data.avg_size, data.idx_to_char).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.NLLLoss()
    
    print('Start training...')
    for epoch in range(epochs):
        # Print current epoch and reset total loss.
        print(f'\nepoch {epoch+1}')
        total_loss = 0
        # Iterate over training batches.
        for batch in tqdm(train_batches):
            # Reset gradient.
            optimizer.zero_grad()
            # Make predictions on batched data.
            pred = (model(batch[0], mode='train')).double()
            # Get gold labels.
            gold = batch[1]
            # Calculate + log loss, backpropagate, and update gradient.
            loss = loss_func(pred, gold)
            total_loss += loss
            loss.backward()
            optimizer.step()
            
        print(f'loss {total_loss}')
            
    return model

def init_train(src_dir: str, specs: dict, device: torch.device, epochs: int=5,
               batch_size: int=32, savefile: str=None) -> CNN:
    """Call functions to read & process training data, train (& save) a CNN."""
    # Read data from source directory & process it according to specs.
    print('\nSelecting files for training:', specs)
    datasets = DataLoader(src_dir, specs)
    
    # Train model.
    m = train(datasets, device, epochs, batch_size)
    print('Training complete.\n')
    
    # Save model to file.
    if savefile:
        torch.save(m, savefile)
        print('Model saved to', savefile)
    
    return m

if __name__=="__main__":
    # Set device and default source directory.
    if torch.cuda.is_available():
        device = 'cuda:1'
        src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
    else:
        device = 'cpu'
        src_dir = '../ThaiOCR/ThaiOCR-TrainigSet/'
    
    # Get specifcations for training data from argparse.
    args = parser.parse_args()
    specs = {'languages': args.languages, 'dpis': args.dpis, 'fonts': args.fonts}
    
    # Initiate training.
    model = init_train(src_dir, specs, device, args.epochs, args.batch_size,
                       args.savefile)