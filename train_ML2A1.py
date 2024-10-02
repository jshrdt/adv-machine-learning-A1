## Training script ###
print('Compiling...')
# Imports
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloader_ML2A1 import DataLoader, OCRData, OCRModel

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
parser.add_argument('-ep', '--epochs', type=int, default=20,
                    help='Number of training epochs. Any.')
parser.add_argument('-bs', '--batch_size', type=int, default=128,
                    help='Size of training data batches. Any.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025,
                    help='Learning rate to use during training. 0-1')
parser.add_argument('-s', '--savefile', default=None,
                    help='Enable saving of model, specify filename/path.')
parser.add_argument('-srcd', '--source_dir',
                    default='/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/',
                    help='Pass a custom source directory pointing to image data.')

        
def train(data: DataLoader, device: torch.device, epochs: int,
          batch_size: int, learning_rate: float) -> OCRModel:
    """Train and return a CNN for OCR."""
    # Transform & batch training data.
    print('Transforming data...')        
    train_data = OCRData(data.train, device, size_to=data.avg_size).transformed
    print('Batching data...')
    train_batches = batch_data(train_data, batch_size)
    
    # Initialise model, optimizer, and loss function.
    model = OCRModel(data.n_classes, data.avg_size,data.idx_to_char).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.NLLLoss()
    
    # Train model.
    print('Start training...')
    for epoch in range(epochs):
        # Print current epoch and reset total loss.
        print(f'\nepoch {epoch+1}')
        total_loss = 0

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
        
        
def init_train(src_dir: str, specs: dict, device: torch.device,
               epochs: int = 20, batch_size: int = 128, lr: float = 0.0025,
               savefile: str = None) -> OCRModel:
    """Read + process training data and train (& save) an OCRModel."""
    # Read data from source directory & process it according to specs.
    print('\nSelecting files for training:', specs)
    datasets = DataLoader(src_dir, specs)
    
    # Train model.
    m = train(datasets, device, epochs, batch_size, lr)
        
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
        src_dir = './ThaiOCR/ThaiOCR-TrainigSet/'
    
    # Get specifcations for training data from argparse.
    args = parser.parse_args()
    specs = {'languages': args.languages, 'dpis': args.dpis,
             'fonts': args.fonts}
    
    # Initiate training.
    model = init_train(src_dir, specs, device, args.epochs, args.batch_size,
                       args.learning_rate, args.savefile)