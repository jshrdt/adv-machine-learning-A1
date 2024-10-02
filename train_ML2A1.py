## Training script ###
print('Compiling...')
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader_ML2A1 import DataLoader, OCRData, OCRModel, parse_input_args

def train(data: DataLoader, device: torch.device, epochs: int,
          batch_size: int, learning_rate: float) -> OCRModel:
    """Train and return a CNN for OCR.

    Args:
        data (DataLoader): Custom class containing mainly dataset splits
            and functions to de/encode gold labels.
        device (torch.device): CPU or GPU details to use for tensors/model.
        epochs (int): Number of epochs in training loop.
        batch_size (int): Number of characters in a single batch.
        learning_rate (float): Size of learning rate.

    Returns:
        OCRModel: CNN model for OCR, with label decoder function and expected
            image dimensions for input.
    """
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
               savefile: str = None, mode: str = None) -> OCRModel:
    """Read + process training data and train (& save) an OCRModel."""
    # Read data from source directory & process it according to specs.
    print('\nSelecting files for training:', specs)
    datasets = DataLoader(src_dir, specs)

    # Train model.
    m = train(datasets, device, epochs, batch_size, lr)

    # Save model to file.
    if savefile:
        torch.save(m, savefile)
        print('Model saved to', savefile, '\n')

    if mode=='test':
        return m, datasets
    else:
        return m


if __name__=="__main__":
    # Set device and default source directory.
    if torch.cuda.is_available():
        device = 'cuda:3'
        src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
    else:
        device = 'cpu'
        src_dir = './ThaiOCR/ThaiOCR-TrainigSet/'

    # Get specifcations for training data from argparse.
    args = parse_input_args().parse_args()
    if args.source_dir: src_dir = args.souce_dir
    specs = {'languages': args.languages, 'dpis': args.dpis,
             'fonts': args.fonts}

    # Initiate training, optionally saves model.
    model = init_train(src_dir, specs, device, args.epochs, args.batch_size,
                       args.learning_rate, args.savefile)