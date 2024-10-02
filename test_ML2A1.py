### Testing script ###
# Imports.
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score)

from dataloader_ML2A1 import DataLoader, OCRData, OCRModel
from train_ML2A1 import init_train

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
parser.add_argument('-ld', '--loadfile', default=None,
                    help='Specify filename/path to load pretrained model from.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Pass to receive per-class evaluation metrics '
                    + 'and lowest performing classes')
parser.add_argument('-srcd', '--source_dir',
                    default='/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/',
                    help='Pass a custom source directory to read image data from.')

def get_model(loadfile: str, test_specs: dict) -> OCRModel:
    """Return model to be tested. Loaded from file or newly trained."""
    # Load model, get test data according to specs.
    if loadfile:
        print('Loading model from: ', loadfile)
        if not torch.cuda.is_available():
            m = torch.load(loadfile, weights_only=False,
                           map_location=torch.device('cpu'))
        else:
             m = torch.load(loadfile, weights_only=False)
    # Get info to train new model.
    else:
        print('-'*80)
        if input('No pre-trained model found, train new model?\n(y/n) >> ') == 'y':
            # Get specifications on what training data to use for new model.          
            if input('\nTrain new model on same specifications as test data?'
                    +f'\n{specs}\n(y/n) >> ') == 'y':
                # Train on same data specs as used in testing.
                train_specs = test_specs
            else:
                # Get new train specs.
                train_specs = get_new_train_specs()
                
            # Get info on params/savefile for training loop.
            if input('\nKeep default for epochs (20) | batch_size (128) | learning'
                    + ' rate (0.0025) | savefile (None)?\n(y/n) >> ') == 'y':
                m = init_train(src_dir, train_specs, device) 
            else:
                epochs = input('\nNumber of epochs:\n(None|int) >> ')
                epochs = int(epochs) if epochs else 20
                b_s = input('\nSize of batches:\n(None|int) >> ')
                b_s = int(b_s) if b_s else 64
                lr = input('\nLearning rate:\n(None|float) >> ')
                lr = float(lr) if lr else 0.001
                save = input('\nFile/pathname to save model to:\n(None|str) >> ')
                m = init_train(src_dir, train_specs, device, epochs, b_s, lr,
                               save)
        else:
            print('Test script exited.')
            
    return m
 
def get_new_train_specs() -> dict:
    """Return separate training specifications for new model."""
    # Shorthand mappings.
    lg_read = {'1': 'English', '2': 'Thai'}
    dpi_read = {'1': '200', '2': '300', '3': '400'}
    ft_read = {'1': 'normal', '2': 'bold', '3': 'italic', '4': 'bold_italic'}
    print('-'*80)
    print('\nChoose specifications for training data.\nEnter single number, '
          +'or combination (e.g. 1 -> English; 12 -> English+Thai).')
    # Get user input for languages/resolution/fonts to train on.
    try:
        lg = [lg_read[idx] for idx 
            in input(f'\nTrain on which language(s)?\n{lg_read}\n>> ')]
        dpi = [dpi_read[idx] for idx 
            in input(f'\nTrain on which resolution(s)?\n{dpi_read}\n>> ')]
        ft = [ft_read[idx] for idx 
            in input(f'\nTrain on which font(s)?\n{ft_read}\n>> ')]
    except KeyError:
        print('Invalid keyboard input. Test script exited.\n')
    
    specs = {'languages': lg, 'dpis': dpi, 'fonts': ft}
    
    return specs

def test(data: DataLoader, model: OCRModel, device: torch.device) -> tuple[list, list]:
    """Test model on data, return model predictions and gold labels."""
    # Transform test data.
    print('Transforming test data...')        
    test_data = OCRData(data.test, device, size_to=model.img_dims).transformed

    # Make predictions & extract gold labels. Both converted to single character.
    print('Testing model...')
    X = test_data['imgs']
    y_true = [data.idx_to_char(label.cpu()) for label in test_data['labels']]
    y_preds = list()
    for i in tqdm(range(len(X))):
        y_preds.append(model(X[i].reshape(1, X[i].shape[0], X[i].shape[1])))
        
    return y_preds, y_true

def evaluate(y_preds: list, y_true: list, verbose: bool=False):
    """Evaluate and print model's performance."""
    # Set of predicted labels.    
    pred_labels = list(set(y_true))
    # Get precision, recall, f1, and overall accuracy measures.
    precision = precision_score(y_true, y_preds, labels=pred_labels, average=None,
                                zero_division=0.0)
    recall = recall_score(y_true, y_preds, labels=pred_labels, average=None)
    f1 = f1_score(y_true, y_preds, labels=pred_labels, average=None)
    accuracy = accuracy_score(y_true, y_preds)

    # Sort measures into df, calculate macros.
    measures = ['Precision', 'Recall', 'F1-score']
    evals = pd.DataFrame((precision, recall, f1), index=measures, 
                         columns=pred_labels).transpose()    
    # Print evaluation.
    print('-'*80)
    print('Evaluation')
    print('\nOverall accuracy:', round(accuracy, 2))

    if verbose:
        # Details on per-class measures.
        print('\nOverview of measures across classes:\n')
        print(evals.describe().round(2).iloc[1:])
        
        print('\nOverview of 5 worst performing classes per measure:')
        for measure in measures:
            print()
            print(measure)
            print(evals[measure].sort_values()[:5].round(2))
                        
    else:
        print('\nMean performances across all classes:\n')
        print(evals.describe().loc['mean'])
        
if __name__=="__main__":
    # Set device and default source directory.
    if torch.cuda.is_available():
        device = 'cuda:1'
        src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
    else:
        device = 'cpu'
        src_dir = './ThaiOCR/ThaiOCR-TrainigSet/'
        
    # Get specifications for testing from argparse.
    args = parser.parse_args()
    specs = {'languages': args.languages, 'dpis': args.dpis, 'fonts': args.fonts}

    # Get model (either loaded from file, or trained from scratch) & test data.
    model = get_model(args.loadfile, specs).eval()
    print('\nSelecting files for testing:', specs)
    test_data = DataLoader(src_dir, specs)

    # Test & evaluate model.
    preds, gold = test(test_data, model, device)
    evaluate(preds, gold, verbose=args.verbose)
