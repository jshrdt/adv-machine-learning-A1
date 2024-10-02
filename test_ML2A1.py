## Testing script ##
# Imports
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score)
from dataloader_ML2A1 import DataLoader, OCRData, OCRModel, parse_input_args
from train_ML2A1 import init_train


def get_model(loadfile: str, test_specs: dict) -> OCRModel:
    """Return model to be tested. Loaded from file or newly trained."""
    keep_dataset = None
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
        if input('No model loaded, train new model?\n(y/n) >> ') == 'y':
            # Get info on savefile.
            save = input('\nFile/pathname to save model to:\n(None|str) >> ')

            # Get specifications on what training data to use for new model.
            if input('\nTrain new model on same specifications as test data?'
                    +f'\n{specs}\n(y/n) >> ') == 'y':
                # Train on same data specs as used in testing.
                train_specs = test_specs
                keep_dataset = True
            else:
                # Get new train specs.
                train_specs = get_new_train_specs()

            # Get info on params for training loop.
            if input('\nKeep defaults for epochs (20) | batch_size (128) | '
                    + 'learningrate (0.0025)?\n(y/n) >> ') == 'y':
                m, test_data = init_train(src_dir, train_specs, device,
                                          savefile=save, mode='test')
            else:
                epochs = input('\nNumber of epochs:\n(None|int) >> ')
                epochs = int(epochs) if epochs else 20
                b_s = input('\nSize of batches:\n(None|int) >> ')
                b_s = int(b_s) if b_s else 64
                lr = input('\nLearning rate:\n(None|float) >> ')
                lr = float(lr) if lr else 0.001
                m, test_data = init_train(src_dir, train_specs, device, epochs,
                                          b_s, lr, savefile=save, mode='test')
        else:
            print('Test script exited.')

    return m, test_data, keep_dataset


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


def test(data: DataLoader, model: OCRModel, device: torch.device) -> \
        tuple[list, list]:
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
    precision = precision_score(y_true, y_preds, labels=pred_labels,
                                average=None, zero_division=0.0)
    recall = recall_score(y_true, y_preds, labels=pred_labels, average=None)
    f1 = f1_score(y_true, y_preds, labels=pred_labels, average=None)
    accuracy = accuracy_score(y_true, y_preds)

    # Sort measures into df, calculate macros.
    measures = ['Precision', 'Recall', 'F1-score']
    evals = pd.DataFrame((precision, recall, f1), index=measures,
                         columns=pred_labels).transpose()
    # Print evaluation.
    print('\n', '-'*80)
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
        device = 'cuda:3'
        src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
    else:
        device = 'cpu'
        src_dir = './ThaiOCR/ThaiOCR-TrainigSet/'

    # Get specifications for testing from argparse.
    args = parse_input_args(mode='test').parse_args()
    specs = {'languages': args.languages, 'dpis': args.dpis,
             'fonts': args.fonts}

    # Get model (either load from file, or train from scratch).
    model, train_specs_dataset, keep_dataset = get_model(args.loadfile, specs)

    # Get test data, test & evaluate model.
    if keep_dataset:
        test_data = train_specs_dataset
    else:
        print('\nSelecting files for testing:', specs)
        test_data = DataLoader(src_dir, specs)

    preds, gold = test(test_data, model.eval(), device)
    evaluate(preds, gold, verbose=args.verbose)
