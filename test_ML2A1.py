### testing script ###
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from dataloader_ML2A1 import *
from train_ML2A1 import *

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
parser.add_argument('-ld', '--load', default=None,
                    help='Specify filename/path to load pretrained model from.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Pass to receive per-class evaluation metrics '
                    + 'and lowest performing classes')
parser.add_argument('-srcd', '--source_dir',
                    default='/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/',
                    help='Pass a custom source directory to read image data from.')


## testing ##
def test(data: DataLoader, model: CNN, device: torch.device):
    # Transform test data.
    print('Transforming test data...')        
    test_data = OCRData(data.test, device, size_to=model.img_dims).transformed

    print('Testing model...')
    model.eval()
    X = test_data['imgs']
    
    y_true = [data.idx_to_char(label) for label in test_data['labels']]

    y_preds = [model(X[i].reshape(1, X[i].shape[0], X[i].shape[1]))
               for i in range(len(X))]
        
    return y_preds, y_true

def eval(y_preds, y_true, verbose=False):
    # accuracy
    accuracy = accuracy_score(y_true, y_preds)
    # precision
    class_labels = list(set(y_true))
    
    precision = precision_score(y_true, y_preds,labels=class_labels, average=None,
                                zero_division=0.0)
    # recall
    recall = recall_score(y_true, y_preds, labels=class_labels, average=None)
    # F1
    f1 = f1_score(y_true, y_preds, labels=class_labels, average=None)
    
    measures = ['Precision', 'Recall', 'F1-score']
    evals = pd.DataFrame((precision, recall, f1), 
                                  index=measures,
                                  columns=class_labels)
    evals['MACROS'] = [round(sum(vals)/len(vals), 4)
                       for vals in (precision, recall, f1)]
    
    print('-'*80)
    print('Evaluation')
    print('\nOverall accuracy:', round(accuracy, 2))
    if verbose:
        print('\nPer-class measures')
        print(evals.transpose().round(2))
        print('-'*80)
        df = evals.transpose()    
        for measure in measures:
            perc25 = round(df.describe().loc['25%'].loc[measure], 4) 
            min_val = perc25 if perc25 < 0.6 else 0.6
            print(f'\n{measure} score below {min_val}:')
            print(df[df[measure] < min_val][measure])
    else:
        print('\nPerformance across all classes')
        print(evals['MACROS'])
        
def get_alt_train_specs():
    lg_read = {'1': 'English', '2': 'Thai'}
    dpi_read = {'1': '200', '2': '300', '3': '400'}
    ft_read = {'1': 'normal', '2': 'bold', '3': 'italic', '4': 'bold_italic'}
    
    print('Choose specifications for training data, enter single number, '
          +'or combination (e.g. 1 -> English; 12 -> English+Thai).')
    
    # lg_idx = input(f'Train on which languages?\n{lg_read}\n')
    lg = [lg_read[idx] for idx 
          in input(f'Train on which languages?\n{lg_read}\n')]
    
    #dpi_idx = input(f'Train on which resolution?\n{dpi_read}\n')
    dpi = [dpi_read[idx] for idx 
           in input(f'Train on which resolution?\n{dpi_read}\n')]
    
    #ft_idx = input(f'Train on which fonts?\n{ft_read}\n')
    ft = [ft_read[idx] for idx 
          in input(f'Train on which fonts?\n{ft_read}\n')]
    
    specs = {'languages': lg, 'dpis': dpi, 'fonts': ft}
    
    return specs
        
if __name__=="__main__":
     # Set device and default soruce directory
    if torch.cuda.is_available():
        device = 'cuda:1'
        src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
    else:
        device = 'cpu'
        src_dir = '../ThaiOCR/ThaiOCR-TrainigSet/'
        
    # Get specifcations for training data from argparse.
    args = parser.parse_args()
    specs = {'languages': args.languages, 'dpis': args.dpis, 'fonts': args.fonts}

    if args.load:
        # try to load model
        print('Loading model from', args.load)
        m = torch.load(args.load, weights_only=False)
    else:
        # get info to train new model
        if input('No pre-trained model found, train new model?\n(y/n) ') == 'y':            
            if input(f'Train new model on same specifications as test data?\n{specs}\n(y/n) ') == 'y':
                keep_specs = True
                train_specs = specs
            else:
                keep_specs = False
                train_specs = get_alt_train_specs()

            if input('Keep default params for epochs(5)/batch_size(32)/savefile(None)?\n(y/n) ') =='y':
                m, datasets = init_train(src_dir, train_specs, device) 
            else:
         ## TODO dummy proof these inputs, so u can pass none, and it doesnt fail due to the 
        # attempted int() conversion
                epochs = int(input('Number of epochs: '))
                b_s = int(input('Size of batches: '))
                save = input('File/pathname to save model to: ')
                m, datasets = init_train(src_dir, train_specs, device, epochs, b_s, save)

        else:
            print('Test script exited.')

    ## testing ##
    if keep_specs == True:
        # use test data from datasets created with train_specs
        preds, gold = test(datasets, m, device)

    else:
        print('Selecting files for testing:', specs)
        datasets = DataLoader(src_dir, specs)
        preds, gold = test(datasets, m, device)

        
    eval(preds, gold, verbose=args.verbose)
