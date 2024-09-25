### testing script ###
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
# precision, recall, F1 (harmonic mean of precision and recall), and accuracy.
# You can also include any other statistics or useful analysis output you feel
# like.
from ML2A1_2_helper import *
from ML2A1_train import *


## testing ##
def test(data, model, verbose=False):
    ## TBD load data in here
    
    model.eval()
    
    X = data.test['imgs']
    y_true = [data.idx_to_char(label) for label in data.test['labels']]

    y_preds = [model(X[i].reshape(1, X[i].shape[0], X[i].shape[1])) for i in range(len(X))]
    
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
    evals['MACROS'] = [round(sum(vals)/len(vals), 4) for vals in (precision, recall, f1)]
    
    print('-'*80)
    print('Evaluation')
    print('\nOverall accuracy:', accuracy)
    if verbose:
        print('\nPer-class measures')
        print(evals.transpose().round(2))
        print('-'*80)
        df = evals.transpose()    
        for measure in measures:
            min_val = round(df.describe().loc['25%'].loc[measure], 4)
            print(f'\n{measure} performance below 25th percentile ({min_val}):')
            print(df[df[measure] <= min_val][measure])
    else:
        print('\nPerformance across all classes')
        print(evals['MACROS'])
        
   
    
    
    #print(evals.min( axis='columns' ))
    # print(df.idxmin().loc[df.min().idxmin()], df.min().idxmin() )

# if __name__=="__main__":
#     train_specs = {'Language(s)': ['English'], 'DPI': ['200'], 'Font(s)': ['normal']}
#     src_dir = '../ThaiOCR/ThaiOCR-TrainigSet/'
#     savefile = 'modelsep25'
#     data = DataLoader(src_dir, train_specs, limit=5000)

#     m = CNN(data.n_classes, data.avg_size, data.idx_to_char)
#     m.load_state_dict(torch.load(savefile, weights_only=True))
#     m.eval()