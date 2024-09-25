### testing script ###
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
# precision, recall, F1 (harmonic mean of precision and recall), and accuracy.
# You can also include any other statistics or useful analysis output you feel
# like.

## testing ##
def test(data, model, verbose=False):
    model.eval()
    
    X = data.test['imgs']
    y_true = [data.idx_to_char(label) for label in data.test['labels']]

    y_preds = [model(X[i].reshape(1, X[i].shape[0], X[i].shape[1])) for i in range(len(X))]
    
    # redo metrics according to assignment
    # precision
    class_labels = list(set(y_true))
    precision = precision_score(y_true, y_preds,labels=class_labels,  average=None)  #micro or macro?
    # recall
    recall = recall_score(y_true, y_preds, labels=class_labels, average=None)
    # F1
    f1 = f1_score(y_true, y_preds, labels=class_labels, average=None)

    # accuracy
    
    evals = pd.DataFrame((precision, recall, f1), 
                                  index=["precision", "recall", "f1-score"],
                                  columns=class_labels)
    evals['MACROS'] = [round(sum(vals)/len(vals), 4) for vals in (precision, recall, f1)]
    if verbose:
        print(evals)
    else:
        print(evals['MACROS'])
