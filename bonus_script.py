## bonus part ##
# Imports
from PIL import Image
import numpy as np

from test_ML2A1 import *

def get_words(fname):
    img = np.array(Image.open(fname).convert('1'))  #array of True/False pixels of document
    bin_img = np.array([[0 if dot else 1 for dot in line] for line in img])  #convert to 1/0 vals
    
    # split into lines, by using empty lines as dividers
    lines = [line for line in np.split(bin_img, np.where(bin_img.sum(axis=1)==0)[0])
             if sum(line.sum(axis=1)) != 0]
    
    # split lines into character & try to intuit word borders by using empty columns
    # crop empty lines around char but one, get list of probable words
    words = list()

    for line in lines:
        col_split = np.hsplit(line, np.where(line.sum(axis=0)==0)[0])
        space_len = 0
        word = list()
        # get characters and guesstimate whitespaces
        for unit in col_split:
            if unit.shape[1] > 1:  # is character   
                # first: check if is beginning of word
                # aka: is previous space_len a whitespace?
                if space_len > unit.shape[1]/2:  # whitespace happened
                    # if yes: add previous word to wordlist, reset word
                    if len(word) != 0:  # ignore initial trailing whitespace
                        #print(len(word))
                        words.append(word)
                        word = list()
                # always: 
                # crop & add current char unit to current word & reset whitespace
                word.append(crop_char(unit))
                space_len = 0
            # current unit was not character = empty line, add to whitespace len
            else:
                space_len+=1
                
        # at end of line: check if theres sth in word (cant be followed by whitepace),
        # add to wordlist
        if word:
            words.append(word)

    return words
        
def crop_char(char):
    cropped_char_ver = np.array([row for row in char if row.sum()!=0])
    cropped_char = np.transpose(np.array([row for row in np.transpose(cropped_char_ver)if row.sum()!=0 ]))
    pad = np.zeros(cropped_char.shape[1]).reshape(1, cropped_char.shape[1])    
    padded_char = np.concatenate((pad, cropped_char, pad))
    return (cropped_char, 999)  # dummylabel 999

def test(data, model):
    X = data.transformed['imgs']
    y_preds = list()
    for i in (range(len(X))):
        y_preds.append(model(X[i].reshape(1, X[i].shape[0], X[i].shape[1])))
    return y_preds

if __name__=="__main__":
    # Set device, load model
    if torch.cuda.is_available():
            device = 'cuda:1'
            src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
    else:
        device = 'cpu'
        src_dir = './ThaiOCR/ThaiOCR-TrainigSet/'
    
    # model = torch.load('En_200_normal', weights_only=False).eval()
    model = torch.load('bonus_alldirs_allstyles', weights_only=False,
                       map_location=torch.device('cpu')).eval()

    fname = './bc_002sb copy.bmp' #dummyfile
    #fname = './ThaiOCR/ThaiOCR-TestSet/Journal/Image/200dpi_BW/jn_002sb.bmp'
    # read img, get isolated chars
    words = get_words(fname)
    
    # for w in words:
    #     print('\nis word')
    #     print(w)

    # use OCRData class to resize & rescale isolated characters to suited input
    # format for pre trained model
    pred_words = list()
    for word in words:
        #print(word, len(word))
        asOCR = OCRData(pd.DataFrame(word, columns=['files', 'labels']),
                    device, size_to=model.img_dims, mode='bonus')
    
        pred_words.append(test(asOCR, model))
    print((' '.join([''.join(word) for word in pred_words])))
    
    # TBD train model on numeric+special character data from trainig set as well
        
