## bonus part ##
# Imports
from PIL import Image
import numpy as np
import torch
import pandas as pd
from dataloader_ML2A1 import OCRData

# proper solution would be something like this with openCV, https://stackoverflow.com/questions/57249273/how-to-detect-paragraphs-in-a-text-document-image-for-a-non-consistent-text-stru


def get_blocks(fname):
    #fname = './jn_002sb copy.bmp'
    img = np.array(Image.open(fname).convert('1'))  #array of True/False pixels of document
    bin_img = np.array([[0 if dot else 1 for dot in line] for line in img])  #convert to 1/0 vals
    #print(Image.open(fname).convert('1').getbbox())
    
    space_len = 0
    blocks = list()
    row_split = np.split(bin_img, np.where(bin_img.sum(axis=1)==0)[0])
    
    block = list()

    for unit in row_split:
        if unit.shape[0] > 1:  # is non-empty
            # unit is beginning/top of line
            # check: is previous space_len indiciative of space between blocks
            if space_len > 30:
                blocks.append(block)
                block = list()
            block.append(unit)
            space_len = 0

        else:
            space_len += 1
    if block:
        blocks.append(block)
        
    
    # split into lines, by using empty lines as dividers
    #lines = [line for line in np.split(bin_img, np.where(bin_img.sum(axis=1)==0)[0])
     #        if sum(line.sum(axis=1)) != 0]
    
    return blocks
    

def get_words(lines):
    
    # split lines into character & try to intuit word borders by using empty columns
    # crop empty lines around char but one, get list of probable words
    words_in_line = list()

    for line in lines:
        words = list()

        col_split = np.hsplit(line, np.where(line.sum(axis=0)==0)[0])
        space_len = 0
        word = list()
        # get characters and guesstimate whitespaces
        for unit in col_split:
            if unit.shape[1] > 1:  # is character   
                # first: check if is beginning of word
                # aka: is previous space_len a whitespace?
                if space_len > unit.shape[1]/3:  # whitespace happened
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
        words_in_line.append(words)
        
        

    return words_in_line
        
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
        model = torch.load('bonus_alldirs_allstyles', weights_only=False).eval()

    else:
        device = 'cpu'
        src_dir = './ThaiOCR/ThaiOCR-TrainigSet/'
        model = torch.load('bonus_model_all', weights_only=False,
                       map_location=torch.device('cpu')).eval()


    #fname = './bc_002sb copy.bmp' #dummyfile
    #fname = './ThaiOCR/ThaiOCR-TestSet/Journal/Image/200dpi_BW/jn_002sb.bmp'
    fname= './ThaiOCR/ThaiOCR-TestSet/Journal/Image/300dpi_BW/jn_002tb.bmp'


    # read img, get list of paragraph blocks
    blocks = get_blocks(fname)
    #print(len(blocks))
    
    for block in blocks:
        # print(len(block))
        block_pred = list()
        
        # get list of words in block, as isolated characters
        words_in_line = get_words(block)

        # use OCRData class to resize & rescale isolated characters to suited input
        # format for pre trained model
        for line in words_in_line:
            pred_words = list()

            for word in line:
                #print(word, len(word))
                asOCR = OCRData(pd.DataFrame(word, columns=['files', 'labels']),
                            device, size_to=model.img_dims, mode='bonus')
            
                pred_words.append(test(asOCR, model))
            block_pred.append((' '.join([''.join(word) for word in pred_words])))
            #print((' '.join([''.join(word) for word in pred_words])))
        print(block_pred)
            
        print()
    
    
    goldf = ['./ThaiOCR/ThaiOCR-TestSet/Journal/Txt/jn_002z2.txt', 
             './ThaiOCR/ThaiOCR-TestSet/Journal/Txt/jn_002z3.txt',
             './ThaiOCR/ThaiOCR-TestSet/Journal/Txt/jn_002z4.txt',
             './ThaiOCR/ThaiOCR-TestSet/Journal/Txt/jn_002z5.txt']
    
    gold = list()
    
    for fname in goldf:
        with open(fname, 'rb') as f:
            text =  f.read().decode(encoding='cp874', errors='backslashreplace')
            block_pred = text.replace('\r', '').split('\n')
        gold.append(block_pred)
   # print(gold[3])
    
    # TBD train model on numeric+special character data from trainig set as well
        
