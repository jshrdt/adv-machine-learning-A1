## Bonus part ##
# Imports
from PIL import Image
import numpy as np
import torch
import pandas as pd
from dataloader_ML2A1 import OCRData

def get_blocks(fname):
    """Return list of probable paragraph blocks contained in image."""
    # Open iamge as black & white, convert True/False encoding to 1/0 (1=black).
    img = np.array(Image.open(fname).convert('1'))  #array of True/False pixels of document
    bin_img = np.array([[0 if dot else 1 for dot in line] for line in img])  #convert to 1/0 vals

    # Split along all empty row.
    row_split = np.split(bin_img, np.where(bin_img.sum(axis=1)==0)[0])
    space_len = 0  # Count successive empty rows.
    # Collect rows in paragraph, and complete paragraphs.
    paragraph, blocks = list(), list()
    # Try to find paragraph borders...
    for row in row_split:
        # Row is non-empty
        if row.shape[0] > 1:
            # unit is beginning/top of text line
            # check: is previous space_len indiciative of space between blocks
            if space_len > 30:
                blocks.append(paragraph)
                paragraph = list()
            paragraph.append(row)
            space_len = 0  # reset consecutie space counter
        # Empta row
        else:
            space_len += 1
    # Save document final paragraph
    if paragraph:
        blocks.append(paragraph)

    return blocks


def get_words(lines):
    """Isolate words and characters for lines in a paragraph."""
    # Collect lines (with character segmented words) in paragraph
    words_in_line = list()
    for line in lines:
        # Reset list of words for current line.
        words = list()
        # Split line across empty columns.
        col_split = np.hsplit(line, np.where(line.sum(axis=0)==0)[0])
        space_len = 0  # count consecutive empty lines
        # get single word as list of its characters, suitable for model input
        word = list()
        # get characters and guesstimate whitespaces
        for unit in col_split:
            if unit.shape[1] > 1:  # is character
                # first: check if it's the beginning of word, i.e. is current
                # space_len indicative of being whitespace?
                if space_len > unit.shape[1]/3:  # guess minimum whitespace size
                    # if yes: add previous word to wordlist, reset word
                    if len(word) != 0:  # ignore initial trailing whitespace
                        # update word list for line, reset word
                        words.append(word)
                        word = list()
                # always: 
                # crop & add current char unit to current word & reset whitespace
                word.append(crop_char(unit))
                space_len = 0
            # current unit was not character = empty line, add to whitespace len
            else:
                space_len+=1
        # save line-final word
        if word:
            words.append(word)
        # add words in current line to list collecting lines in paragraph
        words_in_line.append(words)

    return words_in_line


def crop_char(char):
    """Crop empty rows/columns around character unit."""
    cropped_top = np.array([row for row in char if row.sum()!=0])
    cropped_sides = [row for row in np.transpose(cropped_top) if row.sum()!=0 ]
    cropped_char = np.transpose(np.array(cropped_sides))
    # padding scrapped due to hurting performance
    #pad = np.zeros(cropped_char.shape[1]).reshape(1, cropped_char.shape[1])
    #padded_char = np.concatenate((pad, cropped_char, pad))
    return (cropped_char, 999)  # dummylabel 999 fr compatibility with OCRData

def test(data, model):
    """Return predictions on character basis."""
    X = data.transformed['imgs']
    y_preds = list()
    for i in (range(len(X))):
        y_preds.append(model(X[i].reshape(1, X[i].shape[0], X[i].shape[1])))
    return y_preds


if __name__=="__main__":
    # Set device, load model
    modelfile = 'bonus_model_all'
    if torch.cuda.is_available():
        device = 'cuda:1'
        src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
        model = torch.load(modelfile, weights_only=False).eval()
    else:
        device = 'cpu'
        src_dir = './ThaiOCR/ThaiOCR-TrainigSet/'
        model = torch.load(modelfile, weights_only=False,
                       map_location=torch.device('cpu')).eval()

    fname = './ThaiOCR/ThaiOCR-TestSet/Journal/Image/200dpi_BW/jn_002sb.bmp'
    #fname= './ThaiOCR/ThaiOCR-TestSet/Journal/Image/300dpi_BW/jn_002tb.bmp'

    # Read img, get list of paragraph blocks
    blocks = get_blocks(fname)

    # make & print predicitons per paragraph
    print('-'*50)
    print('Prediction from script.\n\n')
    for block in blocks:
        block_pred = list()
        # get list of words in block as isolated characters
        words_in_line = get_words(block)
        for line in words_in_line:
            pred_words = list()
            for word in line:
                # use OCRData class to resize & rescale isolated characters to suited input
                # format for pre trained model
                asOCR = OCRData(pd.DataFrame(word, columns=['files', 'labels']),
                            device, size_to=model.img_dims, mode='bonus')
                pred_words.append(test(asOCR, model))
            block_pred.append((' '.join([''.join(word) for word in pred_words])))
        print(block_pred)

    goldf = ['./ThaiOCR/ThaiOCR-TestSet/Journal/Txt/jn_002z2.txt', 
             './ThaiOCR/ThaiOCR-TestSet/Journal/Txt/jn_002z3.txt',
             './ThaiOCR/ThaiOCR-TestSet/Journal/Txt/jn_002z4.txt',
             './ThaiOCR/ThaiOCR-TestSet/Journal/Txt/jn_002z5.txt']

    print('-'*50)
    print('Gold output from txt file\n\n')
    gold_txt = list()
    for fname in goldf:
        with open(fname, 'rb') as f:
            text =  f.read().decode(encoding='cp874', errors='backslashreplace')
            block_gold = text.replace('\r', '').split('\n')
        gold_txt.append(block_gold)
    for gold in gold_txt:
        print(gold)
    
    # TBD train model on numeric+special character data from trainig set as well
        
