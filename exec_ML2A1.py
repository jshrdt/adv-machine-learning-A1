from dataloader_ML2A1 import *
from train_ML2A1 import *
from test_ML2A1 import *
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# train and test script are gonna b separate i think
# called in pipe like as: testscript with specs | using model trained on these train specs?

#TBD sys args
#src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
src_dir = '../ThaiOCR/ThaiOCR-TrainigSet/'

train_specs = {'Language(s)': ['English'], 'DPI': ['200', '300'], 'Font(s)': ['normal']}
test_specs = {'Language(s)': ['English'], 'DPI': ['200'], 'Font(s)': ['normal']}

#est_specs=False
# ? call with optional separate test set specs?, like in func call keyword

### LOAD DATA ###
# Get relevant filenames according to specs
data = DataLoader(src_dir, train_specs, device, limit=None)
batch_size=64
# if test_specs:
#     train_files = train_fnames
#     test_files = load_data(src_dir, test_specs)
# else:
#     train_files, dev_files, test_files = split_data(train_fnames)


## TBD LIST ##
# test script
# parse args

## get clarification for 1 input vs batching; and output format
# ?The model trained will have as input a single image in the format of the dataset
# and produce as output a character corresponding to the identifiers in the ThaiOCR
# dataset or corresponding Unicode values.?


## training ##

# belongs to test script?
load=True  #sys arg for test script
savefile='modelsep25-cpu'  #sys arg for train script, else doesnt save
#'modelsep25'
# rewrite to check for default model name file with isfile to prio loading over new training
if load:
    # load model
    print('Loading model from', savefile)
    m = torch.load(savefile, weights_only=False)
    m.eval()
else:
    m = train(data, 2, device, batch_size, savefile=savefile)

## testing ##
test_data = DataLoader(src_dir, test_specs, device, size_to=m.img_dims)
test(data, m, verbose=False)
 

