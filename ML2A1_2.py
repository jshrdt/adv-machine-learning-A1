from ML2A1_2_helper import *
from ML2A1_train import *
from ML2A1_test import *

import torch
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# train and test script are gonna b separate i think
# called in pipe like as: testscript with specs | using model trained on these train specs?

#TBD sys args
src_dir0 = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'
src_dir = '../ThaiOCR/ThaiOCR-TrainigSet/'

train_specs = {'lgs': ['English'], 'dpis': ['200'], 'fonts': ['normal']}
test_specs = {'lgs': ['English'], 'dpis': ['200'], 'fonts': ['normal']}

#est_specs=False
# ? call with optional separate test set specs?, like in func call keyword

### LOAD DATA ###
# Get relevant filenames according to specs
data = DataLoader(src_dir, train_specs, limit=5000)

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
load=False
save=False
# rewrite to check for default model name file with isfile to prio loading over new training
if load:
    # load model
    m = CNN(data.n_classes, data.avg_size, data.idx_to_char)
    m.load_state_dict(torch.load('modelsep23', weights_only=True))
    m.eval()
else:
    m = train(data, 2, device, save=save)

## testing ##
test_data = DataLoader(src_dir, test_specs, m.img_dims)
test(data, m, verbose=True)


## use cpu ##
# import torch
# device = torch.device('cuda:1')
# train_X_tensor = torch.Tensor(train_X).to(device)
# train_y_tensor = toch.Tensor(train_y).to(device)
# repeat for dev/test

# send entire train_X matrix to cuda
 

