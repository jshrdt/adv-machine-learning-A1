# adv-machine-learning-A1

#### Required modules

* argparse, numpy, os, pandas, PIL, random, sklearn.metrics, sklearn.preprocessing, torch, tqdm

___

## Quickstart

### Running train_ML2A1.py:

(1) Train a new model, optional: -s [file/pathname], save to file.  
> $ python3 train_ML2A1.py -lg Thai -dpi 200 -ft normal -s Thai200normal_model

### Running test_ML2A1.py:

(1) Test a pre-trained model loaded from file.
> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -ld Thai200normal_model

(2) Train & test a new model, optional: save to file.
> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal  

> No pre-trained model found, train new model?  
> (y/n) 
>> y  

> Train new model on same specifications as test data?  
> {'languages': ['Thai'], 'dpis': ['200'], 'fonts': ['normal']}  
> (y/n) 
>> y  

> Keep default params for epochs(5)/batch_size(32)/savefile(None)?  
> (y/n)
>> y  

___

## Detailed overview

### Training a model with train_ML2A1.py

To train a new model, run the train script and specify what data to use from the training repository. This file also contains the NN model architecture.

The following arguments are required and determine which files will be extracted from the source directory to use during training. They can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...).
* --languages (-lg):	Thai | English | Thai English
* --dpis (-dpi): 	200 | 300 | 400 | ...
* --fonts (-ft):	normal | bold | italic | italic_bold | ...

The following arguments are optional. The first two alter behaviour during the training loop. --savefile allows the trained model to be saved under the passed filename/path. The final argument may be used to specify a different directory to read the source data from.
* --epochs (-ep):	any integer, defaults to 5
* --batch_size (-bs):	any integer, defaults to 32
* --savefile (-s):	any filename/path, defaults to None
* --source_dir (-srcd):	pathname to directory for OCR data, defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/' on gpu, '../ThaiOCR/ThaiOCR-TrainigSet/' on cpu

> $ python3 train_ML2A1.py -lg Thai English -dpi 200 300 -ft italic -ep 8 -bs 64 -s ThaiEn_200300_ita_custom -srcd [custom/path/to/OCR/data]



### Testing a model with test_ML2A1.py

To test a model, run the test script and specify what data to test on. The script can either test a pre-trained model (passed with --load), or will otherwise interactively ask for information to train a new model on execution.

The following arguments are required and determine which files will be extracted from the source directory to use during testing. They can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...).
* --languages (-lg):	Thai | English | Thai English
* --dpis (-dpi): 	200 | 300 | 400 | ...
* --fonts (-ft):	normal | bold | italic | italic_bold | ...

The following arguments are optional. --load specifies where to find the pre-trained model, if invalid/left unspecified, the test script allows the user to specify information to train a new model using the train script. --verbose increases the amount of detail printed during model evaluation. The final argument may be used to specify a different directory to read the source data from.
* --load (-ld):		any filename/path, defaults to None
* --verbose (-v):	on/off flag
* --source_dir (-srcd):	pathname to directory for OCR data, defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/' on gpu, '../ThaiOCR/ThaiOCR-TrainigSet/' on cpu

> $ python3-s test_ML2A1.py -lg Thai -dpi 400 -ft bold -ld ThaiEn_200300_ita_custom -v -srcd [custom/path/to/OCR/data]

## Dataloader

Contains DataLoader class for OCR data, used to load image from source directory data for both training and testing, as well as the MyBatcher class for batching of data during training. File has no main function.

___

# TODO

## Experiments

1) Thai normal text, 200dpi - Thai normal text, 200dpi

> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -v  
Evaluation

Overall accuracy: 0.8

Per-class measures
          Precision  Recall  F1-score
อ็ไ(ติด)       0.63    1.00      0.77
อ๋             0.89    1.00      0.94
ด              0.93    0.82      0.88
ไ              0.93    0.70      0.80
ล              0.94    1.00      0.97
...             ...     ...       ...
ศื(ติด)        0.55    1.00      0.71
ป้(ติด)        0.81    0.85      0.83
ซ              0.63    0.75      0.69
า              0.60    0.88      0.71
MACROS         0.81    0.79      0.78

Precision performance below 0.5:
อ้ใ(ติด)    0.000000
สี(ติด)     0.454545
อ์โ(ติด)    0.485714
อึ่(ติด)    0.443182
อ์ใ(ติด)    0.289474
Name: Precision, dtype: float64  
Recall performance below 0.5:
ศี(ติด)     0.285714
อี่(ติด)    0.142857
อ้ใ(ติด)    0.000000
อึ้(ติด)    0.043478
อื          0.444444
อื่(ติด)    0.277778
อ้ไ(ติด)    0.040000
อ์ไ(ติด)    0.133333
Name: Recall, dtype: float64  
F1-score performance below 0.5:
ศี(ติด)     0.363636
อี่(ติด)    0.240000
อ้ใ(ติด)    0.000000
อึ้(ติด)    0.083333
อื่(ติด)    0.357143
อ์ใ(ติด)    0.431373
อ้ไ(ติด)    0.076923
อ์ไ(ติด)    0.210526
Name: F1-score, dtype: float64


2) Thai normal 400 -> Thai normal 200

Overall accuracy: 0.76

Performance across all classes  
Precision    0.7674
Recall       0.7572
F1-score     0.7390


3) Thai normal 400 - Thai bold 400

Overall accuracy: 0.77

Performance across all classes  
Precision    0.7666
Recall       0.7537
F1-score     0.7412


4) Thai bold - Thai normal
Evaluation

Overall accuracy: 0.75

Performance across all classes  
Precision    0.7695
Recall       0.7475
F1-score     0.7360
___

### Other

? For training on en, 200, normal to 300 -> performance on w/W increased dramatically
Only underperforming classes are I, I, and l (to be expected)
Overall accuracy: 0.87

Precision performance below 0.5:
i    0.250000
I    0.466667
l    0.278689
Name: Precision, dtype: float64

Recall performance below 0.5:
i    0.390244
I    0.264151
l    0.386364
Name: Recall, dtype: float64

F1-score performance below 0.5:
i    0.304762
I    0.337349
l    0.323810

___

## Challenges:

? size_to, resizing images to work for any dpi combination; make accessible in testing, without looking at training data again

? Some struggles with resizing tensors to allow for both batched input, as well as single images during testing

? Fixing the txt file from source directory, for correct label extraction/encoding/decoding

? Too many open files problem (open img -> get sizes -> resize ; vs open img+get sizes -> open images+resize) as PIL only closes images when img data is used

? Turns out sets don't work on tensors,,,for a while predicted over 13k classes instead of ~150

? On testing thai200normal, a new model would usually hover around 0.8 overall accuracy but sometimes?? Drop down to 0.01???

? Inconsistent torch behaviour: https://github.com/pytorch/pytorch/issues/51112 solved : .reset_index(drop=True)

? Restructuring: open files twice, but transform only those when needed
