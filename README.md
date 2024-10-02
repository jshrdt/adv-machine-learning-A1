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
> (y/n) >> y  

> Train new model on same specifications as test data?  
> {'languages': ['Thai'], 'dpis': ['200'], 'fonts': ['normal']}  
> (y/n) >> y  

> Keep default params for epochs(5)/batch_size(32)/savefile(None)?  
> (y/n) >> y  

___

## Detailed overview

### Training a model with train_ML2A1.py

To train a new model, run the train script and specify what data to use from the training repository. This file also contains the NN model architecture.

The following arguments are required and determine which files will be extracted from the source directory to use during training. They can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...).
* --languages (-lg):	Thai | English | Thai English
* --dpis (-dpi): 	200 | 300 | 400 | ...
* --fonts (-ft):	normal | bold | italic | bold_italic | ...

The following arguments are optional. The first two alter behaviour during the training loop. --savefile allows the trained model to be saved under the passed filename/path. The final argument may be used to specify a different directory to read the source data from.
* --epochs (-ep):	any integer, defaults to 5
* --batch_size (-bs):	any integer, defaults to 32
* --savefile (-s):	any filename/path, defaults to None
* --source_dir (-srcd):	pathname to directory for OCR data, defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/' on gpu, '../ThaiOCR/ThaiOCR-TrainigSet/' on cpu

> $ python3 train_ML2A1.py -lg Thai English -dpi 200 300 -ft italic -ep 8 -bs 64 -s ThaiEn_200300_ita_custom -srcd [custom/path/to/OCR/data]


### Testing a model with test_ML2A1.py

To test a model, run the test script and specify what data to test on. The script can either test a pre-trained model (passed with --loadfile), or will otherwise interactively ask for information to train a new model on execution.

The following arguments are required and determine which files will be extracted from the source directory to use during testing. They can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...).
* --languages (-lg):	Thai | English | Thai English
* --dpis (-dpi): 	200 | 300 | 400 | ...
* --fonts (-ft):	normal | bold | italic | bold_italic | ...

The following arguments are optional. --loadfile specifies where to find the pre-trained model, if invalid/left unspecified, the test script allows the user to specify information to train a new model using the train script. --verbose increases the amount of detail printed during model evaluation. The final argument may be used to specify a different directory to read the source data from.
* --loadfile (-ld):		any filename/path, defaults to None
* --verbose (-v):	on/off flag
* --source_dir (-srcd):	pathname to directory for OCR data, defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/' on gpu, '../ThaiOCR/ThaiOCR-TrainigSet/' on cpu

> $ python3-s test_ML2A1.py -lg Thai -dpi 400 -ft bold -ld ThaiEn_200300_ita_custom -v -srcd [custom/path/to/OCR/data]

### Dataloader

Contains DataLoader and OCRData classes, used to filter relevant files from source directory and transform to required format for both training and testing. File has no main function.

___

## Experiments

1) Thai normal text, 200dpi -> Thai normal text, 200dpi

> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -v  
Evaluation

Overall accuracy: 0.92

Overview of measures across classes:

      Precision  Recall  F1-score
mean       0.92    0.92      0.92
std        0.10    0.11      0.09
min        0.62    0.47      0.58
25%        0.88    0.88      0.86
50%        0.94    0.95      0.95
75%        1.00    1.00      0.98
max        1.00    1.00      1.00

Overview of 5 worst performing classes per measure:

Precision
า           0.62
ซ           0.64
อี่(ติด)    0.66
สี(ติด)     0.67
ช           0.70
Name: Precision, dtype: float64

Recall
อ์ไ(ติด)    0.47
ซ           0.53
ๅ           0.55
อื          0.67
อึ่(ติด)    0.71
Name: Recall, dtype: float64

F1-score
ซ           0.58
อ์ไ(ติด)    0.59
ๅ           0.63
า           0.70
อื          0.78
Name: F1-score, dtype: float64

___

2) Thai normal 400 -> Thai normal 200

> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.93

Overview of measures across classes:

      Precision  Recall  F1-score
mean       0.94    0.93      0.93
std        0.09    0.08      0.07
min        0.62    0.64      0.73
25%        0.90    0.90      0.90
50%        1.00    0.95      0.96
75%        1.00    1.00      0.98
max        1.00    1.00      1.00

Overview of 5 worst performing classes per measure:

Precision
อ่         0.62
ๅ          0.64
สี(ติด)    0.67
ต          0.67
ซ          0.72
Name: Precision, dtype: float64

Recall
อึ    0.64
า     0.65
ด     0.69
ฏ     0.71
ฃ     0.71
Name: Recall, dtype: float64

F1-score
อ่    0.73
า     0.74
ซ     0.74
ๅ     0.75
ฃ     0.77
Name: F1-score, dtype: float64

___

3) Thai normal 400 - Thai bold 400

> $ python3 test_ML2A1.py -lg Thai -dpi 400 -ft bold -v -ld Thai400normal

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.93

Overview of measures across classes:

      Precision  Recall  F1-score
mean       0.93    0.92      0.92
std        0.11    0.13      0.11
min        0.50    0.33      0.50
25%        0.90    0.89      0.89
50%        1.00    0.96      0.96
75%        1.00    1.00      1.00
max        1.00    1.00      1.00

Overview of 5 worst performing classes per measure:

Precision
อ์ใ(ติด)    0.50
ต           0.50
อึ้(ติด)    0.58
อ้ไ(ติด)    0.65
ท           0.65
Name: Precision, dtype: float64

Recall
ศั(ติด)     0.33
อ้ใ(ติด)    0.39
อี้(ติด)    0.61
า           0.65
สี(ติด)     0.67
Name: Recall, dtype: float64

F1-score
ศั(ติด)     0.50
อ้ใ(ติด)    0.56
อ์ใ(ติด)    0.64
ต           0.67
อึ้(ติด)    0.67
Name: F1-score, dtype: float64

___

4) Thai bold -> Thai normal

> $ python3 test_ML2A1.py -lg Thai -dpi 200 300 400 -ft normal -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.92

Overview of measures across classes:

      Precision  Recall  F1-score
mean       0.92    0.91      0.92
std        0.08    0.09      0.07
min        0.61    0.63      0.73
25%        0.88    0.87      0.88
50%        0.94    0.94      0.93
75%        0.99    0.97      0.96
max        1.00    1.00      1.00

Overview of 5 worst performing classes per measure:

Precision
ข           0.61
อ์ไ(ติด)    0.71
ๅ           0.72
อี้(ติด)    0.73
อ่          0.77
Name: Precision, dtype: float64

Recall
อ์ใ(ติด)    0.63
ฃ           0.65
ซ           0.65
ส้(ติด)     0.66
า           0.66
Name: Recall, dtype: float64

F1-score
ฃ           0.73
ข           0.73
า           0.73
อ์ใ(ติด)    0.75
ซ           0.76
Name: F1-score, dtype: float64

___

5) All Thai -> All Thai

> $ python3 test_ML2A1.py -lg Thai -dpi 200 300 400 -ft normal bold italic bold_italic -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.99

Overview of measures across classes:

      Precision  Recall  F1-score
mean       0.99    0.99      0.99
std        0.02    0.02      0.01
min        0.88    0.92      0.92
25%        0.98    0.98      0.99
50%        0.99    0.99      0.99
75%        1.00    1.00      1.00
max        1.00    1.00      1.00

Overview of 5 worst performing classes per measure:

Precision
อ่          0.88
อำ          0.94
า           0.94
อื          0.95
อึ่(ติด)    0.95
Name: Precision, dtype: float64

Recall
อุ          0.92
อี่(ติด)    0.94
อำ          0.95
ๅ           0.95
อี          0.95
Name: Recall, dtype: float64

F1-score
อ่    0.92
อำ    0.94
า     0.95
อุ    0.96
ๅ     0.96
Name: F1-score, dtype: float64

___

6) Thai & en normal -> Thai & en normal

> $ python3 test_ML2A1.py -lg English Thai -dpi 200 300 400 -ft normal -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.97

Overview of measures across classes:

      Precision  Recall  F1-score
mean       0.97    0.97      0.97
std        0.07    0.07      0.07
min        0.37    0.51      0.47
25%        0.97    0.98      0.97
50%        0.99    0.99      0.99
75%        1.00    1.00      1.00
max        1.00    1.00      1.00

Overview of 5 worst performing classes per measure:

Precision
l     0.37
I     0.68
i     0.71
อ่    0.83
V     0.83
Name: Precision, dtype: float64

Recall
i     0.51
.     0.64
l     0.65
อ่    0.66
I     0.74
Name: Recall, dtype: float64

F1-score
l     0.47
i     0.59
I     0.71
อ่    0.74
.     0.74
Name: F1-score, dtype: float64

___


7) All styles -> All styles

> $ python3 test_ML2A1.py -lg English Thai -dpi 200 300 400 -ft normal italic bold bold_italic -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.97

Overview of measures across classes:

      Precision  Recall  F1-score
mean       0.97    0.97      0.97
std        0.07    0.06      0.06
min        0.54    0.59      0.56
25%        0.97    0.97      0.97
50%        0.99    0.99      0.99
75%        1.00    1.00      1.00
max        1.00    1.00      1.00

Overview of 5 worst performing classes per measure:

Precision
l     0.54
i     0.58
อ่    0.73
I     0.75
.     0.76
Name: Precision, dtype: float64

Recall
l     0.59
i     0.65
I     0.72
อ่    0.74
.     0.75
Name: Recall, dtype: float64

F1-score
l     0.56
i     0.61
อ่    0.74
I     0.74
.     0.76
Name: F1-score, dtype: float64
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

___ 

## Bonus part

$ python3 test_ML2A1.py -lg Thai Numeric Special English -dpi 200 300 400 -ft normal bold bold_italic italic -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.96

Overview of measures across classes:

      Precision  Recall  F1-score
mean       0.96    0.96      0.96
std        0.08    0.08      0.08
min        0.52    0.46      0.52
25%        0.96    0.97      0.97
50%        0.99    0.99      0.99
75%        1.00    1.00      1.00
max        1.00    1.00      1.00

Overview of 5 worst performing classes per measure:

Precision
|     0.52
-     0.58
l     0.60
อ่    0.64
I     0.67
Name: Precision, dtype: float64

Recall
i     0.46
|     0.51
l     0.54
I     0.65
อ่    0.66
Name: Recall, dtype: float64

F1-score
|     0.52
i     0.55
l     0.57
อ่    0.65
I     0.66
Name: F1-score, dtype: float64


? Isolating characters, guessing whitespaces

* model is surprisingly sensitive to even one row or column of empty pixel in the margins

i.e.  Output for dummy image with '2.5 Drawing Editor'. 
- No empty lines: 2.s DraWing EdItor  
- single row above/below: 2*s Dอ์aW1ng Edi*๐r  
- 1+ empty rows above/below: ?อื่(ติด)5 pfฐฟs6ฏ rdฝั(ติด)xอิE - -
- single empty column left/right: 2อึ่(ติด)5 p๔อีฟs6ฏ rdๆfbE - -




* data being equally represented might make classification harder: most test/'natural' has lower amount of special and/or numeric characters


