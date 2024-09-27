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

To test a model, run the test script and specify what data to test on. The script can either test a pre-trained model (passed with --loadfile), or will otherwise interactively ask for information to train a new model on execution.

The following arguments are required and determine which files will be extracted from the source directory to use during testing. They can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...).
* --languages (-lg):	Thai | English | Thai English
* --dpis (-dpi): 	200 | 300 | 400 | ...
* --fonts (-ft):	normal | bold | italic | italic_bold | ...

The following arguments are optional. --loadfile specifies where to find the pre-trained model, if invalid/left unspecified, the test script allows the user to specify information to train a new model using the train script. --verbose increases the amount of detail printed during model evaluation. The final argument may be used to specify a different directory to read the source data from.
* --loadfile (-ld):		any filename/path, defaults to None
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

Overall accuracy: 0.88

Overview of per-class measures

       Precision    Recall  F1-score
mean   0.889251  0.879119  0.868755
std    0.143038  0.155446  0.128142
min    0.285714  0.357143  0.444444
25%    0.833333  0.850000  0.800000
50%    0.941176  0.928571  0.914286
75%    1.000000  1.000000  0.965517
max    1.000000  1.000000  1.000000

Overview of 5 best/worst performing classes per measure.

Top Precision
ศื(ติด)    1.0
ป          1.0
ษ          1.0
ย          1.0
ก          1.0
Name: Precision, dtype: float64

Bottom Precision
สี(ติด)     0.29
ซ           0.41
อ์โ(ติด)    0.48
อ็ไ(ติด)    0.57
ส           0.58
Name: Precision, dtype: float64

Top Recall
ฝั(ติด)    1.0
ฟั(ติด)    1.0
ษ          1.0
ศื(ติด)    1.0
ม          1.0
Name: Recall, dtype: float64

Bottom Recall
อึ้(ติด)    0.36
ช           0.43
ฃ           0.43
ส้(ติด)     0.50
ฏ           0.50
Name: Recall, dtype: float64

Top F1-score
ศื(ติด)    1.0
ฑ          1.0
ษ          1.0
ย          1.0
ศั(ติด)    1.0
Name: F1-score, dtype: float64

Bottom F1-score
สี(ติด)     0.44
อึ้(ติด)    0.53
ฃ           0.58
ฏ           0.58
ซ           0.59
Name: F1-score, dtype: float64
___

2) Thai normal 400 -> Thai normal 200

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.89

Overview of per-class measures

       Precision    Recall  F1-score
mean   0.891072  0.882895  0.875839
std    0.120250  0.138983  0.112729
min    0.400000  0.200000  0.333333
25%    0.833333  0.818182  0.833333
50%    0.928571  0.928571  0.900000
75%    1.000000  1.000000  0.958333
max    1.000000  1.000000  1.000000

Overview of 5 best/worst performing classes per measure.

Top Precision
ศั(ติด)     1.0
ผ           1.0
สิ(ติด)     1.0
ห           1.0
อ์ไ(ติด)    1.0
Name: Precision, dtype: float64

Bottom Precision
สี(ติด)    0.40
อู         0.57
ฏ          0.59
อี         0.62
ป้(ติด)    0.64
Name: Precision, dtype: float64

Top Recall
ศั(ติด)     1.0
อ์          1.0
สิ(ติด)     1.0
อ็ไ(ติด)    1.0
ฤ           1.0
Name: Recall, dtype: float64

Bottom Recall
ศื(ติด)    0.20
ๅ          0.40
อื         0.54
ฎ          0.60
ฃ          0.62
Name: Recall, dtype: float64

Top F1-score
ศั(ติด)    1.0
สิ(ติด)    1.0
ป็(ติด)    1.0
ฉ          1.0
ถ          1.0
Name: F1-score, dtype: float64

Bottom F1-score
ศื(ติด)    0.33
ๅ          0.53
สี(ติด)    0.57
ฏ          0.65
อู         0.69
Name: F1-score, dtype: float64

___


3) Thai normal 400 - Thai bold 400

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.89

Overview of per-class measures

       Precision    Recall  F1-score
mean   0.889282  0.869996  0.865172
std    0.153575  0.184635  0.150033
min    0.000000  0.000000  0.000000
25%    0.814815  0.833333  0.814815
50%    0.941176  0.941176  0.888889
75%    1.000000  1.000000  0.967742
max    1.000000  1.000000  1.000000

Overview of 5 best/worst performing classes per measure.

Top Precision
ฅ          1.0
ป          1.0
อ๋         1.0
ช          1.0
ป็(ติด)    1.0
Name: Precision, dtype: float64

Bottom Precision
ศี(ติด)     0.00
ต           0.55
ศื(ติด)     0.57
อึ้(ติด)    0.57
สื(ติด)     0.60
Name: Precision, dtype: float64

Top Recall
ก           1.0
อ็          1.0
ป           1.0
อ๋          1.0
อ์ไ(ติด)    1.0
Name: Recall, dtype: float64

Bottom Recall
ศี(ติด)    0.00
ป้(ติด)    0.33
ศั(ติด)    0.33
สี(ติด)    0.33
ฝ้(ติด)    0.48
Name: Recall, dtype: float64

Top F1-score
ป         1.0
ฑ         1.0
อ๋        1.0
ฐ(ติด)    1.0
น         1.0
Name: F1-score, dtype: float64

Bottom F1-score
ศี(ติด)     0.00
ป้(ติด)     0.50
ศั(ติด)     0.50
สี(ติด)     0.50
อึ้(ติด)    0.57
Name: F1-score, dtype: float64


___

4) Thai bold -> Thai normal

Evaluation

Overall accuracy: 0.83

Overview of per-class measures

       Precision    Recall  F1-score
mean   0.841818  0.819553  0.821671
std    0.125316  0.137928  0.111830
min    0.531250  0.337838  0.433121
25%    0.741573  0.764706  0.748387
50%    0.881188  0.867647  0.857143
75%    0.937500  0.920000  0.910053
max    1.000000  0.988889  0.974684

Overview of 5 best/worst performing classes per measure.

Top Precision
ฉ          1.0
ฒ          1.0
อ๊         1.0
อิ         1.0
สื(ติด)    1.0
Name: Precision, dtype: float64

Bottom Precision
ฏ    0.53
ฃ    0.59
า    0.59
ข    0.60
ฎ    0.61
Name: Precision, dtype: float64

Top Recall
ง          0.99
ฟ          0.99
สั(ติด)    0.99
ม          0.96
ศ          0.95
Name: Recall, dtype: float64

Bottom Recall
อี่(ติด)    0.34
ฏ           0.37
ส้(ติด)     0.47
ซ           0.49
สื(ติด)     0.51
Name: Recall, dtype: float64

Top F1-score
เ     0.97
ฒ     0.97
อ๋    0.96
ผ     0.95
พ     0.95
Name: F1-score, dtype: float64

Bottom F1-score
ฏ           0.43
อี่(ติด)    0.50
ข           0.58
า           0.59
ศื(ติด)     0.61
Name: F1-score, dtype: float64


5) All Thai -> All Thai

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.96

Overview of per-class measures

       Precision    Recall  F1-score
mean   0.961004  0.959113  0.958155
std    0.054127  0.063133  0.050617
min    0.667742  0.491304  0.653179
25%    0.956000  0.952381  0.951456
50%    0.976834  0.975309  0.971209
75%    0.991736  0.990119  0.986900
max    1.000000  1.000000  0.998088

Overview of 5 best/worst performing classes per measure.

Top Precision
ณ          1.0
ปั(ติด)    1.0
ป็(ติด)    1.0
ฬ          1.0
ส          1.0
Name: Precision, dtype: float64

Bottom Precision
อึ้(ติด)    0.67
ๅ           0.69
อื          0.82
อี่(ติด)    0.83
ฟ้(ติด)     0.85
Name: Precision, dtype: float64

Top Recall
ผ          1.0
ฝ้(ติด)    1.0
พ          1.0
สิ(ติด)    1.0
ศั(ติด)    1.0
Name: Recall, dtype: float64

Bottom Recall
า           0.49
อึ่(ติด)    0.75
อี้(ติด)    0.79
ฟั(ติด)     0.86
อึ          0.86
Name: Recall, dtype: float64

Top F1-score
ส         1.0
ง         1.0
ผ         1.0
ม         1.0
ญ(ติด)    1.0
Name: F1-score, dtype: float64

Bottom F1-score
า           0.65
อึ้(ติด)    0.79
ๅ           0.81
อี้(ติด)    0.83
อึ่(ติด)    0.84
Name: F1-score, dtype: float64




6) Thai & en normal -> Thai & en normal

Evaluation

Overall accuracy: 0.95

Overview of per-class measures

       Precision    Recall  F1-score
mean   0.954209  0.950250  0.949650
std    0.081460  0.092199  0.084805
min    0.384615  0.311688  0.457143
25%    0.950000  0.949367  0.935673
50%    0.985915  0.985714  0.982036
75%    1.000000  1.000000  0.993377
max    1.000000  1.000000  1.000000

Overview of 5 best/worst performing classes per measure.

Top Precision
Y          1.0
สิ(ติด)    1.0
g          1.0
ส          1.0
ซ          1.0
Name: Precision, dtype: float64

Bottom Precision
อ่    0.38
I     0.57
i     0.71
า     0.73
V     0.74
Name: Precision, dtype: float64

Top Recall
Y          1.0
R          1.0
ฟั(ติด)    1.0
สิ(ติด)    1.0
ญ(ติด)     1.0
Name: Recall, dtype: float64

Bottom Recall
l    0.31
i    0.48
I    0.58
.    0.66
v    0.74
Name: Recall, dtype: float64

Top F1-score
Y          1.0
สิ(ติด)    1.0
Z          1.0
r          1.0
ศั(ติด)    1.0
Name: F1-score, dtype: float64

Bottom F1-score
l     0.46
อ่    0.53
I     0.58
i     0.58
.     0.71
Name: F1-score, dtype: float64



7) All styles -> All styles
--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.94

Overview of per-class measures

       Precision    Recall  F1-score
mean   0.942098  0.938176  0.937964
std    0.096064  0.101215  0.093534
min    0.434524  0.507692  0.484245
25%    0.938503  0.935361  0.934156
50%    0.980237  0.980952  0.977131
75%    0.991770  0.991701  0.989648
max    1.000000  1.000000  1.000000

Overview of 5 best/worst performing classes per measure.

Top Precision
L           1.0
k           1.0
ฝ           1.0
ฑ           1.0
อ้โ(ติด)    1.0
Name: Precision, dtype: float64

Bottom Precision
i     0.43
l     0.46
อ่    0.52
o     0.68
า     0.71
Name: Precision, dtype: float64

Top Recall
k          1.0
h          1.0
m          1.0
ศั(ติด)    1.0
ผ          1.0
Name: Recall, dtype: float64

Bottom Recall
I     0.51
i     0.55
อ่    0.55
w     0.60
c     0.65
Name: Recall, dtype: float64

Top F1-score
k    1.0
A    1.0
Z    1.0
ฝ    1.0
N    1.0
Name: F1-score, dtype: float64

Bottom F1-score
i     0.48
อ่    0.53
l     0.56
I     0.65
.     0.70
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
