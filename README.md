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

### Dataloader

Contains DataLoader and OCRData classes, used to filter relevant files from source directory and transform to required format for both training and testing. File has no main function.

___

## Experiments

1) Thai normal text, 200dpi -> Thai normal text, 200dpi

> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -v  
--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.9

Overview of measures across classes:

      Precision  Recall  F1-score  
mean       0.91    0.90      0.90  
std        0.12    0.13      0.10  
min        0.54    0.47      0.62  
25%        0.86    0.88      0.85  
50%        0.94    0.95      0.94  
75%        1.00    1.00      0.97  
max        1.00    1.00      1.00  

Overview of 5 worst performing classes per measure:

Precision  
ซ       0.54  
อ์ใ(ติด)  0.55  
ๅ       0.56  
อึ้(ติด)   0.58  
อ้       0.67  

Recall  
อ์ไ(ติด)  0.47  
อ้ไ(ติด)  0.50  
ฃ       0.52  
ช       0.57  
ศื(ติด)   0.60  


F1-score  
ๅ        0.62  
อ์ไ(ติด)   0.64  
อ้ไ(ติด)   0.65  
ฃ        0.65  
อึ้(ติด)    0.67  

___

2) Thai normal 400 -> Thai normal 200

> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.86

Overview of measures across classes:

      Precision  Recall  F1-score  
mean       0.86    0.84      0.84  
std        0.17    0.18      0.15  
min        0.00    0.00      0.00  
25%        0.77    0.79      0.74  
50%        0.92    0.90      0.89  
75%        1.00    0.96      0.94  
max        1.00    1.00      1.00  

Overview of 5 worst performing classes per measure:

Precision  
ศื(ติด)    0.00  
อ่         0.46  
ด          0.48  
สี(ติด)    0.50  
ต          0.50  

Recall  
ศื(ติด)     0.00  
อี้(ติด)    0.33  
ศี(ติด)     0.38  
สื(ติด)     0.43  
า           0.46  

F1-score  
ศื(ติด)     0.00  
อี้(ติด)    0.47  
ช           0.54  
ศี(ติด)     0.55  
สื(ติด)     0.55  

___

3) Thai normal 400 - Thai bold 400

> $ python3 test_ML2A1.py -lg Thai -dpi 400 -ft bold -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.84

Overview of measures across classes:

      Precision  Recall  F1-score  
mean       0.86    0.83      0.82  
std        0.18    0.18      0.17  
min        0.18    0.22      0.29  
25%        0.79    0.75      0.73  
50%        0.93    0.89      0.88  
75%        1.00    0.95      0.95  
max        1.00    1.00      1.00  

Overview of 5 worst performing classes per measure:

Precision  
ส้(ติด)     0.18  
ป้(ติด)     0.29  
อึ้(ติด)    0.34  
ด           0.37  
ต           0.42  

Recall  
สี(ติด)     0.22  
ศื(ติด)     0.22  
อี้(ติด)    0.33  
ศั(ติด)     0.33  
ช           0.44  

F1-score  
ส้(ติด)     0.29  
สี(ติด)     0.36  
ศื(ติด)     0.36  
ป้(ติด)     0.40  
อึ้(ติด)    0.47  

___

4) Thai bold -> Thai normal

> $ python3 test_ML2A1.py -lg Thai -dpi 200 300 400 -ft normal -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.85

Overview of measures across classes:

      Precision  Recall  F1-score  
mean       0.86    0.84      0.84  
std        0.12    0.15      0.11  
min        0.57    0.12      0.22  
25%        0.79    0.77      0.78  
50%        0.89    0.89      0.86  
75%        0.95    0.96      0.91  
max        1.00    1.00      1.00  

Overview of 5 worst performing classes per measure:

Precision  
า           0.57  
ฟั(ติด)     0.58  
อี้(ติด)    0.60  
อึ้(ติด)    0.60  
ฃ           0.63  

Recall  
ส้(ติด)     0.12  
ป้(ติด)     0.46  
อี          0.55  
ๅ           0.56  
อึ่(ติด)    0.56  

F1-score  
ส้(ติด)     0.22  
ป้(ติด)     0.61  
อึ้(ติด)    0.64  
ฃ           0.65  
อี          0.66  

___

5) All Thai -> All Thai

> $ python3 test_ML2A1.py -lg Thai -dpi 200 300 400 -ft normal bold italic italic_bold -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.95

Overview of measures across classes:

      Precision  Recall  F1-score  
mean       0.95    0.95      0.95  
std        0.06    0.05      0.04  
min        0.72    0.71      0.78  
25%        0.94    0.93      0.93  
50%        0.98    0.97      0.97  
75%        0.99    0.99      0.98  
max        1.00    1.00      1.00  

Overview of 5 worst performing classes per measure:

Precision  
า           0.72  
อ่          0.75  
อ้ใ(ติด)    0.82  
ฏ           0.82  
อุ          0.83  

Recall  
ๅ     0.71  
ฎ     0.78  
อึ    0.83  
ด     0.83  
อื    0.84  

F1-score  
ๅ           0.78  
า           0.80  
อ่          0.80  
อึ้(ติด)    0.87  
อี          0.87  

___

6) Thai & en normal -> Thai & en normal

> $ python3 test_ML2A1.py -lg English Thai -dpi 200 300 400 -ft normal -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.94

Overview of measures across classes:

      Precision  Recall  F1-score  
mean       0.95    0.94      0.94  
std        0.10    0.11      0.10  
min        0.29    0.31      0.42  
25%        0.94    0.94      0.94  
50%        0.98    0.98      0.97  
75%        1.00    1.00      0.99  
max        1.00    1.00      1.00  

Overview of 5 worst performing classes per measure:

Precision  
อ่    0.29  
l     0.53  
.     0.59  
o     0.60  
I     0.67  

Recall  
i    0.31  
I    0.35  
l    0.49  
V    0.57  
O    0.58  

F1-score  
อ่    0.42  
i     0.46  
I     0.46  
l     0.51  
.     0.64  

___


7) All styles -> All styles

> $ python3 test_ML2A1.py -lg English Thai -dpi 200 300 400 -ft normal italic bold italic_bold -v

--------------------------------------------------------------------------------
Evaluation

Overall accuracy: 0.91

Overview of measures across classes:

      Precision  Recall  F1-score  
mean       0.92    0.91      0.91  
std        0.11    0.14      0.12  
min        0.37    0.21      0.34  
25%        0.92    0.92      0.90  
50%        0.97    0.96      0.96  
75%        0.99    0.99      0.98  
max        1.00    1.00      1.00  

Overview of 5 worst performing classes per measure:

Precision  
อ่    0.37  
i     0.43  
C     0.55  
l     0.57  
o     0.59  

Recall  
c    0.21  
l    0.35  
i    0.45  
v    0.46  
O    0.46  

F1-score  
c     0.34  
l     0.43  
i     0.44  
อ่    0.50  
v     0.56  

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

? Isolating characters, guessing whitespaces

* model is surprisingly sensitive to even one row or column of empty pixel in the margins

i.e.  Output for dummy image with '2.5 Drawing Editor'. 
- No empty lines: 2.s DraWing EdItor  
- single row above/below: 2*s Dอ์aW1ng Edi*๐r  
- 1+ empty rows above/below: ?อื่(ติด)5 pfฐฟs6ฏ rdฝั(ติด)xอิE - -
- 1+ empty columns left/right: 2.s DraWIng Ed|*or - -
- single empty column left/right: 2อึ่(ติด)5 p๔อีฟs6ฏ rdๆfbE - -




* data being equally represented might make classification harder: most test/'natural' has lower amount of special and/or numeric characters


