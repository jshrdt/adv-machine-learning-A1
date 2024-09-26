# adv-machine-learning-A1

Quickstart:

Train: 
(1) Train new model, [optional: save to file].
* $ python3 train_ML2A1.py -lg Thai -dpi 200 -ft normal [-s Thai200normal_model]

Test: 
(1) Test pre-trained model loaded from file.
* $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -ld Thai200normal_model

(2) Tain & test new model, optional: save to file.
* $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal
* > No pre-trained model found, train new model?
* > (y/n) y
* > Train new model on same specifications as test data?
* > {'languages': ['Thai'], 'dpis': ['200'], 'fonts': ['normal']}
* > (y/n) y
* > Keep default params for epochs(5)/batch_size(32)/savefile(None)?
* > (y/n) y


## Dataloader

TODO

_____________________________________________

Detailed overview | full functionality

## Train
To train a new model, call the train script and specify what data to use from training repository. 
The following required arguments can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...)
* --languages (-lg):	Thai | English | Thai English
* --dpis (-dpi): 		200 | 300 | 400 | ...
* --fonts (-ft):		normal | bold | italic | italic_bold | ...

The following arguments are optional. The first two alter behaviour during the training loop. --Savefile allows the trained model to be saved under the passed filename/path. The final argument may be used to specify a different directory to read the source data from.
	--epochs (-ep):		any integer, defaults to 5
	--batch_size (-bs):	any integer, defaults to 32
	--savefile (-s)		any filename/path, defaults to None
	--source_dir (-srcd):	pathname to directory for OCR data, defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'

File also holds the NN definition.


## Test
To test a model, call the test script and specify what data to test it on. The script can either test a pretrained model (passed with --load), or will otherwise interactively ask for information to train a new model on execution.

The following required arguments can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...)
	--languages (-lg):	Thai | English | Thai English
	--dpis (-dpi): 		200 | 300 | 400 | ...
	--fonts (-ft):		normal | bold | italic | italic_bold | ...

The following arguments are optional. --load specifies where to find the pretrained model, if left unspecified/invalid, a new model may be trained with the train script. --verbose determines the amount of detail printed during model evaluation. The final argument may be used to specify a different directory to read the source data from.
	--load (-ld):		any filename/path, defaults to None
	--verbose (-v):		on/off flag
	--source_dir (-srcd):	pathname to directory for OCR data, defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'

## Dataloader

TODO

## experiments
1)
python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -v
Evaluation

Overall accuracy: 0.77

Per-class measures
          Precision  Recall  F1-score
อ์ไ(ติด)       0.00    0.00      0.00
อู             0.89    0.84      0.86
ศื(ติด)        1.00    0.50      0.67
ฐ(ติด)         1.00    0.94      0.97
ฅ              1.00    0.42      0.59
...             ...     ...       ...
ธ              1.00    0.62      0.77
ม              0.83    0.95      0.89
อื             0.60    0.83      0.70
ฒ              0.84    0.94      0.89
MACROS         0.81    0.77      0.76

[98 rows x 3 columns]
--------------------------------------------------------------------------------

Precision performance below 25th percentile (0.7143):
อ์ไ(ติด)    0.000000
อ้          0.695652
ซ           0.382353
ฮ           0.703704
ฐ           0.700000
อี          0.529412
ฤ           0.565217
อี้(ติด)    0.666667
อ์ใ(ติด)    0.211538
ส้(ติด)     0.444444
ๅ           0.625000
ศี(ติด)     0.555556
ค           0.540541
อ์โ(ติด)    0.600000
อื่(ติด)    0.560000
อุ          0.714286
ฏ           0.428571
ปั(ติด)     0.714286
ต           0.560000
ญ           0.714286
อ้โ(ติด)    0.714286
อ็ไ(ติด)    0.681818
อ้ใ(ติด)    0.363636
อึ่(ติด)    0.590909
ข           0.615385
อึ้(ติด)    0.631579
า           0.666667
อื          0.600000
Name: Precision, dtype: float64

Recall performance below 25th percentile (0.67):
อ์ไ(ติด)    0.000000
ศื(ติด)     0.500000
ฅ           0.421053
ไ           0.600000
สิ(ติด)     0.600000
อ่          0.647059
อี          0.450000
อี้(ติด)    0.444444
ฎ           0.142857
ๅ           0.277778
อ้ไ(ติด)    0.200000
อ์โ(ติด)    0.352941
อึ          0.411765
ฃ           0.666667
อ็          0.650000
ปั(ติด)     0.555556
อั          0.600000
ด           0.411765
ก           0.500000
ช           0.200000
อ้ใ(ติด)    0.444444
สั(ติด)     0.666667
อึ้(ติด)    0.521739
อี่(ติด)    0.095238
ธ           0.625000
Name: Recall, dtype: float64

F1-score performance below 25th percentile (0.681):
อ์ไ(ติด)    0.000000
ศื(ติด)     0.666667
ฅ           0.592593
ซ           0.520000
อี          0.486486
อี้(ติด)    0.533333
ฎ           0.250000
อ์ใ(ติด)    0.338462
ส้(ติด)     0.615385
ๅ           0.384615
ศี(ติด)     0.625000
ค           0.677966
อ้ไ(ติด)    0.322581
อ์โ(ติด)    0.444444
อึ          0.583333
อื่(ติด)    0.651163
ฏ           0.588235
ปั(ติด)     0.625000
ด           0.583333
ก           0.606061
ช           0.320000
อ้ใ(ติด)    0.400000
ข           0.666667
อึ้(ติด)    0.571429
อี่(ติด)    0.173913

Challenges:
On testing thai200normal, a new model would usually hover around 0.8 overall accuracy but sometimes?? Drop down to 0.01???