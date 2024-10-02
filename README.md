# adv-machine-learning-A1

#### Required modules

* argparse, numpy, os, pandas, PIL, sklearn.metrics, sklearn.preprocessing, torch, tqdm

#### Note

* '>>' indicates interactive terminal menu waiting for user input, example input is included below. Inputs are not proofed for exceptions, expected format is given in this file and on execution.

___

## Quickstart

### Running train_ML2A1.py:

(1) Train a new model and save model to file (optional).  
> $ python3 train_ML2A1.py -lg Thai -dpi 200 -ft normal -s Thai200normal_model

### Running test_ML2A1.py:

(1) Test a pre-trained model loaded from file.
> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -ld Thai200normal_model

(2) Train & test a new model on the same data specifications in succession, save to file (optional).  
> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal  

> No model loaded, train new model?  
> (y/n) >> y  
>
> File/pathname to save model to:  
> (None|str) >> Thai_200_normal  
>
> Train new model on same specifications as test data?  
> {'languages': ['Thai'], 'dpis': ['200'], 'fonts': ['normal']}  
> (y/n) >> y  
>
> Keep defaults for epochs (20) | batch_size (128) | learning rate (0.0025)?  
> (y/n) >> y  

___

## Detailed overview

### Training a model with train_ML2A1.py

To train a new model, run the train script and specify what data from the source repository to train on. This file also contains the NN model architecture.

The following arguments are required and determine which files will be extracted from the source directory as training data. They can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...).
* --languages (-lg):	Thai | English | Thai English
* --dpis (-dpi): 	200 | 300 | 400 | ...
* --fonts (-ft):	normal | bold | italic | bold_italic | ...

The following arguments are optional. --epochs, --batch_size, and --learning_rate alter behaviour during the training loop. --savefile allows the trained model to be saved under the passed filename/path. Finally, --source_dir may be used to specify a different directory to read the data from.
* --epochs (-ep):	any integer, defaults to 20
* --batch_size (-bs):	any integer, defaults to 128
* --learning_rate (-lr): float between 0-1, defaults to 0.0025
* --savefile (-s):	any filename/path, defaults to None
* --source_dir (-srcd):	pathname to directory for OCR data, on GPU defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/', on CPU defaults to './ThaiOCR/ThaiOCR-TrainigSet/'

#### Example execution:  
* Train new model, change default params, save new model to file
> $ python3 train_ML2A1.py -lg Thai English -dpi 200 300 -ft italic -ep 8 -bs 64 -lr 0.005 -s ThaiEn_200300_ita


### Testing a model with test_ML2A1.py

To test a model, run the test script and specify what data to test on. The script can either test a pre-trained model (passed with --loadfile), or will otherwise interactively ask for information to train a new model on execution.

The following arguments are required and determine which files will be extracted from the source directory to use during testing. They can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...).
* --languages (-lg):	Thai | English | Thai English
* --dpis (-dpi): 	200 | 300 | 400 | ...
* --fonts (-ft):	normal | bold | italic | bold_italic | ...

The following arguments are optional. --loadfile specifies where to find the pre-trained model, if invalid/left unspecified, the test script allows the user to specify information to train a new model using the train script. --verbose increases the amount of detail printed during model evaluation. Finally, --source_dir may be used to specify a different directory to read the data from.
* --loadfile (-ld):	any filename/path, defaults to None
* --verbose (-v):	on/off flag
* --source_dir (-srcd):	pathname to directory for OCR data, on GPU defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/', on CPU defaults to './ThaiOCR/ThaiOCR-TrainigSet/'

#### Example execution:  
* With model loaded from file  
> $ python3 test_ML2A1.py -lg Thai -dpi 300 -ft normal bold -ld ThaiEn_200300_ita_custom -v  

* With new model trained from scratch, same specifications, same params, saved to file
> $ python3 test_ML2A1.py -lg Thai -dpi 400 -ft normal bold -v 

> No model loaded, train new model?  
> (y/n) >> y
> 
> File/pathname to save model to:  
> (None|str) >> Thai400_normalbold
> 
> Train new model on same specifications as test data?  
> {'languages': ['Thai'], 'dpis': ['400'], 'fonts': ['normal', 'bold']}  
> (y/n) >> y
> 
> Keep defaults for epochs (20) | batch_size (128) | learning rate (0.0025)?  
> (y/n) >> y

* With new model trained from scratch, different specifications (Thai+English, 300dpi, normal+italic), different params, not saved
> $ python3 test_ML2A1.py -lg Thai -dpi 400 -ft bold -v   

> No model loaded, train new model?  
> (y/n) >> y
> 
> File/pathname to save model to:  
> (None|str) >> 
> 
> Train new model on same specifications as test data?  
> {'languages': ['Thai'], 'dpis': ['400'], 'fonts': ['bold']}  
> (y/n) >> n
> 
> Choose specifications for training data.  
> Enter single number, or combination (e.g. 1 -> English; 12 -> English+Thai).
> 
> Train on which language(s)?  
> {'1': 'English', '2': 'Thai'}  
> (int) >> 12
> 
> Train on which resolution(s)?  
> {'1': '200', '2': '300', '3': '400'}  
> (int) >> 2
> 
> Train on which font(s)?  
> {'1': 'normal', '2': 'bold', '3': 'italic', '4': 'bold_italic'}  
> (int) >> 13
> 
> Keep defaults for epochs (20) | batch_size (128) | learning rate (0.0025)?  
> (y/n) >> n
> 
> Number of epochs:  
> (None|int) >> 15
> 
> Size of batches:  
> (None|int) >> 64
> 
> Learning rate:  
> (None|float) >> 0.003


### Dataloader

Contains DataLoader, OCRData, and OCRModel classes, as well as ArgumentParser details. The former two are used to filter relevant files from source directory and transforming the data to the required format for both training and testing. OCRModel latter is a blueprint for the CNN model. File has no main function.

___

## Challenges:

### Data loading and overall structure

In general I did quite a bit of restructuring as time went on, because I wanted to minimise the amount of iterations over the datasets and ended up creating two classes to handle data loading/preparation. While a list of filenames and gold labels is always created, shuffled, and split for the entire dataset matching given input specifications (language, dpi, fonts), actual reading and processing of those images only occurs as needed: For training only the field in the training_data split are read, and likewise is the case for the testing split and the test script. Additionally, when the test script is used to both train and test a new model from scratch, if the data specifications are shared across training & testing data, walking the source directory and filtering out relevant filenames is only done once.

To ensure uniform input shapes across images (and resolutions), I decided to use the resize method for PIL Image objects and to resize to the average image size across all images for the given training specifications (i.e. all images for Thai-200dpi-normal). By storing this size as an attribute in OCRModel.img_dims, it became easily available to allow for appropriate resizing when testing a pre-trained model as well.

To get this average size in the DataLoader._get_avg_size function, I would originally open the images in a separate list first, then get the average size, and resize them. However, as PIL doesn't close images' files until their data has been read (for example after calling the .size method), this caused a problem of having too many opened files as the training data increased. So in the current version, the images are opened twice: once to get their size (DataLoader._get_avg_size), and then once again to resize them (OCRData._resize_img). Though most files are opened twice now, resizing and scaling (OCRData._transform_data) is only applied to images as needed (only training or only test set); in contrast the earlier version used to always transform the entire dataset before creating the splits, which creates a large redundancy when only the test set needs to be considered. Transforming the images (from filenames to resized and scaled tensors) remains a slow, perhaps even the slowest step, in the whole process though. Training epochs in return became sufficiently fast.

In general, the while functionality of being able to train a model during the test script execution, if no pre-trained model is loaded/found, may not have been expected and lead me to quite some redesigning all over the scripts (separate OCRData class, some 'mode' keyword arguments, get_model, get_new_train_specs, and init_train functions). But I found it quite handy for hyperparameter tuning and  easy re-running of experiments in general without saving tons of models. The interactive option for specifying new parameters from the test script execution is probably not the most sophisticated solution (and is not proofed for invalid inputs), but aided my workflow in the end a lot.

As for the extent of using the argument parser: I simply wrote it before we looked at using a json config file in class and by that point I had integrated it with the interactive design which seemed to suit it well.

Finally, after tuning the hidden size and activation function, the adjustment of the batch size and learning rate brought significant improvement across all model performances by around +5-10%.

By this time performance across nearly all experiments was uniformly above 0.9 (bar pure English models having some challenges, more on that later), so with performance not being an explicit criterium in this assignment, and time constraints, I elected to forego experimenting with other NN designs, dropout, or image padding sizes, for example.

### Odd bits

The output size of the CNN (OCRModel.output_size) is passed from the DataLoader class to the model instantiation in the train function and gets its value from the LabelEncoder.classes_ attribute. Before working with the label encoder, this value was created by using the set function on the list of gold labels. At some point during developing I changed the point at which those labels were transformed from strings from the source directory (to integers indexes) to and tensors to be usable for the train function. But as it turns out, the set() function does not work on tensors as expected and did not reduce the label list in the slightest, so for a while I predicted not over more than 10k+ classes. Surprisingly this didn't even hurt model performance too much, but it certainly improved once I noticed that error.

Something else I stumbled upon while working with pandas and tensors, is that transforming a column/series with integers into tensors (OCRData._transform_data, y = ...), fails when the index 0 is not present. This caused no issues during training, as the training set index would start at 0, but for the dev or test set this would suddenly throw errors. Apparently it's just a matter of inconsistent behaviour from torch (https://github.com/pytorch/pytorch/issues/51112) and resetting the index before transformation (pd.DataFram.reset_index(drop=True)) fixed this for me.

___

## Experiments

1) Thai normal text, 200dpi –> Thai normal text, 200dpi

> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -v  

Overall accuracy: 0.92

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| mean      | 0.92   | 0.92     | 0.92 |
| std       | 0.10   | 0.11     | 0.09 |
| min       | 0.62   | 0.47     | 0.58 |
| 25%       | 0.88   | 0.88     | 0.86 |
| 50%       | 0.94   | 0.95     | 0.95 |
| 75%       | 1.00   | 1.00     | 0.98 |
| max       | 1.00   | 1.00     | 1.00 |

Overview of 5 worst performing classes per measure:  

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | ---- | ---  | --- |
| า | 0.62  | อ์ไ(ติด) |0.47 | ซ |0.58  |
| ซ | 0.64  | ซ |0.53  | อ์ไ(ติด)| 0.59 |
| อี่(ติด)| 0.66 | ๅ| 0.55  | ๅ |0.63 |
| สี(ติด)| 0.67 | อื| 0.67  | า |0.70 |
| ช |0.70  | อึ่(ติด)| 0.71  | อื |0.78|

___

2) Thai normal 400 –> Thai normal 200

> $ python3 test_ML2A1.py -lg Thai -dpi 200 -ft normal -v

Overall accuracy: 0.93

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| mean  |     0.94 |   0.93 |      0.93|
| std   |     0.09 |   0.08 |     0.07|
| min   |     0.62 |   0.64 |     0.73|
| 25%   |     0.90 |   0.90 |     0.90|
| 50%   |     1.00 |   0.95 |     0.96|
| 75%   |     1.00 |   1.00 |     0.98|
| max   |     1.00 |   1.00 |      1.00|

Overview of 5 worst performing classes per measure:

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | ---- | ---  | --- |
| อ่ |  0.62 |อึ  |  0.64| อ่  |  0.73 |
| ๅ | 0.64 |า  |   0.65 | า  |   0.74 |
| สี(ติด)| 0.67 | ด  |   0.69 | ซ  |   0.74 | 
| ต    | 0.67 |ฏ   |  0.71 | ๅ  |   0.75 |
| ซ    | 0.72 | ฃ  |   0.71 | ฃ |    0.77 |

___

3) Thai normal 400 –> Thai bold 400

> $ python3 test_ML2A1.py -lg Thai -dpi 400 -ft bold -v -ld Thai400normal

Overall accuracy: 0.93

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| mean|       0.93 |    0.92 |     0.92 |
| std |       0.11 |   0.13  |    0.11 |
| min |       0.50 |   0.33  |    0.50 |
| 25% |       0.90 |   0.89  |    0.89 |
| 50% |       1.00 |   0.96  |    0.96 |
| 75% |       1.00 |   1.00  |    1.00 |
| max |        1.00|    1.00 |      1.00 |

Overview of 5 worst performing classes per measure:

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | --- | ---  | --- |
|อ์ใ(ติด)  |  0.50|ศั(ติด)  |   0.33|ศั(ติด)   |  0.50|
|ต     |      0.50|อ้ใ(ติด)   | 0.39|อ้ใ(ติด) |   0.56|
|อึ้(ติด) |   0.58|อี้(ติด)  |  0.61|อ์ใ(ติด)  |  0.64|
|อ้ไ(ติด) |   0.65|า       |    0.65|ต   |        0.67|
|ท      |     0.65|สี(ติด)  |   0.67|อึ้(ติด) |   0.67|

___

4) Thai bold –> Thai normal

> $ python3 test_ML2A1.py -lg Thai -dpi 200 300 400 -ft normal -v

Overall accuracy: 0.92

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
|mean|       0.92|    0.91|      0.92|
|std |       0.08|    0.09|      0.07|
|min |       0.61|    0.63|      0.73|
|25% |       0.88|    0.87|      0.88|
|50% |       0.94|    0.94|      0.93|
|75% |       0.99|    0.97|      0.96|
|max |       1.00|    1.00|      1.00|

Overview of 5 worst performing classes per measure:

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | --- | ---  | --- |
|ข    |       0.61|อ์ใ(ติด) |   0.63|ฃ   |        0.73|
|อ์ไ(ติด)  |  0.71|ฃ      |     0.65|ข    |       0.73|
|ๅ    |       0.72|ซ   |        0.65|า   |        0.73|
|อี้(ติด)|    0.73|ส้(ติด)  |   0.66|อ์ใ(ติด)   | 0.75|
|อ่    |      0.77|า   |        0.66|ซ     |      0.76|

___

5) All Thai –> All Thai

> $ python3 test_ML2A1.py -lg Thai -dpi 200 300 400 -ft normal bold italic bold_italic -v

Overall accuracy: 0.99

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
|mean|       0.99|    0.99|      0.99|
|std|        0.02|    0.02|      0.01|
|min|        0.88|    0.92|      0.92|
|25%|        0.98|    0.98|      0.99|
|50%|        0.99|    0.99|      0.99|
|75%|        1.00|    1.00|      1.00|
|max|        1.00|    1.00|      1.00|

Overview of 5 worst performing classes per measure:

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | --- | ---  | --- |
|อ่   |       0.88|อุ   |       0.92|อ่  |  0.92|
|อำ   |       0.94|อี่(ติด) |   0.94|อำ  |  0.94|
|า     |      0.94|อำ   |       0.95|า |    0.95|
|อื       |   0.95|ๅ      |     0.95|อุ |   0.96|
|อึ่(ติด)  |  0.95|อี        |  0.95|ๅ  |   0.96|

___

6) Thai & English normal –> Thai & English normal

> $ python3 test_ML2A1.py -lg English Thai -dpi 200 300 400 -ft normal -v

Overall accuracy: 0.97

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
|mean|       0.97|    0.97|      0.97|
|std|        0.07|    0.07|      0.07|
|min|        0.37|    0.51|      0.47|
|25%|        0.97|    0.98|      0.97|
|50%|        0.99|    0.99|      0.99|
|75%|        1.00|    1.00|      1.00|
|max|        1.00|    1.00|      1.00|

Overview of 5 worst performing classes per measure:

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | --- | ---  | --- |
|l |    0.37|i |    0.51|l |    0.47|
|I  |   0.68|.  |   0.64|i  |   0.59|
|i   |  0.71|l   |  0.65|I   |  0.71|
|อ่   | 0.83|อ่   | 0.66|อ่   | 0.74|
|V    | 0.83|I   |  0.74|.  |   0.74|

___


7) All styles –> All styles

> $ python3 test_ML2A1.py -lg English Thai -dpi 200 300 400 -ft normal italic bold bold_italic -v

Overall accuracy: 0.97

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
|mean|       0.97|    0.97|      0.97|
|std        |0.07    |0.06      |0.06|
|min|        0.54|    0.59|      0.56|
|25%        |0.97    |0.97      |0.97|
|50%        |0.99|    0.99|      0.99|
|75%        |1.00    |1.00      |1.00|
|max|        1.00|    1.00|      1.00|

Overview of 5 worst performing classes per measure:

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | --- | ---  | --- |
|l |    0.54|l|     0.59|l|     0.56|
|i  |   0.58|i |    0.65|i |    0.61|
|อ่   | 0.73|I   |  0.72|อ่   | 0.74|
|I    | 0.75|อ่   | 0.74|I    | 0.74|
|.     |0.76|.    | 0.75|.    | 0.76|

___

### Other

* English 200 normal –> English 200 normal. 
> $ python3 test_ML2A1.py -lg English -dpi 200 -ft normal -v

Overall accuracy: 0.92

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
|mean       |0.93    |0.92      |0.92|
|std        |0.13|    0.15|      0.13|
|min        |0.42    |0.33   |   0.43|
|25%        |0.94|    0.90|      0.88|
|50%        |1.00    |1.00   |   0.98|
|75%        |1.00|    1.00|      1.00|
|max|        1.00|    1.00|      1.00|

Overview of 5 worst performing classes per measure:

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | --- | ---  | --- |
|l|    0.42 |i   | 0.33|i|    0.43|
|w |   0.56|W   | 0.48|l  |  0.57|
|i  |  0.62|I  |  0.53|W   | 0.59|
|o   | 0.67|O |   0.57|O    |0.65|
|x    |0.73|X|    0.69|w    |0.67|

As to be expected, the model struggles to tell characters apart which are similar in their upper- vs lowercase formats (e.g. W-w, O-o). From the evaluation above the model appears to begin favouring either one of the classes: Higher precision for o/w are accompanied by low recall for their uppercase equivalent. A similar issue emerges for lowercase i and lowercase L; some sections in the data seem to stem from fonts with serifs, which might aid classification for some cases, for example capital I. Assuming the source data is sorted correctly, a look at the English/105/200 ('i') and English/108/200 ('l') shows that the dot above the i – which I would have assumed must enable the model to distinguish this form from the more similar l/I at times (ironically in GitHub these look identical, lowercase L / uppercase i for clarity) – is hardly ever preserved at all, further conflating these two forms. This pattern seems to be present across all resolutions, so it is possible this is an issue with the original cropping of letters, which appears to have cut off the i dots. Making this issue similar to the upper- vs lowercase problem.

* English 300 normal –> English 300 normal
> $ python3 test_ML2A1.py -lg English -dpi 300 -ft normal -v

Overall accuracy: 0.96

Overview of measures across classes:

| | Precision | Recall | F1-score |
| --- | --- | --- | --- |
|mean       |0.97    |0.96      |0.96|
|std|        0.08|    0.10|      0.08|
|min        |0.51    |0.55   |   0.66|
|25%        |0.99|    0.99|      0.95|
|50%        |1.00    |1.00   |   1.00|
|75%        |1.00|    1.00|      1.00|
|max|        1.00|    1.00|      1.00|

Overview of 5 worst performing classes per measure:

| Precision | | Recall | | F1-Score | |
| --- | ---| --- | --- | ---  | --- |
|i|    0.51|I    |0.55|i|    0.66|
|v |   0.77|l   | 0.56|l |   0.68|
|w  |  0.83|V  |  0.75|I  |  0.70|
|l   | 0.84|W |   0.81|V   | 0.84|
|o    |0.89|.|    0.84|v    |0.86|

Increasing the resolution to 300dpi brings in a couple of points int he overall accuracy, but what's more is that the scores for the lowest performing classes visibly rises with the minimal F1-score across all classes going from 0.43 to 0.66 (lowercase i in both cases). The precision score for lowercase L also suggests that even without the i-dot, the model is able to recognise this character on a more-than-chance basis by proportions alone, given a high enough (here: >200 dpi) resolution.

Scores for W-w and O-o also rise; though most of them are still represented in the bottom 5 in at least one category, scores above 0.8 are promising. Lastly, as suspected above, uppercase i also belongs to this group of lower performing characters.

Training and testing on different fonts (normal->bold, both 200dpi, overall accuracy 0.83) exacerbates these issues (upper- vs lowercase similarity, lower L/i/upper i case) even more with the bottom 5 cases for each measure staying below 0.6 and the lowest per-class measure being F1-Score for lower L (0.17).

___ 

## Bonus part

> $ python3 test_ML2A1.py -lg Thai Numeric Special English -dpi 200 300 400 -ft normal bold bold_italic italic -v
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

| Precision | Recall | F1-Score |
| ------ | ------  | ------ |
||||
||||
||||
||||
||||

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


