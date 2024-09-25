# adv-machine-learning-A1

Quick overview | main functionality:

Train: 
(1) Train new model, optional: save to file.

Test: 
(1) Test pre-trained model loaded from file.
(2) Tain & test new model, optional: save to file.

## Dataloader

TODO

## Train
To train a new model, call the train script and specify what data to use from training repository. 
The following required arguments can take a singular of the listed values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...)
	--languages (-lg):	Thai | English | Thai English
	--dpis (-dpi): 		200 | 300 | 400 | ...
	--fonts (-ft):		normal | bold | italic | italic_bold | ...

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