# adv-machine-learning-A1

## Dataloader

## Train
To train a new model, call the train script and specify what data to use from training repository. 
The following required arguments can take a singular or their respective values, or any combination of them, separated by a whitespace (example shown for --languages, thereafter indicated by ...)
	--languages (-lg):	Thai | English | Thai English
	--dpis (-dpi): 		200 | 300 | 400 | ...
	--fonts (-ft):		normal | bold | italic | italic_bold | ...

The following arguments are optional. The first two can alter behaviour during the training loop. --Savefile allows the trained model to be saved under the passed filename/path. The final argument may be used to specify a different directory to read the source data from.
	--epochs (-ep):		any integer, defaults to 5
	--batch_size (-bs):	any integer, defaults to 32
	--savefile (-s)		any filename/path, defaults to None
	--source_dir (-srcd):	pathname to directory for OCR data, defaults to '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'


Will assume:     
	src_dir = '/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/'


## Test

train+test

Test only

Optional: load m from file (->save file), else: train from scratch