## Class definitons & parser ##
# Imports
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler


def parse_input_args(mode='train'):
    # Initialise argument parser and define command line arguments.
    parser = argparse.ArgumentParser(formatter_class=
                                    argparse.ArgumentDefaultsHelpFormatter)
    # Must-have
    parser.add_argument('-lg', '--languages', nargs='+', required=True,
                        help='Languages to train on. English | Thai')
    parser.add_argument('-dpi', '--dpis', nargs='+', required=True,
                        help='DPI formats to train on. 200 | 300 | 400')
    parser.add_argument('-ft', '--fonts', nargs='+', required=False,
                        help='Fonts to train on. normal|bold|italic|bold_italic')
    # Optional
    parser.add_argument('-srcd', '--source_dir', help='Pass a custom source'
                        + ' directory pointing to image data.')

    if mode=='train':
        # Optional for train script.
        parser.add_argument('-ep', '--epochs', type=int, default=20,
                            help='Number of training epochs. Any.')
        parser.add_argument('-bs', '--batch_size', type=int, default=128,
                            help='Size of training data batches. Any.')
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025,
                            help='Learning rate to use during training. 0-1')
        parser.add_argument('-s', '--savefile', default=None,
                            help='Enable saving of model, specify filename/path.')

    else:
        # Optional for test script.
        parser.add_argument('-ld', '--loadfile', default=None,
                            help='Filename/path to load pretrained model from.')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Pass to receive per-class evaluation metrics '
                            + 'and lowest performing classes')

    return parser


class DataLoader:
    def __init__(self, src_dir: str, data_specs: dict):
        self.le = LabelEncoder()
        self.train, self.dev, self.test = self._read_data(src_dir, data_specs)
        self.avg_size = self._get_avg_size(self.train)
        self.n_classes = len(self.le.classes_)
        self.filenr2char = self._get_mapping(src_dir)

    def idx_to_char(self, idx: int) -> str:
        """Return Latin/Thai character corresponding to unique label idx."""
        return self.filenr2char[self.le.inverse_transform([idx])[0]]

    def _read_data(self, src_dir: str, specs: dict) -> \
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return training, dev, and test data df splits of data
        according to specs. Fit label encoder and encode file labels.
        """
        fileinfo = list()
        # Walk source directory, only searching folders fulfilling specs.
        for root, dirs, files in os.walk(src_dir):
            root_sep = root.split('/')
            # Check directory name against specs.
            # Maxiamlly exploitative of directory structure/file naming
            # convention in this data set specifially.
            if ((root_sep[-4] in specs['languages'])
                and (root_sep[-2] in specs['dpis']) 
                and (root_sep[-1] in specs['fonts'])):
                # Extract all image files + character identifier labels.
                for fname in files:
                    if fname.endswith('bmp'):
                        fileinfo.append((os.path.join(root, fname),
                                         fname[-7:-4]))
        print('Preprocessing data...')
        # Transform files + labels data to pd df for easy accessing.
        data_df = pd.DataFrame(fileinfo, columns=['files', 'labels'])

        # Fit label encoder on labels & transform each to a unique idx number.
        self.le.fit(data_df['labels'])
        data_df['labels'] = self.le.transform(data_df['labels'])

        # Seed shuffling of data, to include all character types per split, but
        # still ensure any test data is always unseen. Strong assumption that 
        # due to size of data this will lead to classes being represented about 
        # equally in each split.
        data_df = data_df.sample(frac=1, random_state=11, ignore_index=True)

        # Set cutoff points.
        train_len = int(len(data_df)*0.8)
        dev_len = int(train_len + len(data_df)*0.1)

        # Split data.
        train_data = data_df[:train_len]
        dev_data = data_df[train_len:dev_len]
        test_data = data_df[dev_len:]

        return train_data, dev_data, test_data

    def _get_avg_size(self, train_df: pd.DataFrame) -> tuple[int, int]:
        """Find average size of training files to resize input images to."""
        # Collect file widths and heights.
        sizes_w, sizes_h = list(), list()
        for file in train_df['files']:
            size = Image.open(file).size
            sizes_w.append(size[0])
            sizes_h.append(size[1])
        # Get average file size in training dataset.
        avg_size = (round(sum(sizes_w) / len(train_df['files'])),
                    round(sum(sizes_h) / len(train_df['files'])))

        return avg_size

    def _get_mapping(self, src_dir: str) -> dict:
        """Return dictionary mapping file labels to English/Thai character."""
        # Get mapping information from txt file.
        for root, dirs, files in os.walk(src_dir+'English'+'/'):
            with open(root+files[0], 'rb') as f:
                legend = f.read().strip().decode(encoding='cp874',
                                                 errors='backslashreplace')
            break

        # Create label: char dictionary, manually fixing some flaws from txt file.
        label2char = {(item.split()[0] if (int(item.split()[0]) < 111
                                        or int(item.split()[0]) >= 123)
                       else str(int(item.split()[0])+1)
                       ): item.split()[1]
                      for item in legend.split('\n')}
        # 111 & 195 are missing from txt file but present as images in data. 
        label2char['111'] = 'o'
        label2char['195'] = 'ร'

        return label2char


class OCRData:
    def __init__(self, data: pd.DataFrame, device: torch.device,
                 size_to: tuple[int, int] = None, mode=None):
        self.device = device
        self.avg_size = size_to
        self.mode = mode
        self.transformed = self._transform_data(data)

    def _transform_data(self, data: pd.DataFrame) -> dict:
        """Resize, encode, rescale images to numpy matrix. Transform image and
        labels to tensors and send to device."""
        # Resize images to average size of training data, recode pixels from
        # True/False to 1/0, then transform to matrix of np arrays.
        imgs = np.array([np.array([[0 if dot else 1 for dot in line]
                                   for line in self._resize_img(item)])
                         for item in data['files']])

        # Rescale image matrix. Send tensors of imgs (X) and labels (y) to device.
        X = torch.tensor(self._scale_imgs(imgs)).float().to(self.device)
        y = torch.tensor(data['labels'].reset_index(drop=True)).to(self.device)

        transformed_data = {'imgs': X, 'labels': y}

        return transformed_data

    def _resize_img(self, img: str) -> np.ndarray:
        """Opens file and resizes image to average size from training data."""
        # Used in bonus task only.
        if self.mode == 'bonus':
            # convert back to image object to resize, then return to expected input
            img_obj = Image.fromarray((img * 255).astype(np.uint8))
            img_arr = np.array(img_obj.convert('1').resize((self.avg_size)))
            # needed because image's 1/0 encoding was flipped during cropping
            converted = np.array([[0 if dot else 1 for dot in line]
                                  for line in img_arr])
            return converted

        # Used during main task.
        else:
            return np.array(Image.open(img).resize((self.avg_size)))    

    def _scale_imgs(self, imgs: np.ndarray) -> np.ndarray:
        """Rescales values in image matrix for convolutional network."""
        size = imgs.shape
        scaled_imgs = StandardScaler().fit_transform(
                        imgs.reshape(size[0], size[1]*size[2])).reshape(size)

        return scaled_imgs


class OCRModel(nn.Module):
    def __init__(self, n_classes: int, img_dims: tuple[int, int], idx_to_char):
        super(OCRModel, self).__init__()
        # Initialise model params.
        self.input_size = img_dims[0]*img_dims[1]
        self.hsize_1 = int(self.input_size/2)
        self.output_size = n_classes
        self.idx_to_char = idx_to_char
        self.img_dims = img_dims  # used to resize test data

        # Define net structure.
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Flatten())

        self.net2 = nn.Sequential(
            nn.Linear(self.input_size, self.hsize_1),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hsize_1, self.output_size),
            nn.LogSoftmax(dim=1))

    def forward(self, X: torch.Tensor, mode: str = None) -> torch.Tensor|str:
        # Reshape batched input and pass through for convolutional layer.
        net1_output = self.net1(X.reshape(1, X.shape[0], X.shape[1]*X.shape[2]))
        # Send output through classifier.
        preds = self.net2(net1_output.reshape(-1, self.input_size))

        if mode=='train':
            # Return LogSoftMax distribution over classes.
            return preds
        else:
            # Return predicted character.
            return self.idx_to_char(int(preds.argmax()))


if __name__=="__main__":
    pass
