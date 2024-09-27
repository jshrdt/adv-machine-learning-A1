## helper classes ##
# Imports
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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
                
    def _read_data(self, src_dir: str,
                   specs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return training, dev, and test data df splits of data according to specs.
        Fit label encoder and encode file labels.
        """
        fileinfo = list()
        # Walk source directory, only searching folders fulfilling specs.
        for root, dirs, files in os.walk(src_dir):
            root_sep = root.split('/')
            if ((root_sep[-4] in specs['languages'])
                and (root_sep[-2] in specs['dpis']) 
                and (root_sep[-1] in specs['fonts'])):
                # Extract all image files + character identifier labels.
                for fname in files:
                    if fname.endswith('bmp'):
                        fileinfo.append((os.path.join(root, fname), fname[-7:-4]))
        
        # Transform files + labels data to pd df for easy accessing.
        data_df = pd.DataFrame(fileinfo, columns=['files', 'labels'])
     
        # Fit label encoder on labels & transform each to a unique idx number.
        self.le.fit(data_df['labels'])
        data_df['labels'] = self.le.transform(data_df['labels'])

        # Seed shuffling of data, to include all character types per split, but 
        # still ensure any test data is always unseen.
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
        sizes = [Image.open(file).size for file in train_df['files']]
        
        avg_size = (round(sum([size[0] for size in sizes]) / len(sizes)),
                    round(sum([size[1] for size in sizes]) / len(sizes)))
        
        return avg_size
    
    def _get_mapping(self, src_dir: str) -> dict:
        """Return dictionary mapping file labels to EnglishTthai character."""
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
                 size_to: tuple[int, int]=None):
        self.device = device
        self.avg_size = size_to
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
        return np.array(Image.open(img).resize((self.avg_size)))    
    
    def _scale_imgs(self, imgs: np.ndarray) -> np.ndarray:
        """Rescales values in image matrix for convolutional network."""
        size = imgs.shape
        scaled_imgs = StandardScaler().fit_transform(
                        imgs.reshape(size[0], size[1]*size[2])).reshape(size)
                                   
        return scaled_imgs
    
if __name__=="__main__":
    pass
    