## helper funcs
import os
from PIL import Image
import random
import numpy as np
import torch
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class OCRData:
    def __init__(self, data, device, size_to=None):
        self.device = device
        self.avg_size = size_to
        self.transformed = self._transform_data(data)
        
    def _transform_data(self, data):
        # Resize images to average size of training data, recode pixels from 
        # True/False to 1/0, then transform to matrix of np arrays.
        imgs = np.array([np.array([[0 if dot else 1 for dot in line] for line
                                   in self._resize_img(item)])
                         for item in data['files']])
        
        # Rescale values in image matrix. Transform X, y to tensor, send to device.
        X = torch.tensor(self._scale_imgs(imgs)).float().to(self.device)
        y = torch.tensor(data['labels']).to(self.device)

        transformed_data = {'imgs': X, 'labels': y}
        
        return transformed_data
        
    def _resize_img(self, item):
        return np.array(Image.open(item).resize((self.avg_size)))    
    
    def _scale_imgs(self, imgs):
        size = imgs.shape
        scaled_imgs = StandardScaler().fit_transform(
                        imgs.reshape(size[0], size[1]*size[2])).reshape(size)
                                   
        return scaled_imgs

        
class DataLoader:
    def __init__(self, src_dir, data_specs):
        self.le = LabelEncoder()  # filenr 2 idx & idx 2 filenr
        self.data_df = self._read_data(src_dir, data_specs)
        self.train, self.dev, self.test = self._split_data(self.data_df)
        self.avg_size = self._get_avg_size(self.train)
        
        self.n_classes = len(self.le.classes_)
        self.filenr2char = self._get_ids(src_dir)  # filenr 2 char
             
    def idx_to_char(self, idx):
        # idx -> reverse transform (to filenr) -> filenr2char
        return self.filenr2char[self.le.inverse_transform([idx])[0]]

    def __len__(self):
        return len(self.raw_data)
                
    def _read_data(self, src_dir, specs):
        # extract relevant filenames, limited by: lg(var) + charID(all) + dpi(1) + font(1/all) -> get all
        fileinfo = list()
        for root, dirs, files in os.walk(src_dir):
            root_sep = root.split('/')
            # only search directories fulfilling the specs for language, dpi & font
            # leverage uniform format across folders
            if ((root_sep[-4] in specs['languages'])
                and (root_sep[-2] in specs['dpis']) 
                and (root_sep[-1] in specs['fonts'])):
                # then extract filename + gold label character identifier
                # add: save the id nrs to use as index filter in charid_dict
                for fname in files:
                    if fname.endswith('bmp'):
                        fileinfo.append((os.path.join(root, fname),fname[-7:-4]))
                        
        # seeded shuffling of data, to ensure any test data is always unseen
        fileinfo_df = pd.DataFrame(fileinfo, columns=['files', 'labels'])
        fileinfo_df = fileinfo_df.sample(frac=1, random_state=11, ignore_index=True)
                
        return fileinfo_df
            
    def _split_data(self, data_df):

        # Fit label encoder on y labels & transform y labels to unique idx.
        self.le.fit(data_df['labels'])
        idx_labels = self.le.transform(data_df['labels'])
                
        # Set cutoff points for data splits.
        train_len = int(len(data_df)*0.8)
        dev_len = int(train_len + len(data_df)*0.1)
        
        # Split data.
        train_data = {'files': data_df['files'][:train_len],
                      'labels': idx_labels[:train_len]}
        dev_data = {'files': data_df['files'][train_len:dev_len],
                    'labels': idx_labels[train_len:dev_len]}
        test_data = {'files': data_df['files'][dev_len:],
                     'labels': idx_labels[dev_len:]}
        
        return train_data, dev_data, test_data
        
    def _get_avg_size(self, train_df):
        # Find average size of training files to resize input images to.
        sizes = [Image.open(file).size for file in train_df['files']]
        
        avg_size = (round(sum([size[0] for size in sizes]) / len(sizes)),
                    round(sum([size[1] for size in sizes]) / len(sizes)))
        
        return avg_size
    
    def _get_ids(self, src_dir):
        # get id2char mapping, files are identical across languages
        # always get all info from txt file, in second step filter according to train specs
        for root, dirs, files in os.walk(src_dir+'English'+'/'):
            with open(root+files[0], 'rb') as f:
                legend = f.read().strip().decode(encoding='cp874',
                                                errors='backslashreplace')  #fine for this assignments, since non alpha shoudl b ignored
                
                id2char = {(item.split()[0] if (int(item.split()[0]) < 111
                                                or int(item.split()[0]) >= 123)
                            else str(int(item.split()[0])+1)
                            ): item.split()[1]
                        for item in legend.split('\n')}
                # manually add characters contained in data but missing from txt file 
                id2char['111'] = 'o'
                id2char['195'] = 'ร'
        
                return id2char


class MyBatcher:
    def  __init__(self, data, batch_size):
        self.batches = self._batch(data, batch_size)
        
    def _batch(self, data, batch_size):
        permutation = torch.randperm(data['imgs'].size()[0])
        permX = data['imgs'][permutation]
        permy = data['labels'][permutation]
        
        batches = [(permX[i*batch_size:(i+1)*batch_size],
                         permy[i*batch_size:(i+1)*batch_size])
                    for i in range(int(data['imgs'].size()[0]/batch_size))]

        return batches
    
if __name__=="__main__":
    pass
    