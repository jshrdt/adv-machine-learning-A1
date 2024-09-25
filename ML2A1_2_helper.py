## helper funcs
import os
from PIL import Image
import random
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
    
def get_ids(src_dir):
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

class DataLoader:
    def __init__(self, src_dir, data_specs, size_to=None, limit=None):
        self.le = LabelEncoder()  # filenr 2 idx & idx 2 filenr
        self.raw_data = self._read_data(src_dir, data_specs)
        if limit:
            output = self._transform_data(self.raw_data[:limit], size_to)
        else:
            output = self._transform_data(self.raw_data, size_to)
        self.train, self.dev, self.test, self.avg_size = output
        self.n_classes = len(set(self.train['labels']))
        
        self.filenr2char = get_ids(src_dir)  # filenr 2 char
             
    def idx_to_char(self, idx):
        # idx -> reverse transform (to filenr) -> filenr2char
        return self.filenr2char[self.le.inverse_transform([idx])[0]]

    def __len__(self):
        return len(self.raw_data)
                
    def _read_data(self, src_dir, specs):
        print('selecting files:', specs)
        # extract relevant filenames, limited by: lg(var) + charID(all) + dpi(1) + font(1/all) -> get all
        fileinfo = list()
        for root, dirs, files in os.walk(src_dir):
            root_sep = root.split('/')
            # only search directories fulfilling the specs for language, dpi & font
            # leverage uniform format across folders
            if ((root_sep[-4] in specs['Language(s)'])
                and (root_sep[-2] in specs['DPI']) 
                and (root_sep[-1] in specs['Font(s)'])):
                # then extract filename + gold label character identifier
                # add: save the id nrs to use as index filter in charid_dict
                for fname in files:
                    if fname.endswith('bmp'):
                        fileinfo.append((os.path.join(root, fname),
                                         fname[-7:-4]))
                        
        # seeded shuffling of data, to ensure any test data is always unseen
        random.Random(11).shuffle(fileinfo)
        
        return fileinfo
        
    def _transform_data(self, data_list, size_to=None):
        print('transforming data')
        # make possible to pass avg_size in here, will be saved in train func,
        # then applied for any test data
        
        # extract and transform images
        raw_imgs = [Image.open(item[0]) for item in data_list]
        
        # find avg size of train files to decide what model should resize input to
        if size_to:
            avg_size = size_to
        else:
            avg_size = (round(sum([img.size[0] for img in raw_imgs])
                              / len(raw_imgs)),
                        round(sum([img.size[1] for img in raw_imgs])
                              / len(raw_imgs)))
        # resize images to avg size, recode image pixels from True/False to 1/0
        # then transform to matrix of np arrays
        np_imgs = np.array([np.array([[0 if dot else 1 for dot in line] for line
                                      in np.array(img.resize((avg_size)))])
                            for img in raw_imgs])
        # rescale each value to be between 0, 1.
        imgs = self._scale_imgs(np_imgs)

        # fit label encoder & transform labels to unique idx
        file_labels = [item[1] for item in data_list]
        self.le.fit(file_labels)
    ## tbd send to device, just after rescale
        idx_labels = torch.tensor(self.le.transform(file_labels))
        
        # split into train, dev, and test set 
        # & transform each set's imgs into single scaled matrix
        # set cutoff points
        train_len = int(len(idx_labels)*0.8)
        dev_len = int(train_len + len(idx_labels)*0.1)
        # split data
        train_data = {'imgs': imgs[:train_len],
                      'labels': idx_labels[:train_len]}
        dev_data = {'imgs': imgs[train_len:dev_len],
                    'labels': idx_labels[train_len:dev_len]}
        test_data = {'imgs': imgs[dev_len:],
                     'labels': idx_labels[dev_len:]}
        
        return train_data, dev_data, test_data, avg_size
    
    def _scale_imgs(self, imgs):
        size = imgs.shape
        scaled_imgs = torch.tensor(StandardScaler().fit_transform(imgs.reshape(
                        size[0], size[1]*size[2])).reshape(size)).float()
        return scaled_imgs


class MyBatcher:
    def  __init__(self, data, batch_size, device):
        self.batches = self._batch(data, batch_size)
        self.device = device
        
    def _batch(self, data, batch_size):
        permutation = torch.randperm(data['imgs'].size()[0])# device=self.device)
        permX = data['imgs'][permutation]
        permy = data['labels'][permutation]
        
        batches = [(permX[i*batch_size:(i+1)*batch_size],
                         permy[i*batch_size:(i+1)*batch_size])
                    for i in range(int(data['imgs'].size()[0]/batch_size))]

        return batches
    