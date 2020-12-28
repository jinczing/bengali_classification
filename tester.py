'''
!!!!!!!!!!!!!!!!!!!!!!!Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!

This code is not used in final submission!!!



'''



# test
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import csv
import os
import cv2
import pandas as pd
import time
from timm.models import create_model
import torch.autograd.profiler as profiler
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from model import MyModel, MultiHeadModel
from torch.autograd import Variable


class BengaliDataset(Dataset):
  def __init__(self, label_csv, unique_csv, train_folder, transforms, cache=True, test=True):
    self.label_csv = label_csv
    self.unique_csv = unique_csv
    self.train_folder = train_folder
    self.label = pd.read_csv(self.label_csv)
    self.label = self.label[self.label['component']=='grapheme_root']
    self.label = self.label.reset_index(drop=True)
    unique_df = pd.read_csv(self.unique_csv)
    #names = self.label[self.label['component']=='grapheme_root']
    self.names = self.label['image_id'].values
    self.uniques = unique_df.grapheme.unique()
    self.transforms = transforms
    self.img = [None] * self.label.shape[0]
    self.test = test

    if cache:
      self.cache_images()

  def cache_images(self):
    pbar = tqdm.tqdm(range(self.label.shape[0]), position=0, leave=True)
    pbar.set_description('caching images...')
    for i in pbar:
      self.img[i] = self.load_image(i)

  def load_image(self, idx):
    img = self.img[idx]
    if img is None:
      name = self.label.loc[idx]['image_id']
      #img = cv2.imread(os.path.join(self.train_folder, name+'.jpg'), cv2.IMREAD_GRAYSCALE)
      img = Image.open(os.path.join(self.train_folder, name+'.jpg'))
      return self.transforms(img)
    else:
      return self.transforms(img)

  def __getitem__(self, idx):
    if not self.test:
      img = self.load_image(idx)
      root = self.label.loc[idx]['grapheme_root']
      consonant = self.label.loc[idx]['consonant_diacritic']
      vowel = self.label.loc[idx]['vowel_diacritic']
      unique = np.where(self.uniques == self.label.grapheme[idx])[0][0]
      return transforms.ToTensor()(img), root, consonant, vowel, unique
    else:
      img = self.load_image(idx)
      root = 0
      consonant = 0
      vowel = 0
      unique = 0
      return transforms.ToTensor()(img), root, consonant, vowel, unique

  def __len__(self):
    return self.label.shape[0]


class MultiHeadTester:
    def __init__(self,
               dataset_path='./drive/MyDrive/datasets/car classification/train_data', 
               batch_size=1, 
               model_name='tf_efficientnet_b3_ns', 
               test_csv='./train_labels.csv', 
               unique_csv='./train_labels.csv',
               output_dir='../drive/MyDrive/ckpt/grapheme/submission.csv',
               ckpt='../drive/MyDrive/ckpt/grapheme/20.pth'):

        # initialize attributes
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.model_name = model_name
        self.test_csv = test_csv
        self.unique_csv = unique_csv
        self.output_dir = output_dir
        self.ckpt = ckpt
        
        if model_name == 'tf_efficientnet_b0_ns':
            self.input_size = (224, 224)
        elif model_name == 'tf_efficientnet_b3_ns':
            self.input_size = (300, 300)
        elif model_name == 'tf_efficientnet_b4_ns':
          scaleelf.input_size = (380, 380)
        elif model_name == 'tf_efficientnet_b6_ns':
            self.input_size = (528, 528)
        else:
            raise Exception('non-valid model name')
        
        # Compose transforms
        transform = []
        transform += [transforms.Resize(self.input_size)]
        self.transform = transforms.Compose(transform)

        self.test_dataset = BengaliDataset(self.test_csv, self.unique_csv, self.dataset_path, self.transform, cache=True)
        self.names = self.test_dataset.names
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_root = MyModel(self.input_size, self.model_name, 168, pretrained=True, dropout=0).to('cuda')
        self.model_consonant = MyModel(self.input_size, self.model_name, 11, pretrained=True, dropout=0).to('cuda')
        self.model_vowel = MyModel(self.input_size, self.model_name, 18, pretrained=True, dropout=0).to('cuda')
        self.model_multihead = MultiHeadModel(self.input_size, self.model_name, pretrained=True, dropout=0).to('cuda')

        ckpt = torch.load(self.ckpt)
        self.model_root.load_state_dict(ckpt['model_root_state_dict'])
        self.model_consonant.load_state_dict(ckpt['model_consonant_state_dict'])
        self.model_vowel.load_state_dict(ckpt['model_vowel_state_dict'])
        self.model_multihead.load_state_dict(ckpt['model_multihead_state_dict'])

    def test(self):
        pbar = tqdm.tqdm(self.test_dataloader)
        pbar.set_description('testing process')
        self.model_root.eval()
        self.model_consonant.eval()
        self.model_vowel.eval()
        self.model_multihead.eval()
        output_roots = []
        output_consonants = []
        output_vowels = []
        count = 0
        with torch.no_grad():
            for it, data in enumerate(pbar):
                inputs = data[0].to(self.device)
                inputs = inputs.repeat(1, 3, 1, 1)
                roots = data[1].to(self.device).long()
                consonants = data[2].to(self.device).long()
                vowels = data[3].to(self.device).long()
                uniques = data[4].to(self.device).long()

                root_preds, root_preds_2 = self.model_root(inputs, roots)
                self.model_root.zero_grad()

                consonant_preds, consonant_preds_2 = self.model_consonant(inputs, consonants)
                self.model_consonant.zero_grad()

                vowel_preds, vowel_preds_2 = self.model_vowel(inputs, vowels)
                self.model_vowel.zero_grad()
                
                root, consonant, vowel, unique, root2, consonant2, vowel2, unique2 = self.model_multihead(inputs, roots, consonants, vowels, uniques)
                self.model_multihead.zero_grad()


                unique_prob = F.softmax(unique, dim=1)
                
                for index in range(inputs.shape[0]):
                  #print('unique:', unique_prob.max(-1).values[index])
                  if unique_prob.max(-1).values[index] > 0.5:
                    output_roots.append(root.argmax(-1)[index].item()) 
                    output_consonants.append(consonant.argmax(-1)[index].item())
                    output_vowels.append(vowel.argmax(-1)[index].item()) 
                  else:
                    output_roots.append(root_preds.argmax(-1)[index].item())
                    output_consonants.append(consonant_preds.argmax(-1)[index].item())
                    output_vowels.append(vowel_preds.argmax(-1)[index].item())

        row_id, target = [], []
        for iid, r, v, c in zip(self.names, output_roots, output_consonants, output_vowels):
            row_id.append(iid + '_grapheme_root')
            target.append(int(r))
            row_id.append(iid + '_vowel_diacritic')
            target.append(int(v))
            row_id.append(iid + '_consonant_diacritic')
            target.append(int(c))
            count += 1

        sub_fn = self.output_dir
        sub = pd.DataFrame({'row_id': row_id, 'target': target})
        sub.to_csv(sub_fn, index=False)
        print(f'Done wrote to {sub_fn}')

