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
from model import MyModel, MultiHeadModel, MultiHeadSimpleModel
from torch.autograd import Variable


class BengaliDataset(Dataset):
  def __init__(self, label_csv, unique_csv, train_folder, transforms, cache=True):
    self.label_csv = label_csv
    self.unique_csv = unique_csv
    self.train_folder = train_folder
    self.label = pd.read_csv(self.label_csv)
    unique_df = pd.read_csv(self.unique_csv)
    self.label[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = self.label[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
    self.uniques = unique_df.grapheme.unique()
    self.transforms = transforms
    self.img = [None] * self.label.shape[0]

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
    img = self.load_image(idx)
    img = transforms.RandomRotation(15)(img)
    root = self.label.loc[idx]['grapheme_root']
    consonant = self.label.loc[idx]['consonant_diacritic']
    vowel = self.label.loc[idx]['vowel_diacritic']
    timer = time.time()
    unique = np.where(self.uniques == self.label.grapheme[idx])[0][0]
    #print('unique: ', time.time() - timer)
    return transforms.ToTensor()(img), root, consonant, vowel, unique

  def __len__(self):
    return self.label.shape[0]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        timer = time.time()
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1,input.size(2))
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class MultiHeadTrainer:
    def __init__(self, epoch, 
               dataset_path='./drive/MyDrive/datasets/car classification/train_data', 
               val_path='./drive/MyDrive/datasets/car classification/val_data', 
               val_crop='five', batch_size=128, model_name='tf_efficientnet_b3_ns', 
               lr=0.001, lr_min=0.0001, weight_decay=1e-4, momentum=0.9, scheduler='plateau', dropout=0.2, log_step=25, save_step=10,
               log_path='./drive/My Drive/cars_log.txt', cutout=False, style_aug=False,
               resume=False, resume_path='./drive/My Drive/ckpt/', train_csv='./train_labels.csv', 
               val_csv='./val_labels.csv', unique_csv='./train_labels.csv', save_dir='../drive/MyDrive/ckpt/grapheme/'):

        # initialize attributes
        self.epoch = epoch
        self.dataset_path = dataset_path
        self.val_path = val_path
        self.val_crop = val_crop
        self.batch_size = batch_size
        self.model_name = model_name
        self.lr = lr
        self.lr_mi = lr_min
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.momentum = momentum
        self.scheduler = scheduler
        self.log_step = log_step
        self.save_step = save_step
        self.log_path = log_path
        self.cutout = cutout
        self.style_aug = style_aug
        self.resume = resume
        self.resume_path = resume_path
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.unique_csv = unique_csv
        self.save_dir = save_dir
        if model_name == 'tf_efficientnet_b0_ns':
            self.input_size = (224, 224)
        if model_name == 'tf_efficientnet_b2_ns':
            self.input_size = (260, 260)
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
        self.val_transform = transforms.Compose(val_transform)

        self.dataset = BengaliDataset(self.train_csv, self.unique_csv, self.dataset_path, self.transform, cache=True)
        self.val_dataset = BengaliDataset(self.val_csv, self.unique_csv, self.dataset_path, self.transform, cache=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_multihead = MultiHeadSimpleModel(self.input_size, self.model_name, pretrained=True, dropout=self.dropout).to('cuda')

        self.optimizer_multihead = torch.optim.SGD(self.model_multihead.parameters(), weight_decay=self.weight_decay, lr=self.lr, momentum=self.momentum, nesterov=True)

        # To use focal loss, uncomment this and change the name of self.criterion function
        #self.criterion = FocalLoss()
    
        self.start_epoch = 0

        if resume:
            ckpt = torch.load(self.resume_path)

            self.model_multihead.load_state_dict(ckpt['model_multihead_state_dict'])
            self.optimizer_multihead.load_state_dict(ckpt['optimizer_multihead_state_dict'])
            self.start_epoch = ckpt['epoch']
        if self.scheduler == 'plateau':
          self.scheduler_multihead = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_multihead, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        else:
          self.scheduler_multihead = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_multihead, T_max=epoch, last_epoch=self.start_epoch-1,
                    eta_min=lr_min)

    def train(self):
        root_last_loss = 1000
        consonant_last_loss = 1000
        vowel_last_loss = 1000
        multihead_last_loss = 1000
        for epoch in range(self.start_epoch, self.epoch):
            pbar = tqdm.tqdm(self.dataloader)
            pbar.set_description('training process')

            root_epoch_loss_mean = 0 
            consonant_epoch_loss_mean = 0 
            vowel_epoch_loss_mean = 0 
            multihead_epoch_loss_mean = 0

            root_epoch_acc_mean = 0
            consonant_epoch_acc_mean = 0
            vowel_epoch_acc_mean = 0
            multihead_root_epoch_acc_mean = 0
            multihead_consonant_epoch_acc_mean = 0
            multihead_vowel_epoch_acc_mean = 0
            multihead_unique_epoch_acc_mean = 0

            multihead_root_epoch_loss_mean = 0
            multihead_consonant_epoch_loss_mean = 0
            multihead_vowel_epoch_loss_mean = 0
            multihead_unique_epoch_loss_mean = 0


            root_loss_mean = 0
            consonant_loss_mean = 0
            vowel_loss_mean = 0
            unique_loss_mean = 0
            multihead_loss_mean = 0

            root_acc_mean = 0
            consonant_acc_mean = 0
            vowel_acc_mean = 0
            unique_acc_mean = 0
            multihead_root_acc_mean = 0
            multihead_consonant_acc_mean = 0
            multihead_vowel_acc_mean = 0
            multihead_unique_acc_mean = 0

            self.model_multihead.train()
            self.model_multihead.zero_grad()
            if self.scheduler == 'plateau':
              self.scheduler_multihead.step(multihead_last_loss)
            else:
              self.scheduler_multihead.step()
            batch_number = len(pbar)
            for it, data in enumerate(pbar):
                
                inputs = data[0].to(self.device)
                inputs = inputs.repeat(1, 3, 1, 1)
                roots = data[1].to(self.device).long()
                consonants = data[2].to(self.device).long()
                vowels = data[3].to(self.device).long()
                uniques = data[4].to(self.device).long()
                
                
                root, consonant, vowel, unique = self.model_multihead(inputs, uniques)
                multihead_root_loss = (1/4)*self.criterion(root, roots)
                multihead_consonant_loss = (1/4)*self.criterion(consonant, consonants)
                multihead_vowel_loss = (1/4)*self.criterion(vowel, vowels)
                multihead_unique_loss = (1/4)*self.criterion(unique, uniques)
                multihead_root_loss.backward(retain_graph=True)
                multihead_consonant_loss.backward(retain_graph=True)
                multihead_vowel_loss.backward(retain_graph=True)
                multihead_unique_loss.backward()
                self.optimizer_multihead.step()
                self.model_multihead.zero_grad()
                

                root_loss_mean += multihead_root_loss.item()
                consonant_loss_mean += multihead_consonant_loss.item()
                vowel_loss_mean += multihead_vowel_loss.item()
                unique_loss_mean += multihead_unique_loss.item()


                multihead_root_epoch_loss_mean += multihead_root_loss.item()
                multihead_consonant_epoch_loss_mean += multihead_consonant_loss.item()
                multihead_vowel_epoch_loss_mean += multihead_vowel_loss.item()
                multihead_unique_epoch_loss_mean += multihead_unique_loss.item()


                multihead_root_acc = (root.argmax(-1) == roots).sum().item() / roots.size()[0]
                multihead_consonant_acc = (consonant.argmax(-1) == consonants).sum().item() / consonants.size()[0]
                multihead_vowel_acc = (vowel.argmax(-1) == vowels).sum().item() / vowels.size()[0]
                multihead_unique_acc = (unique.argmax(-1) == uniques).sum().item() / uniques.size()[0]


                multihead_root_acc_mean += multihead_root_acc
                multihead_consonant_acc_mean += multihead_consonant_acc
                multihead_vowel_acc_mean += multihead_vowel_acc
                multihead_unique_acc_mean += multihead_unique_acc


                multihead_root_epoch_acc_mean += multihead_root_acc
                multihead_consonant_epoch_acc_mean += multihead_consonant_acc
                multihead_vowel_epoch_acc_mean += multihead_vowel_acc
                multihead_unique_epoch_acc_mean += multihead_unique_acc

                
                if (it+1) % self.log_step == 0:
                    root_loss_mean /= self.log_step
                    consonant_loss_mean /= self.log_step
                    vowel_loss_mean /= self.log_step
                    unique_loss_mean /= self.log_step

                    multihead_root_acc_mean /= self.log_step
                    multihead_consonant_acc_mean /= self.log_step
                    multihead_vowel_acc_mean /= self.log_step
                    multihead_unique_acc_mean /= self.log_step
                    with open(self.log_path, 'a+') as f:
                        f.write('epoch: ' + str(epoch) + '\n')
                        f.write('root loss: ' + str(root_loss_mean) + '\n')
                        f.write('consonant loss: ' + str(consonant_loss_mean) + '\n')
                        f.write('vowel loss: ' + str(vowel_loss_mean) + '\n')
                        f.write('unique loss: ' + str(unique_loss_mean) + '\n')
                        f.write('multihead root acc: ' + str(multihead_root_acc_mean) + '\n')
                        f.write('multihead consonant acc: ' + str(multihead_consonant_acc_mean) + '\n')
                        f.write('multihead vowel acc: ' + str(multihead_vowel_acc_mean) + '\n')
                        f.write('multihead unique acc: ' + str(multihead_unique_acc_mean) + '\n')
                        f.write('\n')
                    root_loss_mean = 0
                    consonant_loss_mean = 0
                    vowel_loss_mean = 0
                    unique_loss_mean = 0
                    multihead_loss_mean = 0
                    root_acc_mean = 0
                    consonant_acc_mean = 0
                    vowel_acc_mean = 0
                    multihead_root_acc_mean = 0
                    multihead_consonant_acc_mean = 0
                    multihead_vowel_acc_mean = 0
                    multihead_unique_acc_mean = 0

                    
            root_epoch_loss_mean /= len(pbar)
            root_epoch_acc_mean /= len(pbar)
            consonant_epoch_loss_mean /= len(pbar)
            consonant_epoch_acc_mean /= len(pbar)
            vowel_epoch_loss_mean /= len(pbar)
            vowel_epoch_acc_mean /= len(pbar)
            multihead_epoch_loss_mean /= len(pbar)
            multihead_root_epoch_acc_mean /= len(pbar)
            multihead_consonant_epoch_acc_mean /= len(pbar)
            multihead_vowel_epoch_acc_mean /= len(pbar)
            multihead_unique_epoch_acc_mean /= len(pbar)

            root_last_loss = root_epoch_loss_mean
            consonant_last_loss = consonant_epoch_loss_mean
            vowel_last_loss = vowel_epoch_loss_mean
            multihead_last_loss = multihead_epoch_loss_mean

            # validate
            pbar = tqdm.tqdm(self.val_dataloader)
            pbar.set_description('validating process')
            root_val_loss_mean = 0
            consonant_val_loss_mean = 0
            vowel_val_loss_mean = 0
            unique_val_loss_mean = 0
            root_val_acc_mean = 0
            consonant_val_acc_mean = 0
            vowel_val_acc_mean = 0
            unique_val_acc_mean = 0
            unique_val_max_prob_mean = 0
            
            
            self.model_multihead.eval()
            with torch.no_grad():
                for it, data in enumerate(pbar):
                    inputs = data[0].to(self.device)
                    inputs = inputs.repeat(1, 3, 1, 1)
                    roots = data[1].to(self.device).long()
                    consonants = data[2].to(self.device).long()
                    vowels = data[3].to(self.device).long()
                    uniques = data[4].to(self.device).long()

                    
                    root, consonant, vowel, unique = self.model_multihead(inputs, uniques)
                    multihead_root_loss = (1/4)*self.criterion(root, roots)
                    multihead_consonant_loss = (1/4)*self.criterion(consonant, consonants)
                    multihead_vowel_loss = (1/4)*self.criterion(vowel, vowels)
                    multihead_unique_loss = (1/4)*self.criterion(unique, uniques)
                    self.model_multihead.zero_grad()

                    

                    root_val_loss_mean += multihead_root_loss.item()
                    consonant_val_loss_mean += multihead_consonant_loss.item()
                    vowel_val_loss_mean += multihead_vowel_loss.item()
                    unique_val_loss_mean += multihead_unique_loss.item()
                    
                    unique_val_acc_mean += (unique.argmax(-1) == uniques).sum().item() / uniques.size()[0]
                    unique_prob = F.softmax(unique, dim=1)
                    unique_val_max_prob_mean += unique_prob.max(-1).values.mean(-1).item()

                    root_acc = (root.argmax(-1) == roots).sum().item() / roots.size()[0]
                    consonant_acc = (consonant.argmax(-1) == consonants).sum().item() / consonants.size()[0]
                    vowel_acc = (vowel.argmax(-1) == vowels).sum().item() / vowels.size()[0]
                    root_val_acc_mean += root_acc
                    consonant_val_acc_mean += consonant_acc
                    vowel_val_acc_mean += vowel_acc

                
            root_val_loss_mean /= len(pbar)
            root_val_acc_mean /= len(pbar)

            consonant_val_loss_mean /= len(pbar)
            consonant_val_acc_mean /= len(pbar)

            vowel_val_loss_mean /= len(pbar)
            vowel_val_acc_mean /= len(pbar)


            unique_val_loss_mean /= len(pbar)
            unique_val_acc_mean /= len(pbar)
            unique_val_max_prob_mean /= len(pbar)


            print('root_loss_mean:', root_epoch_loss_mean, 'root_acc_mean:', root_epoch_acc_mean)
            print('root_val_loss_mean:', root_val_loss_mean, 'root_val_acc_mean:', root_val_acc_mean)

            print('consonant_loss_mean:', consonant_epoch_loss_mean, 'consonant_acc_mean:', consonant_epoch_acc_mean)
            print('consonant_val_loss_mean:', consonant_val_loss_mean, 'consonant_val_acc_mean:', consonant_val_acc_mean)

            print('vowel_loss_mean:', vowel_epoch_loss_mean, 'vowel_acc_mean:', vowel_epoch_acc_mean)
            print('vowel_val_loss_mean:', vowel_val_loss_mean, 'vowel_val_acc_mean:', vowel_val_acc_mean)

            print('unique_val_loss_mean:', unique_val_loss_mean)
            print('unique_val_acc_mean:', unique_val_acc_mean)
            print('unique_val_max_prob_mean:', unique_val_max_prob_mean)
            
            with open(self.log_path, 'a+') as f:
                f.write('epoch summary\n')
                f.write('epoch: ' + str(epoch) + '\n')
                f.write('root loss: ' + str(multihead_root_epoch_loss_mean) + '\n')
                f.write('root acc: ' + str(multihead_root_epoch_acc_mean) + '\n')
                f.write('root_val_loss: ' + str(root_val_loss_mean) + '\n')
                f.write('root_val_acc: ' + str(root_val_acc_mean) + '\n')

                f.write('consonant_root loss: ' + str(multihead_consonant_epoch_loss_mean) + '\n')
                f.write('consonant_root acc: ' + str(multihead_consonant_epoch_acc_mean) + '\n')
                f.write('consonant_val_loss: ' + str(consonant_val_loss_mean) + '\n')
                f.write('consonant_val_acc: ' + str(consonant_val_acc_mean) + '\n')

                f.write('vowel_root loss: ' + str(multihead_vowel_epoch_loss_mean) + '\n')
                f.write('vowel_root acc: ' + str(multihead_vowel_epoch_acc_mean) + '\n')
                f.write('vowel_val_loss: ' + str(vowel_val_loss_mean) + '\n')
                f.write('vowel_val_acc: ' + str(vowel_val_acc_mean) + '\n')

                f.write('unique loss: ' + str(multihead_unique_epoch_loss_mean) + '\n')
                f.write('unique acc: ' + str(multihead_unique_epoch_acc_mean) + '\n')
                f.write('unique val loss: ' + str(unique_val_loss_mean) + '\n')
                f.write('unique val acc: ' + str(unique_val_acc_mean) + '\n')
                f.write('\n')
            if (epoch+1) % self.save_step == 0:
                torch.save({
                    'model_multihead_state_dict': self.model_multihead.state_dict(),
                    'optimizer_multihead_state_dict': self.optimizer_multihead.state_dict(),
                    'epoch': epoch + 1
                }, os.path.join(self.save_dir, '%d.pth'%(epoch+1)))
        

    def criterion(self, preds, trues):
        return torch.nn.CrossEntropyLoss()(preds, trues)
