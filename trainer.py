import torch
import numpy as np
import tqdm
import csv
import os
import cv2
import pandas as pd
from timm.models import create_model
import torch.autograd.profiler as profiler
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image

class BengaliDataset(Dataset):
  def __init__(self, label_csv, train_folder, transforms, cache=True):
    self.label_csv = label_csv
    self.train_folder = train_folder
    self.label = pd.read_csv(self.label_csv)
    #self.label = label.drop(['grapheme'], axis=1, inplace=False)
    self.label[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = self.label[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
    #mod = pd.read_csv('./bengaliai-cv19/train_multi_diacritics.csv')

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
    root = self.label.loc[idx]['grapheme_root']
    consonant = self.label.loc[idx]['consonant_diacritic']
    vowel = self.label.loc[idx]['vowel_diacritic']
    return transforms.ToTensor()(img), root, consonant, vowel

  def __len__(self):
    return self.label.shape[0]

# Borrow from Improved Regularization of Convolutional Neural Networks with Cutout (https://github.com/uoguelph-mlrg/Cutout)
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Trainer:
    def __init__(self, epoch, 
               dataset_path='./drive/MyDrive/datasets/car classification/train_data', 
               val_path='./drive/MyDrive/datasets/car classification/val_data', 
               val_crop='five', batch_size=128, model_name='tf_efficientnet_b3_ns', 
               lr=0.001, lr_min=0.0001, weight_decay=1e-4, momentum=0.9, log_step=25, save_step=10,
               log_path='./drive/My Drive/cars_log.txt', cutout=False, style_aug=False,
               resume=False, resume_path='./drive/My Drive/ckpt/', train_csv='./train_labels.csv', val_csv='./val_labels.csv'):

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
        self.momentum = momentum
        self.log_step = log_step
        self.save_step = save_step
        self.log_path = log_path
        self.cutout = cutout
        self.style_aug = style_aug
        self.resume = resume
        self.resume_path = resume_path
        self.train_csv = train_csv
        self.val_csv = val_csv
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
        val_transform = []

        #transform += [transforms.ToPILImage()]
        transform += [transforms.Resize(self.input_size)]
        #transform += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform)
        self.val_transform = transforms.Compose(val_transform)

        self.dataset = BengaliDataset(self.train_csv, self.dataset_path, self.transform, cache=True)
        self.val_dataset = BengaliDataset(self.val_csv, self.dataset_path, self.transform, cache=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_root = create_model(model_name, pretrained=True, num_classes=168).to(self.device)
        self.model_consonant = create_model(model_name, pretrained=True, num_classes=11).to(self.device)
        self.model_vowel = create_model(model_name, pretrained=True, num_classes=18).to(self.device)
        self.optimizer_root = torch.optim.SGD(self.model_root.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
        self.optimizer_consonant = torch.optim.SGD(self.model_consonant.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
        self.optimizer_vowel = torch.optim.SGD(self.model_vowel.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
    
        self.start_epoch = 0

        if resume:
            ckpt = torch.load(self.resume_path)
            self.model_root.load_state_dict(ckpt['model_root_state_dict'])
            self.model_consonant.load_state_dict(ckpt['model_consonant_state_dict'])
            self.model_vowel.load_state_dict(ckpt['model_vowel_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_root_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_consonant_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_vowel_state_dict'])
            self.start_epoch = ckpt['epoch']

        self.scheduler_root = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_root, T_max=epoch, last_epoch=self.start_epoch-1,
                  eta_min=lr_min)
        self.scheduler_consonant = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_consonant, T_max=epoch, last_epoch=self.start_epoch-1,
                  eta_min=lr_min)
        self.scheduler_vowel = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_vowel, T_max=epoch, last_epoch=self.start_epoch-1,
                  eta_min=lr_min)

    def train(self):
        for epoch in range(self.start_epoch, self.epoch):
            pbar = tqdm.tqdm(self.dataloader)
            pbar.set_description('training process')
            root_epoch_loss_mean = 0 
            consonant_epoch_loss_mean = 0 
            vowel_epoch_loss_mean = 0 
            root_epoch_acc_mean = 0
            consonant_epoch_acc_mean = 0
            vowel_epoch_acc_mean = 0
            root_loss_mean = 0
            consonant_loss_mean = 0
            vowel_loss_mean = 0
            root_acc_mean = 0
            consonant_acc_mean = 0
            vowel_acc_mean = 0
            self.model_root.train()
            self.model_consonant.train()
            self.model_vowel.train()
            self.scheduler_root.step()
            self.scheduler_consonant.step()
            self.scheduler_vowel.step()
            batch_number = len(pbar)
            for it, data in enumerate(pbar):

                inputs = data[0].to(self.device)
                inputs = inputs.repeat(1, 3, 1, 1)
                roots = data[1].to(self.device).long()
                consonants = data[2].to(self.device).long()
                vowels = data[3].to(self.device).long()

                root_preds = self.model_root(inputs)
                root_loss = self.criterion(root_preds, roots)
                root_loss.backward()
                self.optimizer_root.step()
                self.model_root.zero_grad()

                consonant_preds = self.model_consonant(inputs)
                consonant_loss = self.criterion(consonant_preds, consonants)
                consonant_loss.backward()
                self.optimizer_consonant.step()
                self.model_consonant.zero_grad()

                vowel_preds = self.model_vowel(inputs)
                
                
                vowel_loss = self.criterion(vowel_preds, vowels)

                vowel_loss.backward()
                
                
                self.optimizer_vowel.step()
                
                
                self.model_vowel.zero_grad()
                
                
                

                root_loss_mean += root_loss.item()
                consonant_loss_mean += consonant_loss.item() 
                vowel_loss_mean += vowel_loss.item()
                root_epoch_loss_mean += root_loss.item() 
                consonant_epoch_loss_mean += consonant_loss.item() 
                vowel_epoch_loss_mean += vowel_loss.item()
                root_acc = (root_preds.argmax(-1) == roots).sum().item() / roots.size()[0]
                consonant_acc = (consonant_preds.argmax(-1) == consonants).sum().item() / consonants.size()[0]
                vowel_acc = (vowel_preds.argmax(-1) == vowels).sum().item() / vowels.size()[0]
                root_acc_mean += root_acc 
                consonant_acc_mean += consonant_acc 
                vowel_acc_mean += vowel_acc
                root_epoch_acc_mean += root_acc 
                consonant_epoch_acc_mean += consonant_acc 
                vowel_epoch_acc_mean += vowel_acc

                if (it+1) % self.log_step == 0:
                    root_loss_mean /= self.log_step
                    consonant_loss_mean /= self.log_step
                    vowel_loss_mean /= self.log_step
                    root_acc_mean /= self.log_step
                    consonant_acc_mean /= self.log_step
                    vowel_acc_mean /= self.log_step
                    with open(self.log_path, 'a+') as f:
                        f.write('epoch: ' + str(epoch) + '\n')
                        f.write('root loss: ' + str(root_loss_mean) + '\n')
                        f.write('consonant loss: ' + str(consonant_loss_mean) + '\n')
                        f.write('vowel loss: ' + str(vowel_loss_mean) + '\n')
                        f.write('root acc: ' + str(root_acc_mean) + '\n')
                        f.write('cosonant acc: ' + str(consonant_acc_mean) + '\n')
                        f.write('vowel acc: ' + str(vowel_acc_mean) + '\n')
                        f.write('\n')
                    root_loss_mean = 0
                    consonant_loss_mean = 0
                    vowel_loss_mean = 0
                    root_acc_mean = 0
                    consonant_acc_mean = 0
                    vowel_acc_mean = 0
            root_epoch_loss_mean /= len(pbar)
            root_epoch_acc_mean /= len(pbar)
            consonant_epoch_loss_mean /= len(pbar)
            consonant_epoch_acc_mean /= len(pbar)
            vowel_epoch_loss_mean /= len(pbar)
            vowel_epoch_acc_mean /= len(pbar)
            # validate
            pbar = tqdm.tqdm(self.val_dataloader)
            pbar.set_description('validating process')
            root_val_loss_mean = 0
            consonant_val_loss_mean = 0
            vowel_val_loss_mean = 0
            root_val_acc_mean = 0
            consonant_val_acc_mean = 0
            vowel_val_acc_mean = 0
            self.model_root.eval()
            self.model_consonant.eval()
            self.model_vowel.eval()
            with torch.no_grad():
                for it, data in enumerate(pbar):
                    inputs = data[0].to(self.device)
                    inputs = inputs.repeat(1, 3, 1, 1)
                    roots = data[1].to(self.device).long()
                    consonants = data[2].to(self.device).long()
                    vowels = data[3].to(self.device).long()

                    root_preds = self.model_root(inputs)
                    consonant_preds = self.model_consonant(inputs)
                    vowel_preds = self.model_vowel(inputs)

                    root_loss = self.criterion(root_preds, roots)
                    consonant_loss = self.criterion(consonant_preds, consonants)
                    vowel_loss = self.criterion(vowel_preds, vowels)
                    root_val_loss_mean += root_loss.item()
                    consonant_val_loss_mean += consonant_loss.item()
                    vowel_val_loss_mean += vowel_loss.item()
                    root_acc = (root_preds.argmax(-1) == roots).sum().item() / roots.size()[0]
                    consonant_acc = (consonant_preds.argmax(-1) == consonants).sum().item() / consonants.size()[0]
                    vowel_acc = (vowel_preds.argmax(-1) == vowels).sum().item() / vowels.size()[0]
                    root_val_acc_mean += root_acc
                    consonant_val_acc_mean += consonant_acc
                    vowel_val_acc_mean += vowel_acc
                
            root_val_loss_mean /= len(pbar)
            root_val_acc_mean /= len(pbar)

            consonant_val_loss_mean /= len(pbar)
            consonant_val_acc_mean /= len(pbar)

            vowel_val_loss_mean /= len(pbar)
            vowel_val_acc_mean /= len(pbar)


            print('root_loss_mean:', root_epoch_loss_mean, 'root_acc_mean:', root_epoch_acc_mean)
            print('root_val_loss_mean:', root_val_loss_mean, 'root_val_acc_mean:', root_val_acc_mean)

            print('consonant_loss_mean:', consonant_epoch_loss_mean, 'consonant_acc_mean:', consonant_epoch_acc_mean)
            print('consonant_val_loss_mean:', consonant_val_loss_mean, 'consonant_val_acc_mean:', consonant_val_acc_mean)

            print('vowel_loss_mean:', vowel_epoch_loss_mean, 'vowel_acc_mean:', vowel_epoch_acc_mean)
            print('vowel_val_loss_mean:', vowel_val_loss_mean, 'vowel_val_acc_mean:', vowel_val_acc_mean)
            
            with open(self.log_path, 'a+') as f:
                f.write('epoch summary\n')
                f.write('epoch: ' + str(epoch) + '\n')
                f.write('root loss: ' + str(root_epoch_loss_mean) + '\n')
                f.write('root acc: ' + str(root_epoch_acc_mean) + '\n')
                f.write('root_val_loss: ' + str(root_val_loss_mean) + '\n')
                f.write('root_val_acc: ' + str(root_val_acc_mean) + '\n')

                f.write('consonant_root loss: ' + str(consonant_epoch_loss_mean) + '\n')
                f.write('consonant_root acc: ' + str(consonant_epoch_acc_mean) + '\n')
                f.write('consonant_val_loss: ' + str(consonant_val_loss_mean) + '\n')
                f.write('consonant_val_acc: ' + str(consonant_val_acc_mean) + '\n')

                f.write('vowel_root loss: ' + str(vowel_epoch_loss_mean) + '\n')
                f.write('vowel_root acc: ' + str(vowel_epoch_acc_mean) + '\n')
                f.write('vowel_val_loss: ' + str(vowel_val_loss_mean) + '\n')
                f.write('vowel_val_acc: ' + str(vowel_val_acc_mean) + '\n')
                f.write('\n')
            if (epoch+1) % self.save_step == 0:
                torch.save({
                    'model_root_state_dict': self.model_root.state_dict(),
                    'model_consonant_state_dict': self.model_consonant.state_dict(),
                    'model_vowel_state_dict': self.model_vowel.state_dict(),
                    'optimizer_root_state_dict': self.optimizer_root.state_dict(),
                    'optimizer_consonant_state_dict': self.optimizer_consonant.state_dict(),
                    'optimizer_vowel_state_dict': self.optimizer_vowel.state_dict(),
                    'epoch': epoch + 1
                }, './drive/MyDrive/ckpt/grapheme/%d.pth'%(epoch+1))
        

    def criterion(self, preds, trues):
        return torch.nn.CrossEntropyLoss()(preds, trues)
