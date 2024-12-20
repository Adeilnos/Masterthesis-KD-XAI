# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:34:09 2023

@author: leonidas
source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

loads and prepares the dataset
"""
import torchvision
torchvision.disable_beta_transforms_warning()
# from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from abc import ABC,abstractmethod
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
from collections import Counter
import random
import numpy as np

from collections import defaultdict
from random import sample
import torch


class Subset(Dataset):
    """Subset of a dataset at specified indices

    Args:
        dataset (Dataset): The whole dataset
        indices (Sequence): Indices in the whole set selected for subset
        transform: transformations
    """

    def __init__(self,dataset,indices,transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        imgs, labels = self.dataset.__getitem__(self.indices[idx])
        if self.transform:
            return self.transform(imgs),labels
        return imgs,labels
    
    def __len__(self):
        return len(self.indices)


class BaseDataset(ABC,pl.LightningDataModule):
    def __init__(self,train_transform,valid_transform,batch_size=32,num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        
    
    @abstractmethod
    def train_data(self):
        pass
    @abstractmethod
    def valid_data(self):
        pass
    @abstractmethod
    def test_data(self):
        pass
    @abstractmethod
    def predict_data(self):
        pass
    
    def setup(self,stage:str):
        if stage == 'fit':
            # print(f"Calling stage {stage} (fit)")
            self.training_data = self.train_data()
            self.validation_data = self.valid_data()
        if stage == 'test':
            # print(f"Calling stage {stage} (test)")
            self.testing_data = self.test_data()
        if stage == 'predict':
            # print(f"Calling stage {stage} (predict)")
            self.predicting_data = self.predict_data()
    
    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,persistent_workers=True,pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,persistent_workers=True,pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.testing_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,persistent_workers=True,pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predicting_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,persistent_workers=True,pin_memory=True)

#create undersampled dataset
class UnderSampledOwnSplit(BaseDataset):
    def __init__(self,train_transform,valid_transform,data_split_ratios=[0.8,0.15,0.05],random_seed=123456789,relative_reduction = 1, batch_size=32,num_workers=4):
        super().__init__(train_transform=train_transform,valid_transform=valid_transform,batch_size=batch_size,num_workers=num_workers)
        
        assert sum(data_split_ratios) == 1, "sum of data_split_ratios should sum to 1"
        
        self.random_seed = random_seed #For recreation purposes
        
        #artificially reduce dataset
        self.relative_reduction = relative_reduction

        
        random.seed(self.random_seed)
        
        #read whole dataset
        total_dataset = ImageFolder(root='train_valid_test',transform=None)
        
        
        #list of classid/labels in right order
        total_targets = total_dataset.targets
        
        #dictionary contains sum of targets per classid
        sum_per_target = Counter(total_targets)
        
        #minimum sum of targets per class in dataset
        min_samples_per_class = round(min(sum_per_target.values()) * self.relative_reduction)

        
        #sorted per class
        sorted_sum_per_target = dict(sorted(sum_per_target.items(), key=lambda item: item[0]))
        
        #contains indices for split
        train_indices, valid_indices, test_indices = [],[],[]
        
        #get correct indices (fill in the above created variables)
        for classid, targetsum in sorted_sum_per_target.items():
            
            #intervall of indices in whole dataset for one class
            indices_for_class = [i for i, target in enumerate(total_targets) if target == classid]
            
            # print("indices_for_class",sorted(indices_for_class),len(indices_for_class))

            
            #randomly select min_samples_per_class indices, balance dataset by undersampling
            rand_indices = random.sample(indices_for_class, min_samples_per_class) #min_samples_per_class
            

            
            # Calculate the sizes of three chunks
            chunk_sizes = np.multiply(data_split_ratios, min_samples_per_class).astype(int) #min_samples_per_class
            
            
            # Split rand_indices into three chunks with indices for each set
            train_class_indices, valid_class_indices, test_class_indices = np.split(rand_indices, np.cumsum(chunk_sizes)[:-1])
            

            
            #concat indices per set and per class for complete dataset
            train_indices.extend(train_class_indices)
            valid_indices.extend(valid_class_indices)
            test_indices.extend(test_class_indices)
            
        #create splitted and balanced datasets with correct indices
        self.data_train = Subset(total_dataset,train_indices,transform=self.train_transform)
        # self.data_train.dataset.transform = self.train_transform

        self.data_validation = Subset(total_dataset,valid_indices,transform=self.valid_transform)
        # self.data_validation.dataset.transform = self.valid_transform
        
        self.data_test = Subset(total_dataset,test_indices,transform=self.valid_transform)
        
        
        #generate to give over train.py for on test epoch end........
        self.test_classes = self.data_test.dataset.classes
        self.test_class_to_idx = self.data_test.dataset.class_to_idx


        
    def train_data(self):
        return self.data_train
    def valid_data(self):
        return self.data_validation
    def test_data(self):
        return self.data_test
    def predict_data(self):
        return self.data_test

#kfoldcrossvalidation dataset
class UnderSampledKFoldDataset(BaseDataset):
    def __init__(self,
                 k,
                 train_transform,
                 valid_transform,
                 random_seed=123456789,
                 batch_size=32,
                 num_workers=2,
                 ):
        super().__init__(train_transform=train_transform,
                         valid_transform=valid_transform,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         )
        #Amount of folds
        self.k = k
        
        #For recreation purposes
        self.random_seed = random_seed
        random.seed(self.random_seed)
        
        #read whole dataset
        total_dataset = ImageFolder(root='crossv',transform=None)
        
        ##begin undersampling
        #list of classid/labels in right order
        total_targets = total_dataset.targets
        
        #dictionary contains sum of targets per classid
        sum_per_target = Counter(total_targets)
        
        #minimum sum of targets per class in dataset
        min_samples_per_class = min(sum_per_target.values())
        
        #sorted per class
        sorted_sum_per_target = dict(sorted(sum_per_target.items(), key=lambda item: item[0]))
        
        #contains indices for split
        indices = []
        
        #get correct indices (fill in the above created variables)
        for classid, targetsum in sorted_sum_per_target.items():
            
            #intervall of indices in whole dataset for one class
            indices_for_class = [i for i, target in enumerate(total_targets) if target == classid]
            
            
            #randomly select min_samples_per_class indices, balance dataset by undersampling
            rand_indices = random.sample(indices_for_class, min_samples_per_class) #min_samples_per_class
            

            #concat indices per class from complete dataset
            indices.extend(rand_indices)
            
        #create balanced dataset
        self.undersampled_data = Subset(total_dataset,indices)
        
        
        ##begin KFoldDataset
        #create K Folds
        kfold = KFold(n_splits=k,
                      shuffle=True,
                      random_state=random_seed)
        self.all_splits = kfold.split(self.undersampled_data)
        
    def __iter__(self):
        return self

    def __next__(self):
        try:
            #get train and validation indices in next fold
            train_indices,validation_indices = next(self.all_splits)
            
            #create subset with train indices and set train transform
            self.data_train = Subset(dataset=self.undersampled_data,indices=train_indices,transform=self.train_transform)

            #create subset with validation indices and set validation transform
            self.data_validation = Subset(dataset = self.undersampled_data,indices = validation_indices,transform = self.valid_transform)
            return self
        except StopIteration:
            raise StopIteration("All K-Fold splits have been processed")

    def train_data(self):
        return self.data_train
    def valid_data(self):
        return self.data_validation
    def test_data(self):
        #no testphase with crossvalidation
        pass
    def predict_data(self):
        #no testphase with crossvalidation
        pass