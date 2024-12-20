# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:43:11 2023

@author: leonidas

starts training for crossvaidation
"""

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import models
from torchvision.transforms import transforms,autoaugment
# from torchvision.transforms.v2 import AugMix
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor#,EarlyStopping
from torchvision.transforms.functional import InterpolationMode
from utils import LitProgressBar
import re
import statistics
from torchdata import UnderSampledKFoldDataset

##Hyperparameters

CKPT_PATH = None

EPOCHS = 6 #minimum 6
K = 10
NUM_WORKERS = 14
BATCH_SIZE = 256 #2^x, max. 256

log_folder = "test_logs"
# log_folder = "CV_" + "resnet18" + f"_{EPOCHS}E_{K}F"


DROPOUT = 0.3
FULL_DROPOUT = 0.6

#KD params 
ALPHA = 0.25 #between 0 and 1 intervall
TEMPERATURE = 3.5 #between 1 and 5 

OPTIMIZER_ALGORITHM = 'sgd'
LEARNING_RATE = 0.5
LR_SCHEDULER = 'cosineannealinglr'
LR_WARMUP_EPOCHS = 5
LR_WARMUP_METHOD = 'linear'
LR_WARMUP_DECAY = 0.01
WEIGHT_DECAY = 0.00002
NORM_WEIGHT_DECAY = 0.0
LABEL_SMOOTHING = 0.1
MOMENTUM = 0.9

RESIZE_SIZE = (224,224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


#transformation pipelines
train_transform_pipeline = transforms.Compose([
    autoaugment.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
    # AugMix(severity=4,mixture_width=4,alpha=0.65),
    transforms.CenterCrop(RESIZE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=60),
    transforms.Resize(RESIZE_SIZE),
    transforms.ToTensor(), #scales the image’s pixel intensity values in the range [0., 1.]
    transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD), #normalize dataset by imagenet mean
    transforms.RandomErasing()
    ])

valid_transform_pipeline = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.ToTensor(), #scales the image’s pixel intensity values in the range [0., 1.]
    transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD) #normalize dataset by imagenet mean
    ])

bird_data = UnderSampledKFoldDataset(k=K,
                                            train_transform=train_transform_pipeline,
                                            valid_transform=valid_transform_pipeline,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS
                                            )


#main
def train():
    
    model_losses, model_accs,model_tr_losses, model_tr_accs = [],[],[],[]
    for current_fold, kfold_iteration_datamodule in enumerate(bird_data,1):
        print('-' * 32,f"Starting fold {current_fold} of {K}",'-' * 32)
        
        #increases performance
        torch.set_float32_matmul_precision('medium')
        
        #specify model from models.py
        model = models.Resnet_18_Dropout(
                                        lr=LEARNING_RATE,
                                        batch_size=BATCH_SIZE,
                                        epochs=EPOCHS,
                                        momentum=MOMENTUM,
                                        weight_decay=WEIGHT_DECAY,
                                        norm_weight_decay=NORM_WEIGHT_DECAY,
                                        label_smoothing=LABEL_SMOOTHING,
                                        lr_scheduler=LR_SCHEDULER,
                                        lr_warmup_epochs=LR_WARMUP_EPOCHS,
                                        lr_warmup_method=LR_WARMUP_METHOD,
                                        lr_warmup_decay=LR_WARMUP_DECAY,
                                        optimizer_algorithm = OPTIMIZER_ALGORITHM,
                                        num_workers = NUM_WORKERS,
                                        resize_size = RESIZE_SIZE,
                                        dropout=DROPOUT,
                                        # full_dropout=FULL_DROPOUT,
                                                         )
        
        #switch off tqdm progressbar for validation- and test-step
        bar = LitProgressBar()
        lr_monitor = LearningRateMonitor(logging_interval='step',log_momentum=False)

        model_checkpoint = ModelCheckpoint(
                                       filename= f"{model.name}_{log_folder}_" + "{epoch}_{TA:.4f}_{VA:.4f}_{TL:.4f}_{VL:.4f}",
                                       save_top_k=1,
                                       monitor="VL",
                                       mode='min')

        
        trainer = pl.Trainer(max_epochs=EPOCHS,callbacks=[model_checkpoint,lr_monitor,bar],precision='bf16-mixed',default_root_dir=log_folder)
        
        trainer.fit(model=model,datamodule=kfold_iteration_datamodule,ckpt_path=CKPT_PATH)
        
        #log path to .ckpt file
        model.logger.experiment.add_text('best_checkpoint_path',model_checkpoint.best_model_path)
        
        #print best_checkpoint_path to console
        print('best_checkpoint_path is',model_checkpoint.best_model_path)

        
        #log transformation pipeline
        model.logger.experiment.add_text('train_transform_pipeline',str(train_transform_pipeline))
        model.logger.experiment.add_text('valid_transform_pipeline',str(valid_transform_pipeline))
        
        #log checkpoint path
        model.logger.experiment.add_text('CKPT_PATH',str(CKPT_PATH))
        
        #log model metrics per fold
        model_losses.append(round(float(model_checkpoint.best_model_score),4)) #VL
        
        model_accs.append(float(re.search(r'VA=([0-9.]+)', model_checkpoint.best_model_path).group(1)))
        
        model_tr_losses.append(float(re.search(r'TL=([0-9.]+)', model_checkpoint.best_model_path).group(1)))
        
        model_tr_accs.append(float(re.search(r'TA=([0-9.]+)', model_checkpoint.best_model_path).group(1)))

    


    #log lists
    model.logger.experiment.add_text('CV_model_losses',str(model_losses))
    model.logger.experiment.add_text('CV_model_accs',str(model_accs))
    model.logger.experiment.add_text('CV_model_tr_losses',str(model_tr_losses))
    model.logger.experiment.add_text('CV_model_tr_accs',str(model_tr_accs))

    
    #compute average and standard deviation across models/folds losses and validation accuracies
    #validation
    avg_model_loss = round(statistics.fmean(model_losses),4)
    avg_model_acc = round(statistics.fmean(model_accs),4)
    stdev_model_loss = round(statistics.stdev(model_losses),4)
    stdev_model_acc = round(statistics.stdev(model_accs),4)
    
    #training
    avg_model_tr_loss = round(statistics.fmean(model_tr_losses),4)
    avg_model_tr_acc = round(statistics.fmean(model_tr_accs),4)
    stdev_model_tr_loss = round(statistics.stdev(model_tr_losses),4)
    stdev_model_tr_acc = round(statistics.stdev(model_tr_accs),4)


    #log average and standard deviation across models/folds losses and validation accuracies
    model.logger.experiment.add_text('CV_average_validation_loss_with_stdev',str(avg_model_loss) + " +- " + str(stdev_model_loss))
    model.logger.experiment.add_text('CV_average_validation_accuracy_with_stdev',str(avg_model_acc) + " +- " + str(stdev_model_acc))
    
    model.logger.experiment.add_text('CV_average_training_loss_with_stdev',str(avg_model_tr_loss) + " +- " + str(stdev_model_tr_loss))
    model.logger.experiment.add_text('CV_average_training_accuracy_with_stdev',str(avg_model_tr_acc) + " +- " + str(stdev_model_tr_acc))
    
#autorun main
if __name__ == '__main__':
    train()