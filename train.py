# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:43:11 2023

@author: leonidas

starts train & test for the model, contains main
"""

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import models
from torchdata import UnderSampledOwnSplit
from torchvision.transforms import transforms,autoaugment
# from torchvision.transforms.v2 import AugMix
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor#,EarlyStopping
from torchvision.transforms.functional import InterpolationMode
from utils import LitProgressBar
# from torchvision.datasets import ImageFolder
import numpy as np
# from discord_bot import send_message,User



##Hyperparameters
NOTE = 'naiveruntime' #keep it short!
test_run = True

CKPT_PATH = None
# TEACHER_CHECKPOINT = r""

EPOCHS = 600 #minimum 6
NUM_WORKERS = 14
BATCH_SIZE = 256 #use multiples of 64 that fits into memory usually set between 64 and 256.
DATA_SPLIT_RATIOS = [0.85,0.1,0.05] #contains the relative amount of  the train-, validation- & test-size.

DROPOUT = 0.3 #0.3
FULL_DROPOUT = 0.1 #0.1

#KD params 
ALPHA = 0.6 #between 0 and 1 intervall 0.1-0.3 for RBO
# ALPHA = 0.8 #between 0 and 1 intervall 0.8-0.95 for FBO
BETA = 0.05 #between 0 and 1 intervall 0.05-0.3
TEMPERATURE = 5 #between 1 and 5

OPTIMIZER_ALGORITHM = 'sgd'

LEARNING_RATE = 0.5 if OPTIMIZER_ALGORITHM == "sgd" else 1e-4
LR_SCHEDULER = 'cosineannealinglr'
LR_WARMUP_EPOCHS = 5
LR_WARMUP_METHOD = 'linear'
LR_WARMUP_DECAY = 0.01
WEIGHT_DECAY = 0.00002 if OPTIMIZER_ALGORITHM == "sgd" else 1e-2
NORM_WEIGHT_DECAY = 0.0
LABEL_SMOOTHING = 0.1 #0.1
MOMENTUM = 0.9


LOG_CONFIG = {
    'confusion_matrix':True,
    'roc_curve':True,
    'auroc':True,
    'classification_report':True,
    'pytorch_cam':True,
    'captum_alg':True,
    'topk':True,
    'bottomk':True,
    'randomk':True
}


RESIZE_SIZE = (224,224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225) #denormalize on_test_epoch_end hardcoded

#only log non test runs into lightning_logs folder
log_folder = "test_logs" if test_run else None #evtl mit strings kd_logs etc unterscheiden



#transformation pipelines
train_transform_pipeline = transforms.Compose([
    autoaugment.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
    # AugMix(severity=4,mixture_width=4,alpha=0.65),
    transforms.CenterCrop(RESIZE_SIZE),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=60),
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



bird_data = UnderSampledOwnSplit(train_transform = train_transform_pipeline, 
                            valid_transform = valid_transform_pipeline, 
                            data_split_ratios = DATA_SPLIT_RATIOS,
                            batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS
                            )



#main
def train():
    #increases performance
    torch.set_float32_matmul_precision('medium')
    #runs a short benchmark and selects algorithm with the best performance to compute convolution
    # torch.backends.cudnn.benchmark = True
    
    #specify model(s) from models.py
    model = models.NaiveClassifier(
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
                                    note = NOTE,
                                    resize_size = RESIZE_SIZE,
                                    data_split_ratios = DATA_SPLIT_RATIOS,
                                    # dropout=DROPOUT,
                                    # full_dropout=FULL_DROPOUT,
                                                     )
    
    # teacher_model = models.Pre_Resnet_101_Dropout(
    #                                 lr=LEARNING_RATE,
    #                                 batch_size=BATCH_SIZE,
    #                                 epochs=EPOCHS,
    #                                 momentum=MOMENTUM,
    #                                 weight_decay=WEIGHT_DECAY,
    #                                 norm_weight_decay=NORM_WEIGHT_DECAY,
    #                                 label_smoothing=LABEL_SMOOTHING,
    #                                 lr_scheduler=LR_SCHEDULER,
    #                                 lr_warmup_epochs=LR_WARMUP_EPOCHS,
    #                                 lr_warmup_method=LR_WARMUP_METHOD,
    #                                 lr_warmup_decay=LR_WARMUP_DECAY,
    #                                 optimizer_algorithm = OPTIMIZER_ALGORITHM,
    #                                 num_workers = NUM_WORKERS,
    #                                 note = NOTE,
    #                                 resize_size = RESIZE_SIZE,
    #                                 data_split_ratios = DATA_SPLIT_RATIOS,
    #                                 dropout=DROPOUT,
    #                                 # full_dropout=FULL_DROPOUT,
    #                                                   )
    
    
    # # teacher_model = models.Resnet_50_Dropout.load_from_checkpoint(checkpoint_path=TEACHER_CHECKPOINT)
    
    
    # # #KD train
    # model = models.FeatureBasedOnline(
    #                                 student_model=student_model,
    #                                 teacher_model=teacher_model,
    #                                 lr=LEARNING_RATE,
    #                                 batch_size=BATCH_SIZE,
    #                                 epochs=EPOCHS,
    #                                 momentum=MOMENTUM,
    #                                 weight_decay=WEIGHT_DECAY,
    #                                 norm_weight_decay=NORM_WEIGHT_DECAY,
    #                                 label_smoothing=LABEL_SMOOTHING,
    #                                 lr_scheduler=LR_SCHEDULER,
    #                                 lr_warmup_epochs=LR_WARMUP_EPOCHS,
    #                                 lr_warmup_method=LR_WARMUP_METHOD,
    #                                 lr_warmup_decay=LR_WARMUP_DECAY,
    #                                 optimizer_algorithm = OPTIMIZER_ALGORITHM,
    #                                 num_workers = NUM_WORKERS,
    #                                 note = NOTE,
    #                                 resize_size = RESIZE_SIZE,
    #                                 data_split_ratios = DATA_SPLIT_RATIOS,
    #                                 alpha=ALPHA,
    #                                 # T=TEMPERATURE
    #                                 beta = BETA,
    #                                               )
        
    #set log config
    model.log_config = LOG_CONFIG
    #switch off tqdm progressbar for validation- and test-step
    bar = LitProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval='step',log_momentum=False)
    #early_stopping = EarlyStopping(monitor='validation_loss',mode='min',patience=50) #cosine better
    model_checkpoint = ModelCheckpoint(
                                   filename= f"{model.name}_{NOTE}_" + "{epoch}_{TA:.3f}_{VA:.4f}_{TAT:.3f}_{VAT:.4f}_{TL:.3f}_{VL:.4f}_{TuCL:.4f}_{TuCLT:.4f}" if model.kd_run else f"{model.name}_{NOTE}_" + "{epoch}_{TA:.4f}_{VA:.4f}_{TL:.4f}_{VL:.4f}",
                                   save_top_k=1,
                                   monitor="VL",
                                   mode='min')
    
    trainer = pl.Trainer(max_epochs=EPOCHS,callbacks=[model_checkpoint,lr_monitor,bar],precision='bf16-mixed',default_root_dir=log_folder,num_sanity_val_steps=0)

    trainer.fit(model=model,datamodule=bird_data,ckpt_path=CKPT_PATH)
        
    #log path to .ckpt file
    model.logger.experiment.add_text('best_checkpoint_path',model_checkpoint.best_model_path)
    
    #print best_checkpoint_path to console
    print('best_checkpoint_path is',model_checkpoint.best_model_path)
    
    #log transformation pipeline
    model.logger.experiment.add_text('train_transform_pipeline',str(train_transform_pipeline))
    model.logger.experiment.add_text('valid_transform_pipeline',str(valid_transform_pipeline))
    
    #log checkpoint path
    model.logger.experiment.add_text('CKPT_PATH',str(CKPT_PATH))
    
    #give label names from dataset to model
    model.label_names = np.array(bird_data.test_classes)
    model.label_names_dict = bird_data.test_class_to_idx

    #switching off inference for test
    trainer.inference_mode = False
    trainer.test_loop.inference_mode = False
    #prevents trainer from logging twice, if test step is executed directly after train step
    model.setup_run = False
    trainer.test(model=model,datamodule=bird_data)
    
    #sends completion message if training is finished
    # send_message(message=model_checkpoint.best_model_path, user=User.Leo)

#autorun main
if __name__ == '__main__':
    train()