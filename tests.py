"""
Created on Fri May  5 16:43:11 2023

@author: leonidas

Test only, no training or validation! make sure to adjust model and CKPT_PATH variable
"""

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import transforms,autoaugment
import pytorch_lightning as pl
from torchvision.transforms.functional import InterpolationMode
import os
from utils import get_model
import numpy as np
from torchdata import UnderSampledOwnSplit
import models
# from discord_bot import send_message,User

#insert paths to .ckpt files
CKPT_PATHS = [r"",r""]


RESIZE_SIZE = (224,224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LOG_CONFIG = {
    'confusion_matrix':False,
    'roc_curve':False,
    'auroc':False,
    'classification_report':False,
    'pytorch_cam':True,
    'captum_alg':True,
    'topk':False,
    'bottomk':False,
    'randomk':True
}


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


#foldername for tests
DIR_NAME = "/evaluation_r34dsd_randomcam50"

#mitigate compatibility issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#main
def train():
    
    for CKPT_PATH in CKPT_PATHS:
        
        # SAVE_PATH = os.path.dirname(os.path.dirname(CKPT_PATH))

        #increases performance
        torch.set_float32_matmul_precision('medium')
        #runs a short benchmark and selects algorithm with the best performance to compute convolution
        # torch.backends.cudnn.benchmark = True
                    
        #automatische model erkennung Für SINGLE models
        model = get_model(checkpoint_path=CKPT_PATH)
        
        #kd models
        #falsch model = models.Resnet_34_Dropout.load_from_checkpoint(CKPT_PATH)

        # # KD train, please comment out for normal train
        # student_model = models.Resnet_34_Dropout()
    
        # teacher_model = models.Resnet_34_Dropout()
        
        # # ##KD test
        # modelkd = models.ResponseBasedOnline.load_from_checkpoint(CKPT_PATH,teacher_model=teacher_model,student_model=student_model)
        
        # model = modelkd.student_model#.model
        # # print(modelkd.student_model.model)
        
        bird_data = UnderSampledOwnSplit(train_transform = train_transform_pipeline, 
                                    valid_transform = valid_transform_pipeline, 
                                    data_split_ratios = model.data_split_ratios,
                                    batch_size = model.batch_size,
                                    num_workers = 1#model.num_workers
                                    )
    
        
        #give label names from dataset to model
        model.label_names = np.array(bird_data.test_classes)
        model.label_names_dict = bird_data.test_class_to_idx
    
        model.log_config = LOG_CONFIG
        #make sure to adjust model and CKPT_PATH variable
        # model = models.OnlineKDImageClassifierBase(student_model=student_model, teacher_model=teacher_model).load_from_checkpoint(checkpoint_path=CKPT_PATH)
    
        # lr_monitor = LearningRateMonitor(logging_interval='step',log_momentum=False)
        #early_stopping = EarlyStopping(monitor='validation_loss',mode='min',patience=50) #cosine better
        # save_config = SaveConfigCallback(parser=LightningArgumentParser,config=None)
        # model_checkpoint = ModelCheckpoint(
        #                                filename= f"{model.name}_{NOTE}_" + "{epoch}_{validation_loss:.4f}_{validation_accuracy:.2f}_{validation_mcc:.2f}",
        #                                save_top_k=1,
        #                                monitor="validation_loss",
        #                                mode='min')
        # trainer = pl.Trainer(max_epochs=EPOCHS,callbacks=[model_checkpoint,lr_monitor],precision='bf16-mixed',default_root_dir=log_folder)
        # trainer.fit(model=model,datamodule=bird_data,ckpt_path=CKPT_PATH)
        #switching off inference for test
        # trainer.inference_mode = False
        # trainer.test_loop.inference_mode = False
        
        #switch off tqdm progressbar for validation- and test-step
        # bar = utils.LitProgressBar()
    
        trainer = pl.Trainer(inference_mode=False,default_root_dir= os.getcwd() + DIR_NAME) #only if test_step is executed from a checkpoint file
        trainer.test(model=model,datamodule=bird_data)
        
    #sends completion message if training is finished
    # send_message(message="tests finished!", user=User.Leo)


#autorun main
if __name__ == '__main__':
    train()