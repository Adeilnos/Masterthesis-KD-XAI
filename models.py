#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 19:34:16 2023

@author: leonidas
"""
import numpy
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy,MatthewsCorrCoef,ROC,AUROC,ConfusionMatrix
from lion_pytorch import Lion
from abc import ABC,abstractmethod
from torch.optim import AdamW,SGD
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,ExponentialLR,ReduceLROnPlateau,StepLR,LinearLR,SequentialLR
from knowledge_distillation_utils import knowledge_distillation_loss,feature_based_distillation_loss,offline_feature_based_distillation_loss
import utils
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from sklearn.metrics import classification_report
from pytorch_grad_cam import GradCAM, HiResCAM, AblationCAM, GradCAMPlusPlus, GradCAMElementWise, RandomCAM
from pytorch_grad_cam.metrics.road import ROADCombined,ROADLeastRelevantFirst,ROADMostRelevantFirst
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from image_utils import array_from_image_path,show_multiple_images,create_multiple_images,show_image_from_array
# from skimage.transform import resize
from image_utils import denormalize
from torchvision.transforms import transforms
from captum.attr import IntegratedGradients,GuidedGradCam,Saliency,NoiseTunnel, Occlusion
from functools import partial
from tqdm import tqdm
from functools import partialmethod


NUM_CLASSES = 524
TOP_K = 10
RAND_K_VALUE = 10

##ImageClassifierBase
class ImageClassifierBase(ABC,pl.LightningModule):
    def __init__(self,
                 lr=0.1,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1, #regularization method, works like temperature, affects test_loss!
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm = 'sgd',
                 num_workers = 4,
                 note = '',
                 resize_size = (224,224),
                 kd_run = False,
                 setup_run = True,
                 log_config = None,
                 data_split_ratios = [0.8,0.15,0.05],
                 ):
        super().__init__()
        self.model = self.init_base_model()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.lr_scheduler = lr_scheduler.lower()
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_method = lr_warmup_method.lower()
        self.epochs = epochs
        self.lr_warmup_decay = lr_warmup_decay
        self.norm_weight_decay = norm_weight_decay
        self.optimizer_algorithm = optimizer_algorithm.lower()
        self.num_workers = num_workers
        self.note = note
        self.resize_size = resize_size
        self.kd_run = kd_run
        self.setup_run = setup_run
        self.data_split_ratios = data_split_ratios
        self.name = self.__class__.__name__

        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.accuracy_top_5 = Accuracy(task="multiclass", num_classes=NUM_CLASSES,top_k=5)
        self.mcc = MatthewsCorrCoef(task="multiclass",num_classes=NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.confusion_matrix = ConfusionMatrix(task="multiclass",num_classes=NUM_CLASSES)
        self.roc_curve = ROC(task='multiclass',num_classes=NUM_CLASSES)
        self.auroc = AUROC(task='multiclass',num_classes=NUM_CLASSES)
        self.test_step_prediction = []
        self.test_step_label = []
        self.test_step_input = []
        
        self.label_names = None
        
        self.label_names_dict = None
        
        self.log_config = {
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
        
        #overwrite dict if log_config is given
        if log_config:
            self.log_config.update(log_config)
        
        self.save_hyperparameters({"lr":self.lr,
                                  "batch_size":self.batch_size,
                                  "epochs":self.epochs,
                                  "momentum":self.momentum,
                                  "weight_decay":self.weight_decay,
                                  "norm_weight_decay":self.norm_weight_decay,
                                  "label_smoothing":self.label_smoothing,
                                  "lr_scheduler":self.lr_scheduler,
                                  "lr_warmup_epochs":self.lr_warmup_epochs,
                                  "lr_warmup_method":self.lr_warmup_method,
                                  "lr_warmup_decay":self.lr_warmup_decay,
                                  "optimizer_algorithm":self.optimizer_algorithm,
                                  "num_workers":self.num_workers,
                                  "note":self.note,
                                  "resize_size":self.resize_size,  
                                  "name":self.name,
                                  "num_classes":NUM_CLASSES,
                                  "data_split_ratios":self.data_split_ratios,
                                  })

        
    @abstractmethod
    def init_base_model(self):
        pass

    def forward(self,x):
        out = self.model(x)
        return out
    
    #freeze layers for offline kd
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        
    def configure_optimizers(self):
        parameters = utils.set_weight_decay(
            self,
            self.weight_decay,
            self.norm_weight_decay,
            None
        )
        
        if self.optimizer_algorithm == 'sgd':
            optimizer = SGD(parameters,lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
        
        elif self.optimizer_algorithm == 'adam':
            optimizer = AdamW(parameters,lr=self.lr,weight_decay=self.weight_decay)

        else:
            raise RuntimeError(
                f"Invalid optimizer '{self.optimizer_algorithm}'."
            )
        
        if self.lr_scheduler == 'cosineannealinglr':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs - self.lr_warmup_epochs)
        else:
            raise RuntimeError(
                f"Invalid scheduler '{self.lr_scheduler}'."
            )
        if self.lr_warmup_epochs > 0:
            if self.lr_warmup_method == 'linear':
                warmup_lr_scheduler = LinearLR(optimizer, start_factor=self.lr_warmup_decay, total_iters=self.lr_warmup_epochs)
            else:
                raise RuntimeError(
                f"Invalid warmup method '{self.lr_warmup_method}'."
            )
            lr_scheduler = SequentialLR(optimizer,schedulers=[warmup_lr_scheduler,scheduler],milestones=[self.lr_warmup_epochs])
        else :
            lr_scheduler = scheduler
        return [optimizer],[{"scheduler": lr_scheduler,'interval':'epoch'}]
    
    def setup(self, stage):
        #prevents trainer from logging twice, if test step is executed directly after train step
        if self.setup_run:

            #log architecture graph
            self.logger._log_graph = True
            self.logger.log_graph(self,torch.rand((1,3) + self.resize_size).to('cuda'))
            
            self.logger.experiment.add_text('Model',str(self.model))
            

        
    def training_step(self, batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output,labels)
        #one-hot encoded (because of cutmix & mixup), convert to class label
        if labels.size(dim=-1) == NUM_CLASSES:
            labels = torch.argmax(labels,dim=1)
        accuracy = self.accuracy(output,labels)
        accuracy_top_5 = self.accuracy_top_5(output,labels)
        mcc = self.mcc(output,labels)
        
        self.logger.experiment.add_scalars('Accuracy', 
                                           {'train': accuracy},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Loss', 
                                           {'train': loss},
                                           global_step=self.current_epoch)
        self.logger.experiment.add_scalars('MCC', 
                                           {'train': mcc},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5', 
                                           {'train': accuracy_top_5},
                                           global_step=self.current_epoch)
        
        self.log("TA",accuracy,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)#batch size evtl rauslöschen
        self.log("TL",loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size) #self.log sachen evtl weg
        self.log("TMCC",mcc,on_step=False,on_epoch=True,batch_size=self.batch_size)
        return loss
    
    
    def validation_step(self,batch,batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = self.criterion(output,labels)
        accuracy = self.accuracy(output,labels)
        accuracy_top_5 = self.accuracy_top_5(output,labels)

        mcc = self.mcc(output,labels)
        
        self.logger.experiment.add_scalars('Accuracy',
                                           {'validation': accuracy},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Loss', 
                                           {'validation': loss},
                                           global_step=self.current_epoch)
        self.logger.experiment.add_scalars('MCC', 
                                           {'validation': mcc},
                                           global_step=self.current_epoch)   
        
        self.logger.experiment.add_scalars('Accuracy_top_5', 
                                           {'validation': accuracy_top_5},
                                           global_step=self.current_epoch)

        
        self.log("VL",loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VA",accuracy,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VMCC",mcc,on_step=False,batch_size=self.batch_size)
        return loss
    
    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        
        #switch to true to test only on specific classes. Used to evaluate consistency of xai methods.
        if False:
            
            
            # List of "random" target classes for the XAI evaluation
            target_classes = [0,1,433, 510, 408, 203, 435, 449, 149, 338, 434, 400, 354, 441, 344, 141, 26, 120, 479, 339, 294, 323, 195, 264, 261, 444, 462, 235, 481, 335, 369, 139, 502, 486, 419, 33, 173, 135, 325, 140]
            
            # Create a mask for the target classes
            mask = torch.isin(labels, torch.tensor(target_classes, device=labels.device))
            
            if not mask.any():  # Skip the batch if no samples match the target classes
                return None            
            # Filter inputs and labels for the target class
            filtered_inputs = inputs[mask]
            filtered_labels = labels[mask]
            
            
            # Forward pass and compute metrics
            output = self(filtered_inputs)
            loss = self.criterion(output, filtered_labels)
            accuracy = self.accuracy(output, filtered_labels)
            accuracy_top_5 = self.accuracy_top_5(output, filtered_labels)
            mcc = self.mcc(output, filtered_labels)
            
            # Log metrics
            self.log("test_accuracy", accuracy, prog_bar=True, batch_size=len(filtered_labels))
            self.log("test_accuracy_top_5", accuracy_top_5, prog_bar=True, batch_size=len(filtered_labels))
            self.log("test_loss", loss, prog_bar=True, batch_size=len(filtered_labels))
            self.log("test_mcc", mcc, prog_bar=True, batch_size=len(filtered_labels))
            
            # Append outputs for later processing in on_test_epoch_end
            self.test_step_prediction.append(output)
            self.test_step_label.append(filtered_labels)
            self.test_step_input.append(filtered_inputs)
            
        else:
            # inputs.requires_grad = True
            output = self(inputs)
            loss = self.criterion(output,labels)
            accuracy = self.accuracy(output,labels)
            accuracy_top_5 = self.accuracy_top_5(output,labels)
            mcc = self.mcc(output,labels)
    
            #predictions = torch.argmax(output,dim=1)
            
            self.log("test_accuracy",accuracy,prog_bar=True,batch_size=self.batch_size)
            self.log("test_accuracy_top_5",accuracy_top_5,prog_bar=True,batch_size=self.batch_size)
            self.log("test_loss",loss,prog_bar=True,batch_size=self.batch_size)
            self.log("test_mcc",mcc,prog_bar=True,batch_size=self.batch_size)
            
            self.test_step_prediction.append(output)
            self.test_step_label.append(labels)
            self.test_step_input.append(inputs)
            
        return loss

    
    def on_test_epoch_end(self):
        
        #disable tqdm progressbar in external libraries too
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        
        # torch.set_grad_enabled(True)
        #test data labels and images
        all_labels = torch.cat(self.test_step_label) #torch.Size([36XX])
                
        all_images = torch.cat(self.test_step_input) #torch.Size([2625, 3, 224, 224])
        
        #select interesting images for the XAI methods and add them to a list
        top_k = partial(torch.topk, k=TOP_K)
        bottom_k = partial(torch.topk, k=TOP_K, largest=False)
        rand_k = partial(utils.get_k_random_values,k=RAND_K_VALUE,device="cuda")
        rand_all = partial(utils.get_k_random_values,k=len(all_labels),device="cuda")
        
        selection_functions = []
        selection_functions += [(top_k,"Top")] if self.log_config['topk'] else []
        selection_functions += [(bottom_k,"Bot")] if self.log_config['bottomk'] else []
        # selection_functions += [(rand_all,"All")] if self.log_config['randomk'] else []
        selection_functions += [(rand_k,"Random")] if self.log_config['randomk'] else []
        selection_functions += [(rand_all,"conf_false")] if self.log_config['randomk'] else []
        selection_functions += [(rand_all,"inconf_false")] if self.log_config['randomk'] else []
        selection_functions += [(rand_all,"inconf_true")] if self.log_config['randomk'] else []
        
        #select models for test procedures one triplet consists: of the model, its predicted output of the testimages & a string to differntiate the model for logging purposes
        models = [(self.student_model.model, self.test_step_prediction, "Student_model_"),(self.teacher_model.model, self.test_step_prediction_teacher, "Teacher_model_")] if self.kd_run else [(self.model, self.test_step_prediction, "")]
        
        
        for model_select, model_prediction, model_str in models:
             
            
            #variables
            all_predictions = torch.cat(model_prediction) #torch.Size([3668, 524])
                        
            all_predictions_prop = torch.softmax(all_predictions,dim=1) #torch.Size([2625, 525])
            all_predictions_max_prop = torch.max(all_predictions_prop,dim=1) #max werte der softmax vektoren von 524 -> 1 eben die prediction klasse
                    
            all_predictions_idx = torch.argmax(all_predictions,dim=1) #one dim pred labels
            
            #xai init variables, evtl target_layers dynamisch machen?
            target_layers = [model_select.layer4[-1]] #specify target_layers on which to perform GradCAM class must have model_selects variable and chose right layers! for resnet 18 & 50: [model_select.layer4[-1]]
            # target_layers = [model_select.layer4] #specify target_layers on which to perform GradCAM class must have model_selects variable and chose right layers! for resnet 18 & 50: [model_select.layer4[-1]]
            #pytorch_grad_cam init cams
            pytorch_gradcam_cams = [
            GradCAM(model=model_select,target_layers=target_layers,use_cuda=True),
            HiResCAM(model=model_select,target_layers=target_layers,use_cuda=True),
            AblationCAM(model=model_select,target_layers=target_layers,use_cuda=True),
            GradCAMPlusPlus(model=model_select,target_layers=target_layers,use_cuda=True),
            GradCAMElementWise(model=model_select,target_layers=target_layers,use_cuda=True),
            RandomCAM(model=model_select,target_layers=target_layers,use_cuda=True)
            ]
            
            #pytorch_grad_cam init XAI metrics
            pytorch_gradcam_metrics = [
            (ROADMostRelevantFirst(percentile=50),{"return_visualization":True}),
            (ROADLeastRelevantFirst(percentile=50),{"return_visualization":True}),
            (ROADCombined(percentiles=[20, 40, 60, 80]),{}),
            
            ]
            #Captum init cams
            captum_alg = [
                IntegratedGradients(model_select),
                Occlusion(model_select),
                GuidedGradCam(model=model_select,layer=target_layers[0]),
                Saliency(model_select)
                ]
            
            #Create the confusion matrix
            if self.log_config['confusion_matrix']:
                print('Computing confusion matrix...')
                self.confusion_matrix(all_predictions,all_labels)
                computed_confusion_pre = self.confusion_matrix.compute()
                computed_confusion = computed_confusion_pre.detach().cpu().numpy().astype(int)
                fig = utils.get_confusion_matrix_figure(computed_confusion=computed_confusion)
                self.logger.experiment.add_figure(f'{model_str}Confusion matrix',fig,self.current_epoch)
                
                
            #Create ROC-Curve
            if self.log_config['roc_curve']:
                print('Computing ROC curves...')
                fpr, tpr, thresholds = self.roc_curve(all_predictions,all_labels)
                #For each class, create a seperate roc-curve
                for i in range(len(fpr)):
                    fig = utils.get_roc_curve_figure(fpr=fpr[i].cpu(),tpr=tpr[i].cpu(),thresholds=thresholds[i].cpu(),classid=i)
                    self.logger.experiment.add_figure(f'{model_str}ROC curve/for class {i}',fig,self.current_epoch)
            
            #Create AUROC evtl für jede klasse?
            if self.log_config['auroc']:
                print('Computing Area under ROC...')
                auroc = self.auroc(all_predictions,all_labels)
                self.log(f'{model_str}Area under ROC',auroc,batch_size=self.batch_size)
    
            #Create classification report
            if self.log_config['classification_report']:
                print('Computing classification report...')
                report = classification_report(all_labels.cpu(),torch.argmax(all_predictions,dim=1).cpu())
                self.logger.experiment.add_text(f'{model_str}Classification report',report)


            #XAI methods
            for function, suffix in selection_functions:
                print(suffix,'XAI methods...')
                
                xai_metrics_dict = {}
                
                best_k_props = function(all_predictions_max_prop.values)
                
                best_props = best_k_props[0]
                
                best_idx = best_k_props[1]
                
                #get labels
                best_labels = all_labels[best_idx]
                
                #get predicted label
                best_predictions = all_predictions_idx[best_idx]
                                
                #filter for confidently wrong predictions
                if suffix == "conf_false":
                    
                    condition_mask = best_labels != best_predictions
                    
                    best_props = best_props[condition_mask][:TOP_K]
                    
                    best_idx = best_idx[condition_mask][:TOP_K]
                    
                    best_labels = best_labels[condition_mask][:TOP_K]
                    
                    best_predictions = best_predictions[condition_mask][:TOP_K]
                    
                    
                #filter for inconfident wrong predictions
                if suffix == "inconf_false":
                    
                    condition_mask = best_labels != best_predictions
                    
                    best_props = best_props[condition_mask][-TOP_K:]
                    
                    best_idx = best_idx[condition_mask][-TOP_K:]
                    
                    best_labels = best_labels[condition_mask][-TOP_K:]
                    
                    best_predictions = best_predictions[condition_mask][-TOP_K:]
                
                #filter for inconfident true predictions
                if suffix == "inconf_true":
                    
                    condition_mask = best_labels == best_predictions
                    
                    best_props = best_props[condition_mask][-TOP_K:]
                    
                    best_idx = best_idx[condition_mask][-TOP_K:]
                    
                    best_labels = best_labels[condition_mask][-TOP_K:]
                    
                    best_predictions = best_predictions[condition_mask][-TOP_K:]
                
                #get labelnames
                best_label_names = self.label_names[best_labels.cpu()]
                
                #get predicted labelname
                best_predicted_label_names = self.label_names[best_predictions.cpu()]
                
                #get images
                best_images = all_images[best_idx]
                        
                #save/log softmax vector for each idx in tensorboard
                best_softmax_idx = all_predictions_prop[best_idx]
    
                #get ground truth probability for each of the top_k softmaxvectors
                best_gt_prob = best_softmax_idx[torch.arange(len(best_labels)),best_labels]
                                
                #create softmax with labelnames
                for idx,softmax in enumerate(best_softmax_idx):
                    best_softmax_idx_wlabelnames = {}
                    
                    for key,val in self.label_names_dict.items():
                        best_softmax_idx_wlabelnames[key] = (val,f"{softmax[val].item():2f}")
                    
                    top_10_probs = sorted(best_softmax_idx_wlabelnames.items(), key=lambda x: float(x[1][1]), reverse=True)[:10]
                    #save softmax with labelnames
                    self.logger.experiment.add_text(f'{model_str}{suffix}_{TOP_K}_Softmax/idx_{best_idx[idx]}',str(best_softmax_idx_wlabelnames) + "********" + "top_10_probs: " + str(top_10_probs))
                
                #preprocess for both captum and pytorchcams
                resize = transforms.Resize(224)
                rgb_images = resize(best_images)
                denormalized_images = denormalize(rgb_images,(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)) #train.imagenetstd, train.imagenetmean
                # targets = [ClassifierOutputTarget(NUM_CLASSES-1)]*len(best_props)
                # targets = best_predictions
                targets_metric = list(map(ClassifierOutputSoftmaxTarget, best_predictions))
                targets_cam = list(map(ClassifierOutputTarget, best_predictions))
                #apply XAI methods for every image in k_images
                if self.log_config['pytorch_cam']:
                    print("pytorch_cams...")
                    
                                        
                    #apply pytorch_gradcam_cam
                    for cam in pytorch_gradcam_cams:
                        
                        with torch.enable_grad():
                            # g_image = dict_image["image"].unsqueeze(0)
                            # cam_images = cam(input_tensor=best_images, targets=targets_cam,aug_smooth=True,eigen_smooth=False)
                            # cam_images = cam(input_tensor=best_images, targets=targets_cam)
                            cam_images = utils.process_function_in_batches(images=best_images, batch_size=TOP_K, func=partial(cam))
                        cam_name = str(type(cam)).split(".")[-1][:-2]
                        print(cam_name)
                        

                        #calc cam metric for best images
                        for cam_metric, arguments in pytorch_gradcam_metrics:
                            metric_name = str(type(cam_metric)).split(".")[-1][:-2]
                            #calculate cam_metric
                                         
                            output = utils.process_function_in_batches(images=best_images, batch_size=TOP_K, func=partial(cam_metric,cams=cam_images, targets=targets_metric, model=model_select,**arguments))
                            
                            #get images if existent
                            if isinstance(output, tuple):
                                scores, perturbated_images = output
                            else:
                                scores = output


                            #save pytorch_gradcam_cam
                            for idx, cam_image in enumerate(cam_images):
                                
                                image_score = scores[idx]
                                
                                #visualize cam_image
                                permuted_image = denormalized_images[idx].permute(1,2,0).cpu()
                                visualization = show_cam_on_image(permuted_image.numpy().astype(numpy.float32)/255, cam_image, use_rgb=True,image_weight=0.5)
                                images,titles = [permuted_image,cam_image,visualization],['Original','Heat map', 'Combined']
                                
                                if isinstance(output, tuple):
                                    
                                    denormalized_perturbated_image = denormalize(perturbated_images[idx].cpu(),(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)).permute(1,2,0)
                                    images.append(denormalized_perturbated_image)
                                    titles.append('Perturbated')
                                
                                result = create_multiple_images(images=images,titles=titles)
                                utils.add_to_nested_dict(dictionary=xai_metrics_dict, outer_key=f"PytorchGradcams_{cam_name}", mid_key=metric_name, inner_key=int(best_labels[idx]), value=image_score)
                                self.logger.experiment.add_figure(f'{model_str}{suffix}_{TOP_K}/PytorchGradcams_{cam_name} *** Groundtruth: {best_label_names[idx]} (label: {best_labels[idx]}) gt_prob: {best_gt_prob[idx]:.2f} *** predicted_label_name: {best_predicted_label_names[idx]} (label: {best_predictions[idx]}) Predicted_prop: {best_props[idx]:.2f} *** {metric_name} = {image_score} *** Idx: {best_idx[idx]}',result)

                                    
                #apply captum
                if self.log_config['captum_alg']:
                    print("captum_cams...")
                    
                    grayscale = transforms.Grayscale(num_output_channels = 1)

                    
                    def normalize_to_01(images: np.ndarray) -> np.ndarray:
                        normalized_images = []
                        for image in images:
                            
                            min_val = np.min(image)
                            max_val = np.max(image)
                            normalized_image = (image - min_val) / (1e-7 + max_val )
                            
                            normalized_images.append(np.float32(normalized_image))
                        
                        return np.stack(normalized_images)


                    for alg in captum_alg:
                        
                        #cam name as string
                        cam_name = str(type(alg)).split(".")[-1][:-2]
                        print(cam_name)
                        
                        noise_tunnel = NoiseTunnel(alg)
                        
                        #apply attributions
                        if cam_name == "IntegratedGradients":
                            # attributions_nt = noise_tunnel.attribute(best_images, nt_samples=1,nt_samples_batch_size=1 ,nt_type='smoothgrad_sq', target=best_predictions,internal_batch_size=1)
                            attributions_nt = utils.process_function_in_batches(images=best_images, batch_size=TOP_K,targets=best_predictions,func=partial(noise_tunnel.attribute,nt_samples=1,nt_samples_batch_size=1,nt_type='smoothgrad_sq',internal_batch_size=TOP_K, n_steps=50))
                            
                        elif cam_name == "Occlusion":
                            # attributions_nt = noise_tunnel.attribute(best_images, nt_samples=2,nt_samples_batch_size=2 ,nt_type='smoothgrad_sq', target=best_predictions,strides=(3, 8, 8),sliding_window_shapes=(3, 15, 15),baselines=0)
                            
                            attributions_nt = utils.process_function_in_batches(images=best_images, batch_size=TOP_K, targets=best_predictions,func=partial(noise_tunnel.attribute,nt_samples=1,nt_samples_batch_size=1 ,nt_type='smoothgrad_sq',strides=(3, 8, 8),sliding_window_shapes=(3, 15, 15),baselines=0))

                        
                        else:
                            # attributions_nt = noise_tunnel.attribute(best_images, nt_type='smoothgrad_sq', target=best_predictions)
                            
                            attributions_nt = utils.process_function_in_batches(images=best_images, batch_size=TOP_K, targets=best_predictions,func=partial(noise_tunnel.attribute,nt_type='smoothgrad_sq'))

                            
                        # attributions_nt = noise_tunnel.attribute(best_images, nt_type='smoothgrad_sq', target=best_predictions)
                        
                        grayscale_attributions_nt = grayscale(attributions_nt)
                        heatmap_images = normalize_to_01(grayscale_attributions_nt.cpu().numpy()).squeeze()
                        
                        #calc cam metric for best images
                        for cam_metric, arguments in pytorch_gradcam_metrics:
                            metric_name = str(type(cam_metric)).split(".")[-1][:-2]
                            #calculate cam_metric
                            # output = cam_metric(best_images, heatmap_images, targets_metric, model_select,**arguments)
                            output = utils.process_function_in_batches(images=best_images, batch_size=TOP_K, func=partial(cam_metric,cams=heatmap_images, targets=targets_metric, model=model_select,**arguments))

                            #get images if existent
                            if isinstance(output, tuple):
                                scores, perturbated_images = output
                            else:
                                scores = output

                        
                        
                            for idx, heatmap_image in enumerate(heatmap_images):
                                            
                                image_score = scores[idx]
                                
                                #visualize cam_image
                                permuted_image = denormalized_images[idx].permute(1,2,0).cpu()
                                visualization = show_cam_on_image(permuted_image.numpy().astype(numpy.float32)/255, heatmap_image, use_rgb=True,image_weight=0.5,colormap=9)
                                images,titles = [permuted_image,heatmap_image,visualization],['Original','Heat map', 'Combined']
                                
                                if isinstance(output, tuple):
                                    
                                    denormalized_perturbated_image = denormalize(perturbated_images[idx].cpu(),(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)).permute(1,2,0)
                                    images.append(denormalized_perturbated_image)
                                    titles.append('Perturbated')

                                result = create_multiple_images(images=images,titles=titles)
                                
                                #print image in console
                                # show_multiple_images(images=images,titles=titles)
    
                                utils.add_to_nested_dict(dictionary=xai_metrics_dict, outer_key=f"Captum_{cam_name}", mid_key=metric_name, inner_key=int(best_labels[idx]), value=image_score)
                                self.logger.experiment.add_figure(f'{model_str}{suffix}_{TOP_K}/Captum_{cam_name} *** Groundtruth: {best_label_names[idx]} (label: {best_labels[idx]}) gt_prob: {best_gt_prob[idx]:.2f} *** predicted_label_name: {best_predicted_label_names[idx]} (label: {best_predictions[idx]}) Predicted_prop: {best_props[idx]:.2f} *** {metric_name} = {image_score} *** Idx: {best_idx[idx]}',result)
                                #clear cuda cache
                                torch.cuda.empty_cache()
                            
                        #clear cuda cache
                        torch.cuda.empty_cache()

      
                #save xai_metrics_dict
                self.logger.experiment.add_text(f'{model_str}{suffix}_{TOP_K}',str(xai_metrics_dict))
                
                # Save to a .txt file
                # with open(self.name + "_xai_metrics_dict_r34dsd_randomcam50.txt", "w") as file:
                #     file.write(str(xai_metrics_dict))

                    


        #Clear values that got created in test_step, necessary if more than one test_step epoch
        if self.kd_run:
            self.test_step_prediction_teacher.clear()
        
        self.test_step_prediction.clear()
        self.test_step_label.clear()
        self.test_step_input.clear()


#################################################################################Online KD Classes##################################################################
class ResponseBasedOnline(ImageClassifierBase):
    #give params to ResponseBasedOnline if used
    def __init__(self,
                 student_model,
                 teacher_model,
                 lr=0.1,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm = 'sgd',
                 num_workers = 4,
                 note = '',
                 resize_size = (224,224),
                 data_split_ratios = [0.8,0.15,0.05],
                 alpha=0.95,
                 T=3.5
                 ):
        #give to ImageClassifierBase
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            weight_decay=weight_decay,
            norm_weight_decay=norm_weight_decay,
            label_smoothing=label_smoothing,
            lr_scheduler=lr_scheduler,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_warmup_method=lr_warmup_method,
            lr_warmup_decay=lr_warmup_decay,
            optimizer_algorithm = optimizer_algorithm,
            num_workers = num_workers,
            note = note,
            resize_size = resize_size,
            data_split_ratios = data_split_ratios,
            kd_run = True
            )
        #änderung die hier passieren werden nicht korrekt geloggt!
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.T = T
        self.test_step_prediction_teacher = []
        self.name = f'{self.student_model.name}_{self.teacher_model.name}'
        self.kd_name = self.__class__.__name__
        self.save_hyperparameters({"T":self.T,"alpha":self.alpha,"name":self.name,"kd_name":self.kd_name})
                
        #add hyperparameters of each model to the hparams dict of the KD class
        for key in self.student_model.hparams:
            self.save_hyperparameters({f"student_{key}":self.student_model.hparams[key]})
            
        for key in self.teacher_model.hparams:
            self.save_hyperparameters({f"teacher_{key}":self.teacher_model.hparams[key]})
    
    def init_base_model(self):
        pass
    
    #self.model ist in KD class nicht verfügbar!
    def forward(self, x):
        return self.student_model(x)


    def training_step(self, batch,batch_idx):
        inputs, labels = batch
        output_student = self.student_model(inputs)
        output_teacher = self.teacher_model(inputs)
                
        #calculate loss and kd loss
        kd_loss, ce_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   label_smoothing=self.label_smoothing,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        
        #remove impact of alpha and T. this is done for better compareability between same models with different alpha and T
        true_kd_loss = kd_loss / (self.alpha * self.T * self.T)
        true_ce_loss = ce_loss / (1. - self.alpha)
        true_total_loss = true_kd_loss + true_ce_loss
        true_ce_loss_teacher = F.cross_entropy(output_teacher,labels,label_smoothing=self.label_smoothing)
                
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)
        
        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)
        
        self.logger.experiment.add_scalars('Accuracy_student',
                                           {'train': accuracy_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_teacher',
                                           {'train': accuracy_teacher},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5_student', 
                                           {'train': accuracy_top_5_student},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Accuracy_top_5_teacher', 
                                           {'train': accuracy_top_5_teacher},
                                           global_step=self.current_epoch)


        self.logger.experiment.add_scalars('KD_loss', 
                                           {'train': kd_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('CE_loss', 
                                           {'train': ce_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Total_loss', 
                                           {'train': total_loss},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('All_losses', 
                                           {'total_loss_train': total_loss,
                                            'ce_loss_train': ce_loss,
                                            'kd_loss_train': kd_loss    
                                            },
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('True_all_losses', 
                                           {'true_total_loss_train': true_total_loss,
                                            'true_ce_loss_train': true_ce_loss,
                                            'true_kd_loss_train': true_kd_loss,
                                            'true_ce_loss_teacher_train': true_ce_loss_teacher,
                                            },
                                           global_step=self.current_epoch)

        
        self.logger.experiment.add_scalars('All_accuracies', 
                                           {'student_train': accuracy_student,
                                            'teacher_train': accuracy_teacher,
                                            },
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('MCC_student', 
                                           {'train': mcc_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('MCC_teacher', 
                                           {'train': mcc_teacher},
                                           global_step=self.current_epoch)
        
        self.log("TA",accuracy_student,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("TAT",accuracy_teacher,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("TL",total_loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)


        return total_loss

    def validation_step(self, batch,batch_idx):
        inputs, labels = batch
        output_student = self.student_model(inputs)
        output_teacher = self.teacher_model(inputs)
                
        #calculate loss and kd loss
        kd_loss, ce_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   label_smoothing=self.label_smoothing,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        
        #remove impact of alpha and T. this is done for better compareability between same models with different alpha and T
        true_kd_loss = kd_loss / (self.alpha * self.T * self.T)
        true_ce_loss = ce_loss / (1. - self.alpha)
        true_total_loss = true_kd_loss + true_ce_loss
        true_ce_loss_teacher = F.cross_entropy(output_teacher,labels,label_smoothing=self.label_smoothing)

        
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)
        
        self.logger.experiment.add_scalars('Accuracy_student',
                                           {'validation': accuracy_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_teacher',
                                           {'validation': accuracy_teacher},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5_student', 
                                           {'validation': accuracy_top_5_student},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Accuracy_top_5_teacher', 
                                           {'validation': accuracy_top_5_teacher},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('KD_loss', 
                                           {'validation': kd_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('CE_loss', 
                                           {'validation': ce_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Total_loss', 
                                           {'validation': total_loss},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('All_losses', 
                                           {'total_loss_validation': total_loss,
                                            'ce_loss_validation': ce_loss,
                                            'kd_loss_validation': kd_loss    
                                            },
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('True_all_losses', 
                                           {'true_total_loss_validation': true_total_loss,
                                            'true_ce_loss_validation': true_ce_loss,
                                            'true_kd_loss_validation': true_kd_loss,
                                            'true_ce_loss_teacher_validation': true_ce_loss_teacher,
                                            },
                                           global_step=self.current_epoch)

        
        self.logger.experiment.add_scalars('All_accuracies', 
                                           {'student_validation': accuracy_student,
                                            'teacher_validation': accuracy_teacher,
                                            },
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('MCC_student', 
                                           {'validation': mcc_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('MCC_teacher', 
                                           {'validation': mcc_teacher},
                                           global_step=self.current_epoch)
        
        #please ignore, only for ModelCheckpoint(....)
        self.log("TuCL",true_ce_loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("TuCLT",true_ce_loss_teacher,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VL",total_loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VAT",accuracy_teacher,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("VA",accuracy_student,on_step=False,prog_bar=True,batch_size=self.batch_size)
        
        return total_loss

    def setup(self, stage):
        #prevents trainer from logging twice, if test step is executed directly after train step
        if self.setup_run:
            #log architecture graph
            self.logger._log_graph = True
            self.logger.log_graph(self,torch.rand((1,3) + self.resize_size).to('cuda'))
            
            self.logger.experiment.add_text('student_model_architecture',str(self.student_model))
            self.logger.experiment.add_text('teacher_model_architecture',str(self.teacher_model))

        
        
    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        
        output_student = self.student_model(inputs)
        output_teacher = self.teacher_model(inputs)
                
        #calculate loss and kd loss
        kd_loss, ce_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   label_smoothing=self.label_smoothing,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        
        #remove impact of alpha and T. this is done for better compareability between same models with different alpha and T
        true_kd_loss = kd_loss / (self.alpha * self.T * self.T)
        true_ce_loss = ce_loss / (1. - self.alpha)
        true_total_loss = true_kd_loss + true_ce_loss
        true_ce_loss_teacher = F.cross_entropy(output_teacher,labels,label_smoothing=self.label_smoothing)

        
        
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)

        #predictions = torch.argmax(output,dim=1)
        
        self.log("test_accuracy_student",accuracy_student,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_top_5_student",accuracy_top_5_student,prog_bar=False,batch_size=self.batch_size)
        self.log("test_accuracy_top_5_teacher",accuracy_top_5_teacher,prog_bar=False,batch_size=self.batch_size)

        self.log("test_mcc_student",mcc_student,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_teacher",accuracy_teacher,prog_bar=True,batch_size=self.batch_size)
        self.log("test_mcc_teacher",mcc_teacher,prog_bar=True,batch_size=self.batch_size)
        
        self.log("test_kd_loss",kd_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_ce_loss",ce_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_total_loss",total_loss,prog_bar=True,batch_size=self.batch_size)
        
        self.log("test_true_kd_loss",true_kd_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_true_ce_loss",true_ce_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_true_total_loss",true_total_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_true_ce_loss_teacher",true_ce_loss_teacher,prog_bar=True,batch_size=self.batch_size)


        
        self.test_step_prediction.append(output_student)
        self.test_step_prediction_teacher.append(output_teacher)
        self.test_step_label.append(labels)
        self.test_step_input.append(inputs)
        return total_loss


class FeatureBasedOnline(ImageClassifierBase):
    #give params to FeatureBasedOnline if used
    def __init__(self,
                 student_model,
                 teacher_model,
                 lr=0.1,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm = 'sgd',
                 num_workers = 4,
                 note = '',
                 resize_size = (224,224),
                 data_split_ratios = [0.8,0.15,0.05],
                 alpha=0.95,
                 beta=0.05
                 ):
        #give to ImageClassifierBase
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            weight_decay=weight_decay,
            norm_weight_decay=norm_weight_decay,
            label_smoothing=label_smoothing,
            lr_scheduler=lr_scheduler,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_warmup_method=lr_warmup_method,
            lr_warmup_decay=lr_warmup_decay,
            optimizer_algorithm = optimizer_algorithm,
            num_workers = num_workers,
            note = note,
            resize_size = resize_size,
            data_split_ratios = data_split_ratios,
            kd_run = True
            )
        #?
        #änderung die hier passieren werden nicht korrekt geloggt!
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.beta = beta
        self.test_step_prediction_teacher = []
        self.name = f'{self.student_model.name}_{self.teacher_model.name}'
        self.kd_name = self.__class__.__name__
        self.save_hyperparameters({"beta":self.beta,"alpha":self.alpha,"name":self.name,"kd_name":self.kd_name})
                
        #add hyperparameters of each model to the hparams dict of the KD class
        for key in self.student_model.hparams:
            self.save_hyperparameters({f"student_{key}":self.student_model.hparams[key]})
            
        for key in self.teacher_model.hparams:
            self.save_hyperparameters({f"teacher_{key}":self.teacher_model.hparams[key]})
    
    def init_base_model(self):
        pass
    
    def forward(self, x):
        return self.student_model(x)


    def training_step(self, batch,batch_idx):
        inputs, labels = batch
                
        #features
        features_student, output_student = self.student_model.model(x=inputs,is_feat=True)
        features_teacher, output_teacher = self.teacher_model.model(x=inputs,is_feat=True)
        
        kd_loss, ce_loss, ce_loss_teacher, total_loss, true_total_loss = feature_based_distillation_loss(student_output=output_student,
                                                                                                                 teacher_output=output_teacher,
                                                                                                                 student_layer_features=features_student[-1],
                                                                                                                 teacher_layer_features=features_teacher[-1],
                                                                                                                 labels=labels,
                                                                                                                 label_smoothing=self.label_smoothing,
                                                                                                                 alpha=self.alpha,
                                                                                                                 beta=self.beta
                                                                                                                 )
        
        
        
        
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)
        
        self.logger.experiment.add_scalars('Accuracy_student',
                                           {'train': accuracy_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_teacher',
                                           {'train': accuracy_teacher},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5_student', 
                                           {'train': accuracy_top_5_student},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Accuracy_top_5_teacher', 
                                           {'train': accuracy_top_5_teacher},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('KD_loss', 
                                           {'train': kd_loss},
                                           global_step=self.current_epoch)
        
        # self.logger.experiment.add_scalars('KD_loss_output', 
        #                                    {'train': kd_loss_output},
        #                                    global_step=self.current_epoch)


        self.logger.experiment.add_scalars('CE_loss', 
                                           {'train': ce_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Total_loss', 
                                           {'train': total_loss},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('All_losses', 
                                           {'total_loss_train': total_loss,
                                            'ce_loss_train': ce_loss,
                                            'kd_loss_train': kd_loss  ,
                                            'true_total_loss_train': true_total_loss,
                                            'ce_loss_teacher_train':ce_loss_teacher,
                                            # 'kd_loss_output_train':kd_loss_output,
                                            },
                                           global_step=self.current_epoch)
        
        
        self.logger.experiment.add_scalars('All_accuracies', 
                                           {'student_train': accuracy_student,
                                            'teacher_train': accuracy_teacher,
                                            },
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('MCC_student', 
                                           {'train': mcc_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('MCC_teacher', 
                                           {'train': mcc_teacher},
                                           global_step=self.current_epoch)
        
        self.log("TA",accuracy_student,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("TAT",accuracy_teacher,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("TL",total_loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)

        return total_loss

    def validation_step(self, batch,batch_idx):
        inputs, labels = batch
        
        #features
        features_student, output_student = self.student_model.model(x=inputs,is_feat=True)
        features_teacher, output_teacher = self.teacher_model.model(x=inputs,is_feat=True)
        
        kd_loss, ce_loss, ce_loss_teacher, total_loss, true_total_loss = feature_based_distillation_loss(student_output=output_student,
                                                                                                                 teacher_output=output_teacher,
                                                                                                                 student_layer_features=features_student[-1],
                                                                                                                 teacher_layer_features=features_teacher[-1],
                                                                                                                 labels=labels,
                                                                                                                 label_smoothing=self.label_smoothing,
                                                                                                                 alpha=self.alpha,
                                                                                                                 beta=self.beta
                                                                                                                 )

        
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)
        
        self.logger.experiment.add_scalars('Accuracy_student',
                                           {'validation': accuracy_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_teacher',
                                           {'validation': accuracy_teacher},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5_student', 
                                           {'validation': accuracy_top_5_student},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Accuracy_top_5_teacher', 
                                           {'validation': accuracy_top_5_teacher},
                                           global_step=self.current_epoch)


        self.logger.experiment.add_scalars('KD_loss', 
                                           {'validation': kd_loss},
                                           global_step=self.current_epoch)
        


        self.logger.experiment.add_scalars('CE_loss', 
                                           {'validation': ce_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Total_loss', 
                                           {'validation': total_loss},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('All_losses', 
                                           {'total_loss_validation': total_loss,
                                            'ce_loss_validation': ce_loss,
                                            'kd_loss_validation': kd_loss  ,
                                            'true_total_loss_validation': true_total_loss,
                                            'ce_loss_teacher_validation':ce_loss_teacher,
                                            # 'kd_loss_output_validation':kd_loss_output,

                                            },
                                           global_step=self.current_epoch)
        
        
        self.logger.experiment.add_scalars('All_accuracies', 
                                           {'student_validation': accuracy_student,
                                            'teacher_validation': accuracy_teacher,
                                            },
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('MCC_student', 
                                           {'validation': mcc_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('MCC_teacher', 
                                           {'validation': mcc_teacher},
                                           global_step=self.current_epoch)
        
        #please ignore, only for ModelCheckpoint(....)
        self.log("TuCL",ce_loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("TuCLT",ce_loss_teacher,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VL",total_loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VAT",accuracy_teacher,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("VA",accuracy_student,on_step=False,prog_bar=True,batch_size=self.batch_size)


        return total_loss

    def setup(self, stage):
        #prevents trainer from logging twice, if test step is executed directly after train step
        if self.setup_run:
            #log architecture graph
            self.logger._log_graph = True
            self.logger.log_graph(self,torch.rand((1,3) + self.resize_size).to('cuda'))
            
            self.logger.experiment.add_text('student_model_architecture',str(self.student_model))
            self.logger.experiment.add_text('teacher_model_architecture',str(self.teacher_model))

        
        
    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        
        #features
        features_student, output_student = self.student_model.model(x=inputs,is_feat=True)
        features_teacher, output_teacher = self.teacher_model.model(x=inputs,is_feat=True)
        
        kd_loss, ce_loss, ce_loss_teacher, total_loss, true_total_loss = feature_based_distillation_loss(student_output=output_student,
                                                                                                                 teacher_output=output_teacher,
                                                                                                                 student_layer_features=features_student[-1],
                                                                                                                 teacher_layer_features=features_teacher[-1],
                                                                                                                 labels=labels,
                                                                                                                 label_smoothing=self.label_smoothing,
                                                                                                                 alpha=self.alpha,
                                                                                                                 beta=self.beta
                                                                                                                 )

                
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)

        #predictions = torch.argmax(output,dim=1)
        
        self.log("test_accuracy_student",accuracy_student,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_top_5_student",accuracy_top_5_student,prog_bar=False,batch_size=self.batch_size)
        self.log("test_accuracy_top_5_teacher",accuracy_top_5_teacher,prog_bar=False,batch_size=self.batch_size)

        self.log("test_mcc_student",mcc_student,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_teacher",accuracy_teacher,prog_bar=True,batch_size=self.batch_size)
        self.log("test_mcc_teacher",mcc_teacher,prog_bar=True,batch_size=self.batch_size)
        
        self.log("test_kd_loss",kd_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_ce_loss",ce_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_ce_loss_teacher",ce_loss_teacher,prog_bar=True,batch_size=self.batch_size)
        self.log("test_total_loss",total_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_true_total_loss",true_total_loss,prog_bar=True,batch_size=self.batch_size)

        
        self.test_step_prediction.append(output_student)
        self.test_step_prediction_teacher.append(output_teacher)
        self.test_step_label.append(labels)
        self.test_step_input.append(inputs)
        return total_loss

#################################################################################Offline KD Classes##################################################################
class ResponseBasedOffline(ImageClassifierBase):
    #give params to ResponseBasedOffline if used
    def __init__(self,
                 student_model,
                 teacher_model,
                 lr=0.1,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm = 'sgd',
                 num_workers = 4,
                 note = '',
                 resize_size = (224,224),
                 data_split_ratios = [0.8,0.15,0.05],
                 alpha=0.95,
                 T=3.5
                 ):
        #give to ImageClassifierBase
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            weight_decay=weight_decay,
            norm_weight_decay=norm_weight_decay,
            label_smoothing=label_smoothing,
            lr_scheduler=lr_scheduler,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_warmup_method=lr_warmup_method,
            lr_warmup_decay=lr_warmup_decay,
            optimizer_algorithm = optimizer_algorithm,
            num_workers = num_workers,
            note = note,
            resize_size = resize_size,
            data_split_ratios = data_split_ratios,
            kd_run = True
            )
        #?
        #änderung die hier passieren werden nicht korrekt geloggt!
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.T = T
        self.test_step_prediction_teacher = []
        self.name = f'{self.student_model.name}_{self.teacher_model.name}'
        self.kd_name = self.__class__.__name__
        self.save_hyperparameters({"T":self.T,"alpha":self.alpha,"name":self.name,"kd_name":self.kd_name})
        
        
        #freeze teacher model for offline kd
        self.teacher_model.freeze_layers()
                
        #add hyperparameters of each model to the hparams dict of the KD class
        for key in self.student_model.hparams:
            self.save_hyperparameters({f"student_{key}":self.student_model.hparams[key]})
            
        for key in self.teacher_model.hparams:
            self.save_hyperparameters({f"teacher_{key}":self.teacher_model.hparams[key]})
            
    
    def init_base_model(self):
        pass
    
    #self.model ist in KD class nicht verfügbar!!!!
    def forward(self, x):
        return self.student_model(x)


    def training_step(self, batch,batch_idx):
        inputs, labels = batch
        
        #turn off dropout for teacher model
        self.teacher_model.eval()
        
        #output
        output_student = self.student_model(inputs)
        
        #only inference for teacher
        with torch.no_grad():
            output_teacher = self.teacher_model(inputs)
                
        #calculate loss and kd loss
        kd_loss, ce_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   label_smoothing=self.label_smoothing,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        
        #remove impact of alpha and T. this is done for better compareability between same models with different alpha and T
        true_kd_loss = kd_loss / (self.alpha * self.T * self.T)
        true_ce_loss = ce_loss / (1. - self.alpha)
        true_total_loss = true_kd_loss + true_ce_loss
        true_ce_loss_teacher = F.cross_entropy(output_teacher,labels,label_smoothing=self.label_smoothing)
                
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)
        
        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)
        
        self.logger.experiment.add_scalars('Accuracy_student',
                                           {'train': accuracy_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_teacher',
                                           {'train': accuracy_teacher},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5_student', 
                                           {'train': accuracy_top_5_student},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Accuracy_top_5_teacher', 
                                           {'train': accuracy_top_5_teacher},
                                           global_step=self.current_epoch)


        self.logger.experiment.add_scalars('KD_loss', 
                                           {'train': kd_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('CE_loss', 
                                           {'train': ce_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Total_loss', 
                                           {'train': total_loss},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('All_losses', 
                                           {'total_loss_train': total_loss,
                                            'ce_loss_train': ce_loss,
                                            'kd_loss_train': kd_loss    
                                            },
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('True_all_losses', 
                                           {'true_total_loss_train': true_total_loss,
                                            'true_ce_loss_train': true_ce_loss,
                                            'true_kd_loss_train': true_kd_loss,
                                            'true_ce_loss_teacher_train': true_ce_loss_teacher,
                                            },
                                           global_step=self.current_epoch)

        
        self.logger.experiment.add_scalars('All_accuracies', 
                                           {'student_train': accuracy_student,
                                            'teacher_train': accuracy_teacher,
                                            },
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('MCC_student', 
                                           {'train': mcc_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('MCC_teacher', 
                                           {'train': mcc_teacher},
                                           global_step=self.current_epoch)
        
        self.log("TA",accuracy_student,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("TAT",accuracy_teacher,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("TL",total_loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)


        return total_loss

    def validation_step(self, batch,batch_idx):
        inputs, labels = batch
        #turn off dropout for teacher model
        self.teacher_model.eval()
        
        #output
        output_student = self.student_model(inputs)
        
        #only inference for teacher
        with torch.no_grad():
            output_teacher = self.teacher_model(inputs)
                
        #calculate loss and kd loss
        kd_loss, ce_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   label_smoothing=self.label_smoothing,
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        
        #remove impact of alpha and T. this is done for better compareability between same models with different alpha and T
        true_kd_loss = kd_loss / (self.alpha * self.T * self.T)
        true_ce_loss = ce_loss / (1. - self.alpha)
        true_total_loss = true_kd_loss + true_ce_loss
        true_ce_loss_teacher = F.cross_entropy(output_teacher,labels,label_smoothing=self.label_smoothing)

        
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)
        
        self.logger.experiment.add_scalars('Accuracy_student',
                                           {'validation': accuracy_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_teacher',
                                           {'validation': accuracy_teacher},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5_student', 
                                           {'validation': accuracy_top_5_student},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Accuracy_top_5_teacher', 
                                           {'validation': accuracy_top_5_teacher},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('KD_loss', 
                                           {'validation': kd_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('CE_loss', 
                                           {'validation': ce_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Total_loss', 
                                           {'validation': total_loss},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('All_losses', 
                                           {'total_loss_validation': total_loss,
                                            'ce_loss_validation': ce_loss,
                                            'kd_loss_validation': kd_loss    
                                            },
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('True_all_losses', 
                                           {'true_total_loss_validation': true_total_loss,
                                            'true_ce_loss_validation': true_ce_loss,
                                            'true_kd_loss_validation': true_kd_loss,
                                            'true_ce_loss_teacher_validation': true_ce_loss_teacher,
                                            },
                                           global_step=self.current_epoch)

        
        self.logger.experiment.add_scalars('All_accuracies', 
                                           {'student_validation': accuracy_student,
                                            'teacher_validation': accuracy_teacher,
                                            },
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('MCC_student', 
                                           {'validation': mcc_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('MCC_teacher', 
                                           {'validation': mcc_teacher},
                                           global_step=self.current_epoch)
        
        #please ignore, only for ModelCheckpoint(....)
        self.log("TuCL",true_ce_loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("TuCLT",true_ce_loss_teacher,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VL",total_loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VAT",accuracy_teacher,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("VA",accuracy_student,on_step=False,prog_bar=True,batch_size=self.batch_size)
        
        return total_loss

    def setup(self, stage):
        #prevents trainer from logging twice, if test step is executed directly after train step
        if self.setup_run:
            #log architecture graph
            self.logger._log_graph = True
            self.logger.log_graph(self,torch.rand((1,3) + self.resize_size).to('cuda'))
            
            self.logger.experiment.add_text('student_model_architecture',str(self.student_model))
            self.logger.experiment.add_text('teacher_model_architecture',str(self.teacher_model))

        
        
    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        
        #turn off dropout for teacher model
        self.teacher_model.eval()
        
        #output
        output_student = self.student_model(inputs)
        
        #only inference for teacher
        with torch.no_grad():
            output_teacher = self.teacher_model(inputs)
                
        #calculate loss and kd loss
        kd_loss, ce_loss, total_loss = knowledge_distillation_loss(student_output=output_student,
                                                                   teacher_output=output_teacher,
                                                                   labels=labels,
                                                                   label_smoothing=self.label_smoothing, 
                                                                   alpha=self.alpha,
                                                                   T=self.T)
        
        #remove impact of alpha and T. this is done for better compareability between same models with different alpha and T
        true_kd_loss = kd_loss / (self.alpha * self.T * self.T)
        true_ce_loss = ce_loss / (1. - self.alpha)
        true_total_loss = true_kd_loss + true_ce_loss
        true_ce_loss_teacher = F.cross_entropy(output_teacher,labels,label_smoothing=self.label_smoothing)

        
        
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)

        #predictions = torch.argmax(output,dim=1)
        
        self.log("test_accuracy_student",accuracy_student,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_top_5_student",accuracy_top_5_student,prog_bar=False,batch_size=self.batch_size)
        self.log("test_accuracy_top_5_teacher",accuracy_top_5_teacher,prog_bar=False,batch_size=self.batch_size)

        self.log("test_mcc_student",mcc_student,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_teacher",accuracy_teacher,prog_bar=True,batch_size=self.batch_size)
        self.log("test_mcc_teacher",mcc_teacher,prog_bar=True,batch_size=self.batch_size)
        
        self.log("test_kd_loss",kd_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_ce_loss",ce_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_total_loss",total_loss,prog_bar=True,batch_size=self.batch_size)
        
        self.log("test_true_kd_loss",true_kd_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_true_ce_loss",true_ce_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_true_total_loss",true_total_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_true_ce_loss_teacher",true_ce_loss_teacher,prog_bar=True,batch_size=self.batch_size)


        
        self.test_step_prediction.append(output_student)
        self.test_step_prediction_teacher.append(output_teacher)
        self.test_step_label.append(labels)
        self.test_step_input.append(inputs)
        return total_loss


class FeatureBasedOffline(ImageClassifierBase):
    #give params to FeatureBasedOffline if used
    def __init__(self,
                 student_model,
                 teacher_model,
                 lr=0.1,
                 batch_size=32,
                 epochs=150,
                 momentum=0.9,
                 weight_decay=2e-5,
                 norm_weight_decay=0.0,
                 label_smoothing=0.1,
                 lr_scheduler='cosineannealinglr',
                 lr_warmup_epochs=5,
                 lr_warmup_method='linear',
                 lr_warmup_decay=0.01,
                 optimizer_algorithm = 'sgd',
                 num_workers = 4,
                 note = '',
                 resize_size = (224,224),
                 data_split_ratios = [0.8,0.15,0.05],
                 alpha=0.95,
                 beta=0.05
                 ):
        #give to ImageClassifierBase
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            weight_decay=weight_decay,
            norm_weight_decay=norm_weight_decay,
            label_smoothing=label_smoothing,
            lr_scheduler=lr_scheduler,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_warmup_method=lr_warmup_method,
            lr_warmup_decay=lr_warmup_decay,
            optimizer_algorithm = optimizer_algorithm,
            num_workers = num_workers,
            note = note,
            resize_size = resize_size,
            data_split_ratios = data_split_ratios,
            kd_run = True
            )
        #?
        #änderung die hier passieren werden nicht korrekt geloggt!
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.beta = beta
        self.test_step_prediction_teacher = []
        self.name = f'{self.student_model.name}_{self.teacher_model.name}'
        self.kd_name = self.__class__.__name__
        self.save_hyperparameters({"beta":self.beta,"alpha":self.alpha,"name":self.name,"kd_name":self.kd_name})
        
        #freeze teacher model for offline kd
        self.teacher_model.freeze_layers()

                
        #add hyperparameters of each model to the hparams dict of the KD class
        for key in self.student_model.hparams:
            self.save_hyperparameters({f"student_{key}":self.student_model.hparams[key]})
            
        for key in self.teacher_model.hparams:
            self.save_hyperparameters({f"teacher_{key}":self.teacher_model.hparams[key]})
    
    def init_base_model(self):
        pass
    
    def forward(self, x):
        return self.student_model(x)


    def training_step(self, batch,batch_idx):
        inputs, labels = batch
        
        #turn off dropout for teacher model
        self.teacher_model.eval()
        
        
        #only inference for teachers features & outputs
        with torch.no_grad():
            features_teacher, output_teacher = self.teacher_model.model(x=inputs,is_feat=True)

                
        #features & outputs
        features_student, output_student = self.student_model.model(x=inputs,is_feat=True)
        
        
        kd_loss, ce_loss, ce_loss_teacher, total_loss, true_total_loss = offline_feature_based_distillation_loss(student_output=output_student,
                                                                                                                 teacher_output=output_teacher,
                                                                                                                 student_layer_features=features_student[-1],
                                                                                                                 teacher_layer_features=features_teacher[-1],
                                                                                                                 labels=labels,
                                                                                                                 label_smoothing=self.label_smoothing,
                                                                                                                 alpha=self.alpha,
                                                                                                                 beta=self.beta
                                                                                                                 )
        
        
        
        
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)
        
        self.logger.experiment.add_scalars('Accuracy_student',
                                           {'train': accuracy_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_teacher',
                                           {'train': accuracy_teacher},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5_student', 
                                           {'train': accuracy_top_5_student},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Accuracy_top_5_teacher', 
                                           {'train': accuracy_top_5_teacher},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('KD_loss', 
                                           {'train': kd_loss},
                                           global_step=self.current_epoch)
        

        self.logger.experiment.add_scalars('CE_loss', 
                                           {'train': ce_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Total_loss', 
                                           {'train': total_loss},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('All_losses', 
                                           {'total_loss_train': total_loss,
                                            'ce_loss_train': ce_loss,
                                            'kd_loss_train': kd_loss  ,
                                            'true_total_loss_train': true_total_loss,
                                            'ce_loss_teacher_train':ce_loss_teacher,
                                            # 'kd_loss_output_train':kd_loss_output,
                                            },
                                           global_step=self.current_epoch)
        
        
        self.logger.experiment.add_scalars('All_accuracies', 
                                           {'student_train': accuracy_student,
                                            'teacher_train': accuracy_teacher,
                                            },
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('MCC_student', 
                                           {'train': mcc_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('MCC_teacher', 
                                           {'train': mcc_teacher},
                                           global_step=self.current_epoch)
        
        self.log("TA",accuracy_student,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("TAT",accuracy_teacher,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("TL",total_loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)

        return total_loss

    def validation_step(self, batch,batch_idx):
        inputs, labels = batch
        
        #turn off dropout for teacher model
        self.teacher_model.eval()
        
        
        #only inference for teachers features & outputs
        with torch.no_grad():
            features_teacher, output_teacher = self.teacher_model.model(x=inputs,is_feat=True)

                
        #features & outputs
        features_student, output_student = self.student_model.model(x=inputs,is_feat=True)
        
        kd_loss, ce_loss, ce_loss_teacher, total_loss, true_total_loss = offline_feature_based_distillation_loss(student_output=output_student,
                                                                                                                 teacher_output=output_teacher,
                                                                                                                 student_layer_features=features_student[-1],
                                                                                                                 teacher_layer_features=features_teacher[-1],
                                                                                                                 labels=labels,
                                                                                                                 label_smoothing=self.label_smoothing,
                                                                                                                 alpha=self.alpha,
                                                                                                                 beta=self.beta
                                                                                                                 )

        
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)
        
        self.logger.experiment.add_scalars('Accuracy_student',
                                           {'validation': accuracy_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_teacher',
                                           {'validation': accuracy_teacher},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('Accuracy_top_5_student', 
                                           {'validation': accuracy_top_5_student},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Accuracy_top_5_teacher', 
                                           {'validation': accuracy_top_5_teacher},
                                           global_step=self.current_epoch)


        self.logger.experiment.add_scalars('KD_loss', 
                                           {'validation': kd_loss},
                                           global_step=self.current_epoch)
        

        self.logger.experiment.add_scalars('CE_loss', 
                                           {'validation': ce_loss},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('Total_loss', 
                                           {'validation': total_loss},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('All_losses', 
                                           {'total_loss_validation': total_loss,
                                            'ce_loss_validation': ce_loss,
                                            'kd_loss_validation': kd_loss  ,
                                            'true_total_loss_validation': true_total_loss,
                                            'ce_loss_teacher_validation':ce_loss_teacher,
                                            # 'kd_loss_output_validation':kd_loss_output,

                                            },
                                           global_step=self.current_epoch)
        
        
        self.logger.experiment.add_scalars('All_accuracies', 
                                           {'student_validation': accuracy_student,
                                            'teacher_validation': accuracy_teacher,
                                            },
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('MCC_student', 
                                           {'validation': mcc_student},
                                           global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars('MCC_teacher', 
                                           {'validation': mcc_teacher},
                                           global_step=self.current_epoch)
        
        #please ignore, only for ModelCheckpoint(....)
        self.log("TuCL",ce_loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("TuCLT",ce_loss_teacher,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VL",total_loss,on_step=False,prog_bar=True,batch_size=self.batch_size)
        self.log("VAT",accuracy_teacher,on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log("VA",accuracy_student,on_step=False,prog_bar=True,batch_size=self.batch_size)


        return total_loss

    def setup(self, stage):
        #prevents trainer from logging twice, if test step is executed directly after train step
        if self.setup_run:
            #log architecture graph
            self.logger._log_graph = True
            self.logger.log_graph(self,torch.rand((1,3) + self.resize_size).to('cuda'))
            
            self.logger.experiment.add_text('student_model_architecture',str(self.student_model))
            self.logger.experiment.add_text('teacher_model_architecture',str(self.teacher_model))

        
        
    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        
        #turn off dropout for teacher model
        self.teacher_model.eval()
        
        
        #only inference for teachers features & outputs
        with torch.no_grad():
            features_teacher, output_teacher = self.teacher_model.model(x=inputs,is_feat=True)

                
        #features & outputs
        features_student, output_student = self.student_model.model(x=inputs,is_feat=True)
        
        kd_loss, ce_loss, ce_loss_teacher, total_loss, true_total_loss = offline_feature_based_distillation_loss(student_output=output_student,
                                                                                                                 teacher_output=output_teacher,
                                                                                                                 student_layer_features=features_student[-1],
                                                                                                                 teacher_layer_features=features_teacher[-1],
                                                                                                                 labels=labels,
                                                                                                                 label_smoothing=self.label_smoothing,
                                                                                                                 alpha=self.alpha,
                                                                                                                 beta=self.beta
                                                                                                                 )

                
        #metrics
        accuracy_student = self.accuracy(output_student,labels)
        accuracy_teacher = self.accuracy(output_teacher,labels)
        
        accuracy_top_5_student = self.accuracy_top_5(output_student,labels)
        accuracy_top_5_teacher = self.accuracy_top_5(output_teacher,labels)

        
        mcc_student = self.mcc(output_student,labels)
        mcc_teacher = self.mcc(output_teacher,labels)

        #predictions = torch.argmax(output,dim=1)
        
        self.log("test_accuracy_student",accuracy_student,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_top_5_student",accuracy_top_5_student,prog_bar=False,batch_size=self.batch_size)
        self.log("test_accuracy_top_5_teacher",accuracy_top_5_teacher,prog_bar=False,batch_size=self.batch_size)

        self.log("test_mcc_student",mcc_student,prog_bar=True,batch_size=self.batch_size)
        self.log("test_accuracy_teacher",accuracy_teacher,prog_bar=True,batch_size=self.batch_size)
        self.log("test_mcc_teacher",mcc_teacher,prog_bar=True,batch_size=self.batch_size)
        
        self.log("test_kd_loss",kd_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_ce_loss",ce_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_ce_loss_teacher",ce_loss_teacher,prog_bar=True,batch_size=self.batch_size)
        self.log("test_total_loss",total_loss,prog_bar=True,batch_size=self.batch_size)
        self.log("test_true_total_loss",true_total_loss,prog_bar=True,batch_size=self.batch_size)

        
        self.test_step_prediction.append(output_student)
        self.test_step_prediction_teacher.append(output_teacher)
        self.test_step_label.append(labels)
        self.test_step_input.append(inputs)
        return total_loss




#################################################################################Models##################################################################
class NaiveClassifier(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
    def init_base_model(self):
        return Naive()


class Naive(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5,2)
        self.conv3 = nn.Conv2d(64, 32, 5,2)
        self.conv4 = nn.Conv2d(32, 32, 5,2)
        self.conv5 = nn.Conv2d(32, 32, 5,2)

        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Resnet_18(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
            
    def init_base_model(self):
        return resnet18(weights=None,num_classes=NUM_CLASSES)

class Resnet_18_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            fc_layer)
        
    def init_base_model(self):
        return resnet18(weights=None,num_classes=NUM_CLASSES)

class Pre_Resnet_18_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        
        #get original in_features
        in_features = self.model.fc.in_features
        
        #modify fc layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            nn.Linear(in_features=in_features, out_features=NUM_CLASSES))        

    def init_base_model(self):
        return resnet18(weights='IMAGENET1K_V1')


    
class Resnet_18_Full_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                full_dropout = 0.5,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        self.dropout = dropout
        self.full_dropout = full_dropout
        self.save_hyperparameters({"dropout":self.dropout, "full_dropout":self.full_dropout}) #evtl bei KD wird nicht seperat von beiden dropout modellen korrekte dropout parameter erfasst?!
        
        #add dropoutlayer to convlayers after each activationfunction
        relu_layer = self.model.relu
        self.model.relu = nn.Sequential(
            relu_layer,
            nn.Dropout2d(p=self.full_dropout), #nn.Dropout2d when used in conv layers, no inplace in convlayers
            )
        #add dropoutlayer before fc layer
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True), #nn.Dropout otherwise
            fc_layer)
        
    def init_base_model(self):
        return resnet18(weights=None,num_classes=NUM_CLASSES)

class Resnet_34(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
            
    def init_base_model(self):
        return resnet34(weights=None,num_classes=NUM_CLASSES)

class Resnet_34_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            fc_layer)
        
    def init_base_model(self):
        return resnet34(weights=None,num_classes=NUM_CLASSES)
    
class Pre_Resnet_34_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        
        #get original in_features
        in_features = self.model.fc.in_features
        
        #modify fc layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            nn.Linear(in_features=in_features, out_features=NUM_CLASSES))        

    def init_base_model(self):
        return resnet34(weights='IMAGENET1K_V1')

    
class Resnet_34_Full_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                full_dropout = 0.5
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        self.dropout = dropout
        self.full_dropout = full_dropout
        self.save_hyperparameters({"dropout":self.dropout, "full_dropout":self.full_dropout}) #evtl bei KD wird nicht seperat von beiden dropout modellen korrekte dropout parameter erfasst?!
        
        #add dropoutlayer to convlayers after each activationfunction
        relu_layer = self.model.relu
        self.model.relu = nn.Sequential(
            relu_layer,
            nn.Dropout2d(p=self.full_dropout), #nn.Dropout2d when used in conv layers, no inplace in convlayers
            )
        #add dropoutlayer before fc layer
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True), #nn.Dropout otherwise
            fc_layer)
        
        
    def init_base_model(self):
        return resnet34(weights=None,num_classes=NUM_CLASSES)

class Resnet_50(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
    def init_base_model(self):
        return resnet50(weights=None,num_classes=NUM_CLASSES)

class Resnet_50_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            fc_layer)
        
    def init_base_model(self):
        return resnet50(weights=None,num_classes=NUM_CLASSES)


class Pre_Resnet_50_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        
        #get original in_features
        in_features = self.model.fc.in_features
        
        #modify fc layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            nn.Linear(in_features=in_features, out_features=NUM_CLASSES))        

    def init_base_model(self):
        return resnet50(weights='IMAGENET1K_V1')



    
class Resnet_50_Full_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                full_dropout = 0.5
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        self.dropout = dropout
        self.full_dropout = full_dropout
        self.save_hyperparameters({"dropout":self.dropout, "full_dropout":self.full_dropout}) #evtl bei KD wird nicht seperat von beiden dropout modellen korrekte dropout parameter erfasst?!
        
        #add dropoutlayer to convlayers after each activationfunction
        relu_layer = self.model.relu
        self.model.relu = nn.Sequential(
            relu_layer,
            nn.Dropout2d(p=self.full_dropout), #nn.Dropout2d when used in conv layers, no inplace in convlayers
            )
        #add dropoutlayer before fc layer
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True), #nn.Dropout otherwise
            fc_layer)
        
    def init_base_model(self):
        return resnet50(weights=None,num_classes=NUM_CLASSES)

class Resnet_101(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
            
    def init_base_model(self):
        return resnet101(weights=None,num_classes=NUM_CLASSES)

class Resnet_101_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            fc_layer)
        
    def init_base_model(self):
        return resnet101(weights=None,num_classes=NUM_CLASSES)


class Pre_Resnet_101_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        
        #get original in_features
        in_features = self.model.fc.in_features
        
        #modify fc layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            nn.Linear(in_features=in_features, out_features=NUM_CLASSES))        

    def init_base_model(self):
        return resnet101(weights='IMAGENET1K_V1')




    
class Resnet_101_Full_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                full_dropout = 0.5
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        self.dropout = dropout
        self.full_dropout = full_dropout
        self.save_hyperparameters({"dropout":self.dropout, "full_dropout":self.full_dropout}) #evtl bei KD wird nicht seperat von beiden dropout modellen korrekte dropout parameter erfasst?!
        
        #add dropoutlayer to convlayers after each activationfunction
        relu_layer = self.model.relu
        self.model.relu = nn.Sequential(
            relu_layer,
            nn.Dropout2d(p=self.full_dropout), #nn.Dropout2d when used in conv layers, no inplace in convlayers
            )
        #add dropoutlayer before fc layer
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True), #nn.Dropout otherwise
            fc_layer)
        
    def init_base_model(self):
        return resnet101(weights=None,num_classes=NUM_CLASSES)


class Resnet_152(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
            
    def init_base_model(self):
        return resnet152(weights=None,num_classes=NUM_CLASSES)


class Resnet_152_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            fc_layer)

    def init_base_model(self):
        return resnet152(weights=None,num_classes=NUM_CLASSES)


class Pre_Resnet_152_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.save_hyperparameters({"dropout":self.dropout})
        
        #get original in_features
        in_features = self.model.fc.in_features
        
        #modify fc layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True),
            nn.Linear(in_features=in_features, out_features=NUM_CLASSES))        

    def init_base_model(self):
        return resnet152(weights='IMAGENET1K_V1')


    
class Resnet_152_Full_Dropout(ImageClassifierBase):
    def __init__(self,
                lr=0.1,
                batch_size=32,
                epochs=150,
                momentum=0.9,
                weight_decay=0.00002,
                norm_weight_decay=0.0,
                label_smoothing=0.1,
                lr_scheduler='cosineannealinglr',
                lr_warmup_epochs=5,
                lr_warmup_method='linear',
                lr_warmup_decay=0.01,
                optimizer_algorithm='sgd',
                num_workers = 4,
                note = '',
                resize_size = (224,224),
                data_split_ratios = [0.8,0.15,0.05],
                dropout=0.2,
                full_dropout = 0.5
                ):
        
        super().__init__(
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        norm_weight_decay=norm_weight_decay,
                        label_smoothing=label_smoothing,
                        lr_scheduler=lr_scheduler,
                        lr_warmup_epochs=lr_warmup_epochs,
                        lr_warmup_method=lr_warmup_method,
                        lr_warmup_decay=lr_warmup_decay,
                        optimizer_algorithm=optimizer_algorithm,
                        num_workers = num_workers,
                        note = note,
                        resize_size = resize_size,
                        data_split_ratios = data_split_ratios,
                         )
        
        self.dropout = dropout
        self.full_dropout = full_dropout
        self.save_hyperparameters({"dropout":self.dropout, "full_dropout":self.full_dropout})
        
        #add dropoutlayer to convlayers after each activationfunction
        relu_layer = self.model.relu
        self.model.relu = nn.Sequential(
            relu_layer,
            nn.Dropout2d(p=self.full_dropout), #nn.Dropout2d when used in conv layers
            )
        #add dropoutlayer before fc layer
        fc_layer = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout,inplace=True), #nn.Dropout otherwise
            fc_layer)
        
    def init_base_model(self):
        return resnet152(weights=None,num_classes=NUM_CLASSES)
    

