# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:34:07 2023

@author: leonidas
"""

import torch
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import importlib
from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar
import math
import numpy as np

#process_function_in_batches to tackle memory constraints
def process_function_in_batches(images, batch_size, func, targets = None):
    num_images = images.shape[0]
    num_batches = math.ceil(num_images/batch_size)
    processed_batches = []
    for i in range(num_batches):            
        # print(f"batch {i} of {num_batches}")
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, num_images)
        batch = images[start_index:end_index]
        if targets is not None:
            target_batch = targets[start_index:end_index]
            processed_batch = func(batch,target=target_batch)
        else:
            
            processed_batch = func(batch)
        processed_batches.append(processed_batch)
        
    # print(type(processed_batch))
    
    if isinstance(processed_batch, np.ndarray):
        result = np.concatenate(processed_batches, axis=0)
        
    if isinstance(processed_batch, torch.Tensor):
        result = torch.cat(processed_batches, dim=0)
        
    if isinstance(processed_batch, tuple):
        first_elements, snd_elements = zip(*processed_batches)
        result = (np.concatenate(first_elements), torch.cat(snd_elements))
        
    # print("hi")
    return result


#log xai metrics
def add_to_nested_dict(dictionary:dict, outer_key, mid_key, inner_key, value):
    """
    Adds values to nested dict

    Args:
    - dictionary: dictionary that gets filled
    - outer_key: first key with library_camname
    - mid_key: middle key with xai_metric_name
    - inner_key: classID
    - value: Image Score
    """
    
    dictionary.setdefault(outer_key,{})
    dictionary[outer_key].setdefault(mid_key,{})
    dictionary[outer_key][mid_key].setdefault(inner_key,[]).append(value)


#https://github.com/pytorch/vision/blob/main/references/classification/utils.py
def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups

#plot roc curve
def get_roc_curve_figure(fpr, tpr, thresholds,step_size=125,font_size=8,classid=0):
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve') #marker='o',markersize=3
    plt.plot([0, 1], [0, 1], color='navy', lw=2, label='No skill', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')  
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for class {classid}')
    plt.legend(loc='lower right')
    result = plt.gcf()
    # plt.show()
    plt.close()
    return result

#plot confusion matrix
def get_confusion_matrix_figure(computed_confusion):
    df_cm = pd.DataFrame(computed_confusion)
    plt.figure(figsize=(20,20))

    #colors
    cmap = sns.color_palette("light:#Ff4100", as_cmap=True)

    fig = sns.heatmap(df_cm,annot=True,cmap=cmap,annot_kws={'fontsize':3}).get_figure()
    # plt.show()
    plt.close(fig)
    return fig


def get_k_random_values(tensor, k,device=None):
    """
    Get k random values from a PyTorch tensor along with their indices.

    Args:
    - tensor (torch.Tensor): Input tensor.
    - k (int): Number of random values to select
    - device (str): A torch device 

    Returns:
    - values (torch.Tensor): Tensor containing the selected random values.
    - indices (torch.Tensor): Tensor containing the indices of the selected values.
    """
    torch.manual_seed(42)

    # Check if k is greater than the number of elements in the tensor
    if k > tensor.numel():
        raise ValueError("k cannot be greater than the number of elements in the tensor.")

    # Flatten the tensor to 1D
    flattened_tensor = tensor.view(-1)

    # Generate random indices
    random_indices = torch.randperm(flattened_tensor.numel(),device=device)[:k]

    # Use topk to get the values and indices
    values, indices = torch.topk(flattened_tensor[random_indices], k)

    return values, random_indices[indices]

#load correct class object for .ckpt file
def get_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_name = checkpoint['hyper_parameters']['name']
    del checkpoint
    class_object = get_class("models",model_name)
    return class_object.load_from_checkpoint(checkpoint_path)

#helper to load correct class
def get_class(module_name, class_name):
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)

        # Get the class object from the module
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # Handle import or attribute errors
        raise ValueError(f"Error finding class '{class_name}' from module '{module_name}': {e}")


#switch off tqdm progressbar
class LitProgressBar(TQDMProgressBar):
    
    def init_train_tqdm(self):
        bar = tqdm(
            disable=False,
            
        )
        return bar    
        
    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
            
        )
        #calculate estimated runtime
        elapsed_secs = self.train_progress_bar.format_dict['elapsed']
        hours_left = round(elapsed_secs * (self.trainer.max_epochs-self.trainer.current_epoch) / 60 /60,1)
        
        print(" ETL:", str(hours_left) + "h")
        
        return bar

    def init_test_tqdm(self):
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            disable=True,
            
        )
        return bar
    
    def init_sanity_tqdm(self):
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            disable=True,
            
        )
        return bar
    
    def init_predict_tqdm(self):
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            disable=True,
            
        )
        return bar
