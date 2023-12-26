"""
Script to generate text embeddings for building ZSD yolo
This should be run on the machine where the base open world model would be trained
so that target & source embeddings are computed on the same GPU platform.

The paths in the code are mainly for the YOLO format dataset.

Changelog :

0.0.1 : Version 1
"""

# Basic Imports
import numpy as np
import random 
import os
from scipy import spatial
import json
import requests
import collections
from tqdm import tqdm
import shutil
import collections
import torch
import logging
import math
import json
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
from importlib import reload

# CLIP imports (instead of Torch we use the hugging face repo for clip)
from transformers import CLIPProcessor, CLIPModel
from torchvision import models
import pickle
import yaml
from nltk.corpus import wordnet


# code reproducability 
seed = 100
random.seed(seed)
np.random.seed(seed)



# loading data config file....(order is very important to maintain)
with open('data/zsd_coco_65.yaml') as f:
    meta = yaml.safe_load(f)

# Loading names of classes.....
unseen_names = [i for i in meta['unseen_class']]
seen_names = [i for i in meta['seen_class']]
all_names = seen_names + unseen_names

# loading word definitions for better text embeddings...
defs = {i: wordnet.synsets(i)[0].definition() if len(wordnet.synsets(i)) else '' for i in all_names}

# giving more context to certain names for better embedding generation....
unseen_names[10] = 'hotdog'
unseen_names[12] = 'computer mouse'

# Better text embeddings for inference....
defs_and_all_names = [i + ', ' + defs[i] + ',' if defs.get(i) else i for i in all_names]

# normal templates
templates = ['a photo of {} in the scene']

# loading & setting up clip model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Base model instantiation CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# laoding model on GPU (Global Variable....)
clip_model = clip_model.to(device)

# helper function
def zeroshot_classifier(classnames, 
                        templates):
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            temp_texts = [template.format(classname) for template in templates] #format with class
            text_batch = clip_processor(text = temp_texts, 
                                        images = [np.ones((225,225,3))*30], 
                                        return_tensors="pt", 
                                        padding=True)
            text_batch.to(device)                          
            op = clip_model(**text_batch)
            class_embeddings = op.text_embeds
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        print("Final shape : ",zeroshot_weights.shape)
    return zeroshot_weights


# for better inference....
all_text_embeddings = zeroshot_classifier(defs_and_all_names, templates)
torch.save(all_text_embeddings.T, '/home/AD/vejoshi/yolo_open_world/yolo_coco_dataset_65_15/all_coco_text_embeddings_65_15_zsd.pt')

# for seen classes...
seen_text_embeddings = zeroshot_classifier(seen_names, templates)
torch.save(seen_text_embeddings.T, '/home/AD/vejoshi/yolo_open_world/yolo_coco_dataset_65_15/seen_coco_text_embeddings_65_15.pt')

# for unseen classes....
unseen_text_embeddings = zeroshot_classifier(unseen_names, templates)
torch.save(unseen_text_embeddings.T, '/home/AD/vejoshi/yolo_open_world/yolo_coco_dataset_65_15/unseen_coco_text_embeddings_65_15.pt')

# simple name inference.....
all_text_embeddings = zeroshot_classifier(all_names, templates)
torch.save(all_text_embeddings.T, '/home/AD/vejoshi/yolo_open_world/yolo_coco_dataset_65_15/simple_all_coco_text_embeddings_65_15_zsd.pt')





