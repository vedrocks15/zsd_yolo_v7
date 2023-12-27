"""
Script to generate image embeddings using CLIP model for a set of image crops.
And also generating additional labels for self-labelling mechanism as mentioned in the zsd_yolo paper.
This should be run on the machine where the base open world model would be trained
so that target & source embeddings are computed on the same GPU platform.

The paths in the code are mainly for the YOLO format dataset.

Changelog :

0.0.1 : Version 1
"""

# Basic Imports
import cv2
import numpy as np
import random 
import os
import matplotlib.pyplot as plt
import PIL
from scipy import spatial
import json
import requests
import collections
from tqdm import tqdm
import shutil
import collections
import torch
import argparse
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
from PIL import Image

# Yolo related imports, so that it can be run in a class agnoistic fashion....
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.ops import nms
from tqdm import tqdm
from PIL import Image
from utils.general import xywhn2xyxy, xywh2xyxy, xyxy2xywh, xyxy2xywhn
from torchvision.transforms import Resize
from nltk.corpus import wordnet
from models.experimental import attempt_load
from models.yolo import Model
from utils.general import non_max_suppression
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume


# code reproducability 
seed = 100
random.seed(seed)
np.random.seed(seed)

# defining pseudo argparser
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# for zsd version of data loader...
opt = Namespace(single_cls=False, 
                zsd=False)


# base paths (specify path with respect to server)
base_yolo_dir = "./yolo_coco_dataset_65_15/"

# fetching GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Base model instantiation CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# laoding model on GPU (Global Variable....)
clip_model = clip_model.to(device)

# Hyper-param loading..
with open('data/hyp.scratch.p5.yaml') as f:
    hyp = yaml.safe_load(f)


# Helper functions
# Loading model weights...
def load_model(model_path, 
              hyp, 
              device='cuda'):

    ckpt = torch.load(model_path, map_location=device)
    model = Model(ckpt['model'].yaml, ch=3, anchors=hyp.get('anchors'), hyp=hyp).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), model_path))
    return model

# clip inference function...
def extract_image_embeddings(images, 
                             boxes):

    all_embeddings = []

    # Looping all labeled boxes list associated with each image...
    for i in range(len(boxes)):
        bboxes = deepcopy(boxes[i]).type(torch.IntTensor)
        regions = []
        include = []

        # crop & extract bounding boxes....
        for j in range(len(bboxes)):
            x1, y1, x2, y2 = [int(k) for k in bboxes[j]]
            # Hugging face handles all required pre-processing...
            regions.append(images[i][:, y1:y2, x1:x2].clone().detach().float())

        if len(regions):
            # preprocessing them for CLIP....
            crop_batch = clip_processor(text=["Random text"], 
                                        images=regions, 
                                        return_tensors="pt", 
                                        padding=True)

            # CLIP inference.....                
            crop_batch = crop_batch.to(device)
            crop_op = clip_model(**crop_batch)
            img_crop_embeddings = crop_op.image_embeds
            all_embeddings.append(img_crop_embeddings)
        else:
            all_embeddings.append(torch.zeros((0, 512)).cuda())
        
    return all_embeddings

# annotation saver....
def save_annot_torch(annot, 
                    data, 
                    out_path):

    paths = [os.path.join(out_path, i.split('/')[-1].split('.')[0] + '.pt') for i in data[2]]

    # saving extracted embeddings....
    for i in range(len(paths)):
        torch.save(annot[i].cpu(), paths[i])

def generate_zsd_data(path, 
                      hyp, 
                      opt, 
                      out_path, 
                      imgsz=640, 
                      batch_size=16, 
                      model_path=None, 
                      score_thresh=0.1, 
                      iou_thresh=0.1, 
                      loader=None, 
                      min_w=0, 
                      min_h=0, 
                      delete=False, 
                      test=False, 
                      remove_tiny=True):

    # Re-create vector directory....
    if os.path.exists(out_path) and delete:
        shutil.rmtree(out_path)
        os.mkdir(out_path)
    
    # loading baseline yolo for the purpose of self labelling....
    model = load_model(model_path, hyp).eval() if model_path else None
    # disable model training phase....
    model.eval() 
    gs = max(int(model.stride.max()), 32)  if model else 32
    
    # check if there is any loader or not.....
    loader, _ = (loader, None) if loader else create_dataloader(path, imgsz, batch_size, gs, opt, hyp=hyp, workers=4)
    
    # self labelling loop....
    removed_boxes, total_boxes, self_label_boxes = 0, 0, 0
    pbar = tqdm(loader, total=len(loader))
    for data in pbar:
        # image batches....
        c_batch_size = len(data[0])
        
        # holds number of labels per image in the batch....
        count_per_batch = [0, ] * c_batch_size
        for i in data[1]:
            # i[0] is the image index in the batch....
            count_per_batch[int(i[0])] += 1

        # Splits the tensor into chunks. Each chunk is a view of the original tensor.
        split_boxes = data[1].split(count_per_batch)
        for i in range(len(split_boxes)):
            # converting the normalised xy-wh to xyxy format.... (2,3,4,5) index
            split_boxes[i][:, 2:] = xywhn2xyxy(split_boxes[i][:, 2:])
        
        # final split boxes... (x1,y1,x2,y2, 1.1, class_id for each boxes...)
        # 1.1 is used since gt labels are given more priority....
        split_boxes = [torch.cat([i[..., 2:], 
                                  torch.ones((i.shape[:-1] + (1, ))) + 0.1, 
                                  i[..., 1].unsqueeze(-1)], dim=1).cuda() for i in split_boxes]
    
        # running YOLO inference....
        if model:
            imgs = data[0].to('cuda', non_blocking=True).float() / 255
            with torch.no_grad():
                output = model(imgs)

            # since model is in eval mode, loop over all the outputs in the 0th index.....
            for idx, i in enumerate(output[0]):
                i[:, :4] = xywh2xyxy(i[:, :4])
                # setting class id to -1
                i[:, 5] = -1
            
            # running a class agnostic inference on the output...[x1,y1,x2,y2,prob,class_id]
            # concating the extra detected boxes with each of the original prediction....
            all_boxes = [torch.cat([output[0][i][:, :6], split_boxes[i]]) for i in range(len(split_boxes))]
            # score threshold for probablity....
            all_boxes = [i[i[:, 4] > score_thresh] for i in all_boxes]
            # nms using the probability scores.... (supressing all useless detections...)
            # gts will have an advantage because their default probability is 1.1 & the model
            # does not predict prob over 1.0
            all_boxes = [i[nms(i[:, :4], i[:, 4], iou_threshold=iou_thresh)] for i in all_boxes]
            all_boxes = [i[i[:, 4] < 1] for i in all_boxes]
            # making detected class as -1....
            all_boxes = [torch.cat([i[..., :-1], torch.zeros(i[...].shape[:-1] + (1, )).cuda() - 1], dim=-1) for i in all_boxes]
            
            # filtering boxes with min_w & min_h (selecting only high confidence boxes...)
            for i in range(len(all_boxes)):
                mask = torch.Tensor([(int(j[2]) - int(j[0])) > min_w and (int(j[3]) - int(j[1])) > min_h for j in all_boxes[i]])
                all_boxes[i] = all_boxes[i][mask.type(torch.BoolTensor)]

            # adding filtered boxes to the og labels...
            split_boxes = [torch.cat([split_boxes[i], all_boxes[i]]) for i in range(len(split_boxes))]
        
        # more filtering... (data[3] is the shape of input image)
        # this filtering ensures that boxes don't overflow..... [x1,y1,x2,y2] format
        for i in range(len(split_boxes)):
            split_boxes[i][:, 0] = torch.clip(split_boxes[i][:, 0], 
                                              min=data[3][i][1][1][0], 
                                              max=data[3][i][0][1] * data[3][i][1][0][1] + data[3][i][1][1][0])

            split_boxes[i][:, 1] = torch.clip(split_boxes[i][:, 1], 
                                              min=data[3][i][1][1][1], 
                                              max=data[3][i][0][0] * data[3][i][1][0][0] + data[3][i][1][1][1])

            split_boxes[i][:, 2] = torch.clip(split_boxes[i][:, 2], 
                                              min=data[3][i][1][1][0], 
                                              max=data[3][i][0][1] * data[3][i][1][0][1] + data[3][i][1][1][0])

            split_boxes[i][:, 3] = torch.clip(split_boxes[i][:, 3], 
                                              min=data[3][i][1][1][1],
                                              max=data[3][i][0][0] * data[3][i][1][0][0] + data[3][i][1][1][1])

            # removing tiny boxes to avoid clip inference issue....
            if remove_tiny:
                mask = torch.Tensor([(((j[2] - j[0]) > 1) and ((j[3] - j[1]) > 1)) for j in split_boxes[i]])
                previous_len = len(split_boxes[i])
                split_boxes[i] = split_boxes[i][mask.type(torch.BoolTensor)]
                removed_boxes += previous_len - len(split_boxes[i])

        # getting image embeddings for each image... [image box details, image vector embedding]
        embeddings = [torch.zeros((i.shape[0], 512)) for i in split_boxes] if test else extract_image_embeddings(data[0], [i[:, :4] for i in split_boxes])
        # converting back to proper label format.....
        for i in range(len(split_boxes)):
            split_boxes[i][:, :4] = xyxy2xywhn(split_boxes[i][:, :4], 
                                              w=data[3][i][0][1] * data[3][i][1][0][1], 
                                              h=data[3][i][0][0] * data[3][i][1][0][0], 
                                              padw=data[3][i][1][1][0], 
                                              padh=data[3][i][1][1][1])
        
        # final save.....
        annot = [torch.cat([split_boxes[i][:, 5].unsqueeze(-1), 
                            split_boxes[i][:, :4], embeddings[i]], dim=1).cpu() for i in range(len(split_boxes))]

        save_annot_torch(annot, data, out_path)
        total_boxes += sum(len(i) for i in split_boxes)
        for i in annot:
            self_label_boxes += sum(i[:, 0] == -1)
            
        pbar.desc = f'Total removed boxes: {removed_boxes}. Total generated boxes: {total_boxes}. Self-label boxes: {self_label_boxes}. Generating Embeddings to {out_path}.1'


# Train data split.....
loader, _ = create_dataloader(path = '/home/AD/vejoshi/yolo_open_world/yolo_coco_dataset_65_15/train/images/',
                              imgsz = 640, 
                              batch_size = 16, 
                              stride = 32, 
                              opt = opt, 
                              hyp = hyp, 
                              workers=8, 
                              augment=False, 
                              cache=False,
                              annot_folder='labels')

# Function to perform self labelling & generate CLIP vectors for each crop.....
generate_zsd_data(path = '/home/AD/vejoshi/yolo_open_world/yolo_coco_dataset_65_15/train/images/', 
                  hyp = hyp, 
                  opt = opt,  
                  out_path ='/home/AD/vejoshi/yolo_open_world/yolo_coco_dataset_65_15/train/label_clip_vectors/', 
                  model_path = '/home/AD/vejoshi/yolo_open_world/yolov7/runs/train/yolov7_normal_unseen_supervised_weights/weights/best.pt',
                  loader = loader, 
                  min_w = 25, 
                  min_h = 25, 
                  iou_thresh=0.2, 
                  score_thresh=0.3, 
                  delete=True, 
                  test=False)
