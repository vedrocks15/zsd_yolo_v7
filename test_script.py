# Basic Imports......
import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test_zsd
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA, ComputeZSDLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)


## global constants
imgsz = 640
batch_size = 16
conf_thres=0.001
iou_thres=0.6
device = torch.device("cuda")
half = False
augment = False


def load_model(weights, hyp, device='cuda', nc=80):
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), hyp=hyp).cuda()
    #exclude = ['anchor'] if (hyp.get('anchors')) else []  # exclude keys
    exclude = []
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    return model


base_dir_path = ""
# loading hyper-parameters...
hyp_script = "/home/AD/vejoshi/yolo_open_world/zsd_yolo_v7/runs/train/" + base_dir_path + "/hyp.yaml"
opt_script = "/home/AD/vejoshi/yolo_open_world/zsd_yolo_v7/runs/train/" + base_dir_path + "/opt.yaml"


with open(hyp_script, 'r') as f:
    hyp = yaml.safe_load(f)

# dummy argparser...
class agp:
    def __init__(self):
        pass
        
with open(opt_script, 'r') as f:
    opt_tmp = yaml.safe_load(f)
    opt = agp()
    for k, v in opt_tmp.items():
        setattr(opt,k,v)
    
weights = "/home/AD/vejoshi/yolo_open_world/zsd_yolo_v7/runs/train/" + base_dir_path + "/weights/best.pt"
#model = attempt_load(weights, map_location=device)
model = load_model(weights, hyp, device=device)

# downsampling factor......
gs = max(int(model.stride.max()), 32)  
imgsz = check_img_size(imgsz, s=gs) 

# Switching the model to evaluation mode to change similarity matrix labels.....
model.eval()
print("Training flag : ",model.training)

# text embeddings load....
text_embedding_path = "/home/AD/vejoshi/yolo_open_world/zsd_yolo_v7/embeddings/unseen_coco_text_embeddings_65_15.pt"

# loading data file
with open('/home/AD/vejoshi/yolo_open_world/zsd_yolo_v7/data/zsd_coco_65.yaml') as f:
    data = yaml.safe_load(f)

testloader = create_dataloader("/home/AD/vejoshi/yolo_open_world/coco_zsd_65_15/test/test_2014/", 
                               imgsz, 
                               batch_size, 
                               gs, 
                               opt, 
                               pad=0.5, 
                               rect=False,
                               cache=False,
                               hyp={'do_zsd': opt.zsd},
                               prefix=colorstr(f'{"valid"}: '), 
                               annot_folder="label_clip_vectors_test_zsd")[0]


opt.no_zsd_post = False
opt.nms_then_zsd = False
# Low NMS thresh
opt.iou_thres = 0.4

# 2nd zsd filtering critirea....
opt.obj_conf_thresh = 0.1

# limiting max dets for safety...
opt.max_det = 15

# main test function 
results, maps, times = test_zsd.test(data,
                                     batch_size = batch_size * 2,
                                     imgsz = 640,
                                     model = model,
                                     single_cls = opt.single_cls or opt.eval_single,
                                     dataloader = testloader,
                                     save_dir = Path(""),
                                     verbose = opt.verbose,
                                     plots=False,
                                     wandb_logger = None,
                                     compute_loss = None,
                                     is_coco = False,
                                     iou_thres = opt.iou_thres,
                                     opt = opt,
                                     text_embedding_path = text_embedding_path)
