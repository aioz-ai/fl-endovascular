import torchmetrics 
import ignite 


def jaccard(pred,label): 
    jaccard = torchmetrics.JaccardIndex(task = 'multiclass', num_classes = 3, ignore_index=0)
    #dice = torchmetrics.classification.Dice(num_classes=3, average='micro', ignore_index =0)
    return jaccard(pred,label)#, dice(pred, label)
# def dice_l(pred,label): 
#     #jaccard = torchmetrics.JaccardIndex(task = 'multiclass', num_classes = 3, ignore_index=0)
#     dice = torchmetrics.classification.Dice(num_classes=3, average='micro', ignore_index =0)
#     return dice(pred, label)
from collections import OrderedDict

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
from torch.nn import functional as F
# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

def  calculate_miou(pred,label): 
    if len(pred.shape)==4 and len(label.shape)==4: 
        x,y,z,t = pred.shape 
        pred=pred.reshape(x*y,z,t)
        label=label.reshape(x*y,z,t)
    #print(pred.shape, label.shape)
    default_evaluator = Engine(eval_step)
    b,h,w = pred.shape 
    pred = pred.reshape(b,h*w) 
    label = label.reshape(b,h*w) 
    
    pred = F.one_hot(pred, num_classes=3)
    pred = pred.permute(0,2,1) 
    cm = ConfusionMatrix(num_classes=3)
    metric = mIoU(cm, ignore_index=0)
    metric.attach(default_evaluator, 'miou')
    state = default_evaluator.run([[pred, label]])
    iou = state.metrics['miou']
    return iou

def dice(pred,label): 
    default_evaluator = Engine(eval_step)
    b,h,w = pred.shape 
    pred = pred.reshape(b,h*w) 
    label = label.reshape(b,h*w) 
    
    pred = F.one_hot(pred, num_classes=3)
    pred = pred.permute(0,2,1) 
    cm = ConfusionMatrix(num_classes=3)
    metric = DiceCoefficient(cm, ignore_index=0)
    metric.attach(default_evaluator, 'dice')
    state = default_evaluator.run([[pred, label]])
    iou = state.metrics['dice']
    return iou.mean()


if __name__ == '__main__': 
    ani_pred = torch.load("pred_ani73.pth", map_location = 'cpu')
    ani_label = torch.load("label_ani73.pth", map_location = 'cpu')

    pt_pred = torch.load("pred_pt73.pth", map_location = 'cpu')
    pt_label = torch.load("label_pt73.pth", map_location = 'cpu')
    print("Jaccard animal: ", jaccard(ani_pred, ani_label))
    print("mIoU: animal ", calculate_miou(ani_pred, ani_label)[0])
    print("Dice: animal ", calculate_miou(ani_pred, ani_label)[1])


    print("Jaccard pt: ", jaccard(pt_pred, pt_label))
    print("mIoU pt: ", calculate_miou(pt_pred, pt_label)[0])
    print("Dice pt: ", calculate_miou(pt_pred, pt_label)[1])
