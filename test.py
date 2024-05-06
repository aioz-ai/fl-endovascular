import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from cal_metric import jaccard,dice, calculate_miou
from evaluate import evaluate
from segmentation_unet import UNet
from med_dataloader import MedDataset
from utils_seg.dice_score import dice_loss, multiclass_dice_coeff, count_f
from evaluate import evaluate
from utils import compute_miou_segmentation

def test_model(model, image, device): 
    image = image.to(device)
    model.eval()  
    out = model(image) 
    return out


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    # parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    # parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
    #                    help='Learning rate', dest='lr')
    # parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
    #                     help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--ckpt_dir', type=str, required=False, default="", help="Checkpoint Directory")
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')

    return parser.parse_args()

def main():
    args = get_args()

    #logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    global_model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    #model = model.to(memory_format=torch.channels_last)
    dir_checkpoint = args.ckpt_dir
    test_dir_img = args.datadir
    test_dir_mask = test_dir_img.replace("images","masks")


    ckpt = torch.load(dir_checkpoint)
    #del ckpt['mask_values']

    # load global model
    global_model.load_state_dict(ckpt)
    global_model.to(device=device)

    val_set = MedDataset(images_dir=test_dir_img, mask_dir=test_dir_mask,domain='animal')
    val_loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)
    test_miou = compute_miou_segmentation(global_model, val_loader, device=device)

    print("mIoU of test data: ", test_miou)



if __name__ == "__main__":
    main()

