import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom
from med_dataloader import MedDataset
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from torch import Tensor
from evaluate import evaluate
from torch.utils.data import DataLoader
from utils_seg.dice_score import dice_coeff, multiclass_dice_coeff, dice_loss


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_med_data(datadir): 
    pass

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(os.path.join(datadir, 'train'), transform=transform)
    xray_test_ds = ImageFolder_custom(os.path.join(datadir, 'val'), transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def segmentation_partition_data(traindir, n_parties):


    list_img_train_id = os.listdir(os.path.join(traindir, "images"))
    l = len(list_img_train_id)
    indices = random.sample(range(1, l-1), n_parties-1)
    indices.sort() 
    net_dataidx_map = {}
    np.random.shuffle(list_img_train_id)
    list_data_each_party = np.split(list_img_train_id, indices)
    for i in range(n_parties):
        # assign data to each party
        net_dataidx_map[i] = list_data_each_party[i]
    return net_dataidx_map
    

    


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel

def compute_miou_segmentation(model, dataloader, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True


    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device:
        criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []
    epoch_loss = 0 

    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx,batch in tqdm(enumerate(loader)):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        images, true_masks = batch['image'], batch['mask']
                        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                        true_masks = true_masks.to(device=device, dtype=torch.long)
                        #x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                    #_, _, out = model(x)
                    masks_pred = model(images)  
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True)
                    
                    epoch_loss += loss.item()
                    loss_collector.append(loss.item())

                _, _, miou = evaluate(model, dataloader, device, amp=True)

        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    images, true_masks = batch['image'], batch['mask']
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                masks_pred = model(images) 
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True)
                epoch_loss += loss.item()
               

                    #, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                    #total += x.data.size()[0]
                    #correct += (pred_label == target.data).sum().item()
            _, _, miou = evaluate(model, dataloader, device, amp=True)                   
                

                # if device == "cpu":
                #     pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                #     true_labels_list = np.append(true_labels_list, target.data.numpy())
                # else:
                #     pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                #     true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)



    if was_training:
        model.train()


    return miou, avg_loss



def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device:
        criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in tqdm(enumerate(loader)):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                    _, _, out = model(x)
                    if len(target)==1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                _,_,out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device:
        criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.to(device), target.to(dtype=torch.int64).to(device)
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss



def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.to(device)
    return model


def get_segmentation_dataloader(train_imgdir,train_maskdir, test_imgdir, test_maskdir, batch_size=8,dataidx = None  ): 
    domain = "phantom" # one of these: phantom/animal/sim/real
    train_dts = MedDataset(train_imgdir, train_maskdir, domain = domain, list_img=dataidx)
    test_dts = MedDataset(test_imgdir, test_maskdir, domain = domain, list_img=None)
    train_loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)

    train_loader = DataLoader(train_dts, shuffle=True, **train_loader_args)
    test_loader = DataLoader(test_dts, shuffle=False, drop_last=True, **val_loader_args)
    return train_loader, test_loader, train_dts, test_dts

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])



        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(os.path.join(datadir, "train"), dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(os.path.join(datadir, 'val'), transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


    return train_dl, test_dl, train_ds, test_ds
