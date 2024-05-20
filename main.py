import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import collections 
import collections.abc
from tqdm import tqdm
from model import *
from utils import *
from segmentation_unet import UNet

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    #parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    #parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    #parser.add_argument('--task', type=str, default='segmentation', help='which task do you want to do on federated learning setup (classification/segmentation)')
    args = parser.parse_args()
    return args


def  init_nets( n_parties):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net = UNet(n_channels=3, n_classes=3, bilinear=False)
        nets[net_i] = net
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    

    return nets, model_meta_data, layer_type

def train_net_segmentation(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    #net = nn.DataParallel(net)
    net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_miou,_ = compute_miou_segmentation(net, train_dataloader, device=device)

    test_miou, _ = compute_miou_segmentation(net, test_dataloader,  device=device)

    logger.info('>> Pre-Training Training mIoU: {}'.format(train_miou))
    logger.info('>> Pre-Training Test mIoU: {}'.format(test_miou))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            masks_pred = net(images)
            loss = criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.softmax(masks_pred, dim=1).float(),
                F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )

            loss.backward()
            optimizer.step()

            try:
                epoch_loss_collector.append(loss.item())
            except: 
                continue

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        print('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:
            train_miou, _ = compute_miou_segmentation(net, train_dataloader, device=device)
            test_miou,  _ = compute_miou_segmentation(net, test_dataloader,  device=device)

            logger.info('>> Training mIoU: %f' % train_miou)
            logger.info('>> Test mIoU: %f' % test_miou)

    train_miou, _ = compute_miou_segmentation(net, train_dataloader, device=device)
    test_miou,  _ = compute_miou_segmentation(net, test_dataloader,  device=device)

    logger.info('>> Training mIoU: %f' % train_miou)
    logger.info('>> Test mIoU: %f' % test_miou)
    net.to('cpu')

    logger.info(' ** Training complete **')

    print('>> Training mIoU: %f' % train_miou)
    print('>> Test mIoU: %f' % test_miou)
    print(' ** Training complete **')

    return train_miou, test_miou


def train_net_fedprox_segmentation(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    net.to(device)
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_miou, _ = compute_miou_segmentation(net, train_dataloader, device=device)
    test_miou, _ = compute_miou_segmentation(net, test_dataloader, device=device)

    logger.info('>> Pre-Training Training mIoU: {}'.format(train_miou))
    logger.info('>> Pre-Training Test mIoU: {}'.format(test_miou))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    global_weight_collector = list(global_net.to(device).parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, batch in enumerate(train_dataloader):
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            

            masks_pred = net(images)
            loss = criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.softmax(masks_pred, dim=1).float(),
                F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()


            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        print('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_miou, _ = compute_miou_segmentation(net, train_dataloader, device=device)
    test_miou, _ = compute_miou_segmentation(net, test_dataloader, device=device)

    logger.info('>> Training mIoU: %f' % train_miou)
    logger.info('>> Test mIoU: %f' % test_miou)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_miou, test_miou




def local_train_net_segmentation(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu"):
    avg_miou = 0.0 #miou
    miou_list = []
    if global_model:
        global_model.to(device)
    if server_c:
        server_c.to(device)
        server_c_collector = list(server_c.to(device).parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        train_imgdir = os.path.join(args.datadir, 'images')
        train_maskdir = os.path.join(args.datadir, 'masks')
        test_imgdir = os.path.join(args.datadir.replace("train", 'test'), 'images')
        test_maskdir = os.path.join(args.datadir.replace("train", 'test'), 'masks')

        train_dl_local, test_dl_local, _, _ = get_segmentation_dataloader(train_imgdir,train_maskdir, test_imgdir, test_maskdir, batch_size=8, dataidx= dataidxs)
        train_dl_global, test_dl_global, _, _ = get_segmentation_dataloader(train_imgdir,train_maskdir, test_imgdir, test_maskdir, batch_size=8, dataidx=None)
        n_epoch = args.epochs



        if args.alg == 'fedavg':
            train_miou, test_miou = train_net_segmentation(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg == 'fedprox':
            train_miou, test_miou = train_net_fedprox_segmentation(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        
        logger.info("net %d final test mIoU %f" % (net_id, test_miou))
        avg_miou += test_miou
        miou_list.append(test_miou)
    avg_miou /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test mIoU %f" % avg_miou)
        logger.info("std acc %f" % np.std(miou_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    args = get_args()
    device = args.device
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    #device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


    logger.info("Partitioning data")
    print("Partitioning data")
    net_dataidx_map = segmentation_partition_data(args.datadir, args.n_parties)
    
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)


    train_image_dir = os.path.join(args.datadir, 'images')
    train_mask_dir = os.path.join(args.datadir, 'masks')
    test_image_dir = os.path.join(args.datadir.replace("train", 'test'), 'images')
    test_mask_dir = os.path.join(args.datadir.replace("train", 'test'), 'masks')
    train_dl_global, test_dl, train_ds_global, test_ds_global = get_segmentation_dataloader(train_image_dir, train_mask_dir, test_image_dir, test_mask_dir, batch_size=8)


    logger.info("Initializing nets")
    print("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets( args.n_parties)

    global_models, global_model_meta_data, global_layer_type = init_nets(1)
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    if args.alg== 'fedavg': 
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            local_train_net_segmentation(nets_this_round, args, net_dataidx_map, train_dl=train_dl_global, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            print('global n_test: %d' % len(test_dl))
            global_model.to(device)
            train_miou, train_loss = compute_miou_segmentation(global_model, train_dl_global, device=device)
            test_miou , _ = compute_miou_segmentation(global_model, test_dl, device=device)

            logger.info('>> Global Model Train mIoU: %f' % train_miou)
            logger.info('>> Global Model Test mIoU: %f' % test_miou)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            print('>> Global Model Train mIoU: %f' % train_miou)
            print('>> Global Model Test mIoU: %f' % test_miou)
            print('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    
    elif args.alg == 'fedprox':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net_segmentation(nets_this_round, args, net_dataidx_map, train_dl=train_dl_global,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.to(device)
            train_miou, train_loss = compute_miou_segmentation(global_model, train_dl_global, device=device)
            test_miou,  _ = compute_miou_segmentation(global_model, test_dl, device=device)

            logger.info('>> Global Model Train mIoU: %f' % train_miou)
            logger.info('>> Global Model Test mIoU: %f' % test_miou)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')

    
