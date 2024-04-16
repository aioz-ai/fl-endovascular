import torch
import torch.nn.functional as F
from tqdm import tqdm
from cal_metric import jaccard, calculate_miou
from utils_seg.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    list_pred= []
    list_label = []
    # iterate over the validation set
    device =torch.device(device)
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                tmp_true = mask_true.squeeze()
                tmp_pred = mask_pred.argmax(dim=1).squeeze()

                if len(tmp_true.shape)==2 and len(tmp_pred.shape) ==2 : 
                    tmp_true = tmp_true.unsqueeze(0) 
                    tmp_pred = tmp_pred.unsqueeze(0) 

                list_pred.append(tmp_true.cpu())
                list_label.append(tmp_pred.cpu())

                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                


    net.train()
    list_pred = torch.cat(list_pred, dim=0)
    list_label = torch.cat(list_label, dim=0)
    j = jaccard(list_pred, list_label)
    miou = calculate_miou(list_pred, list_label)
    return dice_score / max(num_val_batches, 1), j, miou