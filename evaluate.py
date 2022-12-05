import torch
import torch.nn.functional as F

from utils.dice_score import multiclass_dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in dataloader:
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device)
        mask_true = mask_true.to(device=device)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            # mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(F.softmax(mask_pred, dim=1), mask_true)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score

    return dice_score / num_val_batches
