import argparse
import logging

from pathlib import Path

import torch
import torch.nn.functional as F

from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from model.unet_model import UNet

# dir_img = Path('datasets/2d/cell/train/image')
# dir_mask = Path('datasets/2d/cell/train/label')
dir_img = Path('/mnt/data/Datasets/crack_segmentation_dataset/train/images')
dir_mask = Path('/mnt/data/Datasets/crack_segmentation_dataset/train/masks')
dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              classes_label=[0, 255],
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, classes_label, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    global_step = 0
    val_scores = []
    # 5. Begin training
    with tqdm(total=epochs, desc='Epoch:', unit='item') as pbar:
        for epoch in range(1, epochs + 1):
            net.train()
            epoch_loss = 0
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device)
                true_masks = true_masks.to(device=device)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(F.softmax(masks_pred, dim=1), true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1), true_masks, multiclass=True)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                epoch_loss += loss.item()

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(net, val_loader, device)
                        logging.info('Current validation Dice score: {}, Best validation Dice score:{}'.
                                     format(val_score, max(val_scores) if len(val_scores) > 0 else ''))
                        if len(val_scores) == 0 or val_score > max(val_scores):
                            val_scores.append(val_score)
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            torch.save(net.state_dict(), str(dir_checkpoint / 'best_checkpoint.pth'))
                            logging.info(f'Best checkpoint saved!')

            pbar.update()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-2,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes_label', '-c', type=list, default=[0, 255], help='classes label value')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=len(args.classes_label), bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  classes_label=args.classes_label,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
