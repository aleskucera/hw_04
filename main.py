#!/usr/bin/env python

import argparse
import os
import os.path as osp

import numpy as np
import scipy.sparse
import torch
import torch.utils.data as tdata
from PIL import Image
from tqdm import tqdm

from hw_04 import load_model

BORDER = 10
COLORS_CLAZZ = (
        np.array(
            (
                (128, 128, 128, 100),
                (245, 130, 48, 100),
                (255, 255, 25, 100),
                (240, 50, 230, 100),
                (0, 130, 200, 100),
                (60, 180, 75, 100),
            )
        )
        / 255
)

COLORS_OK = np.array(((255, 0, 0, 100), (0, 255, 0, 100))) / 255

# Constants about problem
CLAZZ = ['Background & Buildings', 'Car', 'Humans & Bikes', 'Interest', 'Sky', 'Nature']
WEIGHTS = np.array([1, 1, 1, 1, 1, 1])
NUM_CLAZZ = len(CLAZZ)
OCCURRENCES = torch.tensor([0.24060006, 0.08469768, 0.00358836, 0.24668484, 0.20268513, 0.22174393])
TRAIN_WEIGHTS = 1 / OCCURRENCES


class Dataset(tdata.Dataset):
    def __init__(self, rgb_file, label_file):
        super().__init__()
        self.rgbs = np.load(rgb_file, mmap_mode='r')  # mmap is way faster for these large data
        self.labels = np.load(label_file, mmap_mode='r')  # mmap is way faster for these large data

    def __len__(self):
        return self.rgbs.shape[0]

    def __getitem__(self, i):
        return {
            'labels': np.asarray(self.labels[i]).astype('i8'),
            # torch wants labels to be of type LongTensor, in order to compute losses
            'rgbs': np.asarray(self.rgbs[i]).astype('f4').transpose((2, 0, 1)) / 255,
            'key': i,  # for saving of the data
            # due to mmap, it is necessary to wrap your data in np.asarray. It does not add almost any overhead as it does not copy anything
        }


class Metrics:
    def __init__(self, num_classes, weights=None, clazz_names=None):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes), 'u8')  # confusion matrix
        self.tps = np.zeros(num_classes, dtype='u8')  # true positives
        self.fps = np.zeros(num_classes, dtype='u8')  # false positives
        self.fns = np.zeros(num_classes, dtype='u8')  # false negatives
        self.weights = weights if weights is not None else np.ones(num_classes)  # Weights of each class for mean IOU
        self.clazz_names = clazz_names if clazz_names is not None else np.arange(num_classes)  # for nicer printing
        self.miou = None

    def update(self, labels, predictions, verbose=True):
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()

        predictions = np.argmax(predictions, 1)  # first dimension are probabilities/scores

        tmp_cm = scipy.sparse.coo_matrix(
            (np.ones(np.prod(labels.shape), 'u8'), (labels.flatten(), predictions.flatten())),
            shape=(self.num_classes, self.num_classes)
        ).toarray()  # Fastest possible way to create confusion matrix. Speed is the necessity here, even then it takes quite too much

        tps = np.diag(tmp_cm)
        fps = tmp_cm.sum(0) - tps
        fns = tmp_cm.sum(1) - tps
        self.cm += tmp_cm
        self.tps += tps
        self.fps += fps
        self.fns += fns

        precisions, recalls, ious, weights, miou = self._compute_stats(tps, fps, fns)

        if verbose:
            self._print_stats(tmp_cm, precisions, recalls, ious, weights, miou)

    def _compute_stats(self, tps, fps, fns):
        with np.errstate(
                all='ignore'):  # any division could be by zero, we don't really care about these errors, we know about these
            precisions = tps / (tps + fps)
            recalls = tps / (tps + fns)
            ious = tps / (tps + fps + fns)
            weights = np.copy(self.weights)
            weights[np.isnan(ious)] = 0
            miou = np.ma.average(ious, weights=weights)
        return precisions, recalls, ious, weights, miou

    def _print_stats(self, cm, precisions, recalls, ious, weights, miou):
        print('Confusion matrix:')
        print(cm)
        print('\n---\n')
        for c in range(self.num_classes):
            print(
                f'Class: {str(self.clazz_names[c]):20s}\t'
                f'Precision: {precisions[c]:.3f}\t'
                f'Recall {recalls[c]:.3f}\t'
                f'IOU: {ious[c]:.3f}\t'
                f'mIOU weight: {weights[c]:.1f}'
            )
        print(f'Mean IOU: {miou}')
        print('\n---\n')

    def print_final(self):
        precisions, recalls, ious, weights, miou = self._compute_stats(self.tps, self.fps, self.fns)
        self._print_stats(self.cm, precisions, recalls, ious, weights, miou)

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), 'u8')
        self.tps = np.zeros(self.num_classes, dtype='u8')
        self.fps = np.zeros(self.num_classes, dtype='u8')
        self.fns = np.zeros(self.num_classes, dtype='u8')


def create_vis(rgb, label, prediction):
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if len(prediction.shape) == 3:
        prediction = np.argmax(prediction, 0)

    h, w, _ = rgb.shape

    gt_map = blend_img(rgb, COLORS_CLAZZ[label])  # we can index colors, wohoo!
    pred_map = blend_img(rgb, COLORS_CLAZZ[prediction])
    ok_map = blend_img(rgb, COLORS_OK[
        (label == prediction).astype('u1')])  # but we cannot do it by boolean, otherwise it won't work
    canvas = np.ones((h * 2 + BORDER, w * 2 + BORDER, 3))
    canvas[:h, :w] = rgb
    canvas[:h, -w:] = gt_map
    canvas[-h:, :w] = pred_map
    canvas[-h:, -w:] = ok_map

    canvas = (np.clip(canvas, 0, 1) * 255).astype('u1')
    return Image.fromarray(canvas)


def blend_img(background, overlay_rgba, gamma=2.2):
    alpha = overlay_rgba[:, :, 3]
    over_corr = np.float_power(overlay_rgba[:, :, :3], gamma)
    bg_corr = np.float_power(background, gamma)
    return np.float_power(over_corr * alpha[..., None] + (1 - alpha)[..., None] * bg_corr, 1 / gamma)  # dark magic
    # partially taken from https://en.wikipedia.org/wiki/Alpha_compositing#Composing_alpha_blending_with_gamma_correction


def evaluate(model, metrics, loader, device, verbose=True, create_imgs=False, save_dir='.'):
    model = model.eval().to(device)

    with torch.no_grad():  # disable gradient computation
        for i, batch in enumerate(loader):
            data = batch['rgbs'].to(device)

            predictions = model(data)
            metrics.update(batch['labels'], predictions, verbose)
            if create_imgs:
                for j, img_id in enumerate(batch['key']):
                    img = create_vis(data[j].cpu().numpy(), batch['labels'][j].numpy(), predictions[j].cpu().numpy())
                    os.makedirs(save_dir, exist_ok=True)
                    img.save(osp.join(save_dir, f'{img_id:04d}.png'))
            print(f'Processed {i / len(loader) * 100:.2f}% of validation data')

    metrics.print_final()
    return metrics


def train(model, metrics, train_loader, device, optimizer, loss_fn, verbose=True):
    model.to(device)
    for i, data in enumerate(tqdm(train_loader)):
        x = data['rgbs'].to(device)
        y = data['labels'].to(device)
        y_pred = model.forward(x)

        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            metrics.update(y, y_pred, verbose)
        if i % 50 == 0:
            print(f'Processed {i / len(train_loader) * 100:.2f}% of training data')


def parse_args():
    parser = argparse.ArgumentParser('HW 04')
    parser.add_argument('-ci', '--create_imgs', default=False, action='store_true',
                        help='Whether to create images. Warning! It will take significantly longer!')
    parser.add_argument('-sd', '--store_dir', default='.',
                        help='Where to store images. Only valid, if create_imgs is set to True')
    parser.add_argument('-bs', '--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Whether to print stats of each minibatch')

    return parser.parse_args()


def main():
    args = parse_args()
    # path = 'data/hw_04'
    path = '/mnt/personal/kuceral4/hw_04'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train_ds = Dataset(os.path.join(path, 'train/rgbs.npy'), os.path.join(path, 'train/labels.npy'))
    val_ds = Dataset(os.path.join(path, 'val/rgbs.npy'), os.path.join(path, 'val/labels.npy'))

    train_loader = tdata.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = tdata.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = load_model()[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(weight=TRAIN_WEIGHTS.to(device))
    metrics = Metrics(NUM_CLAZZ, WEIGHTS, CLAZZ)

    limit = 0
    epochs = 100

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1:02d}')
        train(model, metrics, train_loader, device, optimizer, loss_fn, args.verbose)
        metrics.reset()
        metrics = evaluate(model, metrics, val_loader, device, args.verbose, args.create_imgs, args.store_dir)

        if metrics.miou > limit:
            limit = metrics.miou
            torch.save(model.state_dict(), f'unet_{limit}.pth')
            print('Model saved')


if __name__ == '__main__':
    main()
