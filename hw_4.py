from typing import Tuple
import torchvision as tv
import torchvision.transforms as tfms
from torch.utils.data import DataLoader
import torch.utils.data as tdata
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from time import strftime, gmtime
import sys
import scipy.sparse



def main():
    CLAZZ = ['Background & Buildings', 'Car', 'Humans & Bikes', 'Interest', 'Sky', 'Nature']
    WEIGHTS = np.array([0, 1, 1, 1, 0, 0])
    NUM_CLAZZ = len(CLAZZ)  
    #vgg19_bn = tv.models.vgg19_bn(True)
    vgg13_bn = tv.models.vgg13_bn(True)
    #vgg16_bn = tv.models.vgg16_bn(True)
    BATCH_SIZE = 4
    EPOCHS = 50
    num_classes = 6
    # [0.01443778 0.04087064 0.89691808 0.01431744 0.01766317 0.0157929 ]
    weights = torch.tensor([0.5*0.0134883,  0.03831609, 2.7*0.90439325, 1.4*0.0131556,  0.01601149, 1.1*0.01463527])
    #weights = torch.tensor([0.0134883,  0.04231609, 0.92439325, 0.0171556,  0.01601149, 0.01463527])
    weights = weights/weights.sum()
    print(weights)
    train_data = f'/local/temporary/vir/hw04/train/rgbs.npy'
    train_labels = f'/local/temporary/vir/hw04/train/labels.npy'
    test_data = f'/local/temporary/vir/hw04/val/rgbs.npy'
    test_labels = f'/local/temporary/vir/hw04/val/labels.npy'

    #device = 'cpu'
    device = get_device()
    if len(sys.argv) > 1 and int(sys.argv[1]) == 1:
        model, encoder_name = load_model()
        model.to(device)
        print("load")
    else:
        print("new")
        model = UnetFromPretrained(vgg13_bn.features, num_classes).to(device)
   
    #save_model(model, f'best_model.pth')

    train_ds = Dataset(train_data, train_labels)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    
    test_ds = Dataset(test_data, test_labels)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    
    # dist = get_distribution((train_ds, test_ds))
    # weights = 1/dist
    # weights = weights/weights.sum()
    # print(weights)

    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    opt = torch.optim.Adam(model.decoder.parameters(), lr=0.02)
    metrics = Metrics(NUM_CLAZZ, WEIGHTS, CLAZZ)
    #acc = test_model(model, test_dl, device)
    #metr = validate(model, metrics, test_dl, device)
    #print(metr.miou)
    acc_best = 0
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_dl, loss_fn, opt, device, epoch, EPOCHS)
        #acc = test_model(model, test_dl, device)
        metrics.reset()
        metr = validate(model, metrics, test_dl, device)
        acc = metr.miou
        print('Model saved.')
        save_model(model, './best_model_{}_{:.2f}.pth'.format(epoch, acc*100))
        if acc > acc_best:
            acc_best = acc
            save_model(model, './best_model.pth')
        print('Epoch [{}/{}], loss {:.2f}, acc {:.2f}, acc_best {:.2f}'.format(epoch+1, EPOCHS, loss, acc*100, acc_best*100))

def load_model() -> Tuple[nn.Module, str]:
    '''
    :return: model: your trained NN; encoder_name: name of NN, which was used to create your NN
    '''
    vgg13_bn = tv.models.vgg13_bn(True)
    #vgg16_bn = tv.models.vgg16_bn(True)
    #vgg19_bn = tv.models.vgg19_bn(True)
    num_classes = 6
    model = UnetFromPretrained(vgg13_bn.features, num_classes)
    model.decoder.load_state_dict(torch.load(f'best_model.pth', map_location=torch.device('cpu')))
    encoder_name = 'vgg13_bn'
    return model, encoder_name

def validate(model, metrics, test_dl, device):
    model.eval()
    with torch.no_grad():  # disable gradient computation
        data_len = len(test_dl)
        for i, batch in enumerate(test_dl):
            data = batch['rgbs'].to(device)
 
            predictions = model(data)
            metrics.update(batch['labels'], predictions, False)
            if i % 50 == 0:
                print('Eval [{}/{}]'.format(i, data_len), strftime("%H:%M:%S", gmtime()))
    metrics.print_final()
    return metrics

def get_distribution(datasets):
    num_classes = 6
    dist = np.zeros(num_classes)
    for dataset in datasets:
        for img in dataset:
            for i in range(num_classes):
                dist[i] += (img['labels']==i).sum()
    dist = dist/np.sum(dist)
    return dist

class Dataset(tdata.Dataset):
    def __init__(self, rgb_file, label_file):
        super().__init__()
        self.rgbs = np.load(rgb_file, mmap_mode='r')  # mmap is way faster for these large data
        self.labels = np.load(label_file, mmap_mode='r')  # mmap is way faster for these large data
 
    def __len__(self):
        return self.rgbs.shape[0]
 
    def __getitem__(self, i):
        return {
            'labels': np.asarray(self.labels[i]).astype('i8'),  # torch wants labels to be of type LongTensor, in order to compute losses
            'rgbs': np.asarray(self.rgbs[i]).astype('f4').transpose((2, 0, 1)) / 255,
            'key': i,  # for saving of the data
            # due to mmap, it is necessary to wrap your data in np.asarray. It does not add almost any overhead as it does not copy anything
        }

# class Dataset(tdata.Dataset):
#     def __init__(self, rgbs, labels):
#         super().__init__()
#         self.rgbs = rgbs
#         self.labels = labels

#     def __len__(self):
#         return self.rgbs.shape[0]

#     def __getitem__(self, i):
#         return {
#             'labels': np.asarray(self.labels[i]).astype('i8'),
#             'rgbs': np.asarray(self.rgbs[i]).astype('f4').transpose((2, 0, 1)) / 255,
#         }

class UnetFromPretrained(torch.nn.Module):
    '''This is my super cool, but super dumb module'''
 
    def __init__(self, encoder: nn.Module, num_classes: int):
        '''
        :param encoder: nn.Sequential, pretrained encoder
        :param num_classes: Python int, number of segmentation classes
        '''
        super(UnetFromPretrained, self).__init__()
        self.num_classes = num_classes
        self.encoder = encoder
        self.decoder_layers = []
        max_pool = False
        enc_reversed = reversed(self.encoder)
        for i, layer in enumerate(enc_reversed):
            if isinstance(layer, nn.Conv2d):
                if max_pool:
                    in_channels = layer.out_channels*2
                else:
                    in_channels = layer.out_channels
                out_channels = layer.in_channels
                self.decoder_layers.append(
                    nn.Conv2d(in_channels, layer.in_channels, layer.kernel_size,
                                       layer.stride, layer.padding))
                self.decoder_layers.append(nn.BatchNorm2d(layer.in_channels)) 
                self.decoder_layers.append(nn.ReLU(inplace=True))
                max_pool = False
            elif isinstance(layer, nn.MaxPool2d):
                layer.return_indices = True
                self.decoder_layers.append(
                    nn.MaxUnpool2d(layer.kernel_size, layer.stride, layer.padding))
                #self.decoder_layers.append(
                #    nn.ConvTranspose2d(enc_reversed[i+1].out_channels*2, layer.kernel_size, layer.stride, layer.padding))
                max_pool = True
        self.decoder_layers.append(nn.Conv2d(out_channels, num_classes, kernel_size=(3,3), padding=(1,1)))
        #self.decoder_layers.append(nn.Softmax2d())

        self.decoder = nn.Sequential(*self.decoder_layers)
        #print(self.encoder)
        #print(self.decoder)


    def forward(self, x):
        #shape = x.shape
        pool_indices = []
        skip_con_data = []
        out = x
        #print('ENCODER')
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                skip_con_data.append(out)
                #print(out.shape)
                out, indices = layer(out)
                pool_indices.append(indices)
            else:
                out = layer(out)
        #print('DECODER')
        idx_pool = 0
        idx_skip_con = 0
        skip_con = False
        for layer in self.decoder:
            #print(layer)
            if isinstance(layer, nn.MaxUnpool2d):
                idx_pool += 1
                out = layer(out, pool_indices[-idx_pool])
                skip_con = True
            else:
                if skip_con:
                    idx_skip_con +=1
                    #print(out.shape)
                    #print(skip_con_data[-idx_skip_con].shape)
                    out = torch.cat((skip_con_data[-idx_skip_con], out), dim=1)
                #print(out.shape)
                out = layer(out)
                skip_con = False
        #print(shape, out.shape)
        #torch.randn(shape[0], self.num_classes, *shape[2:], device=x.device)
        return out
 
def save_model(model, destination):
    torch.save(model.decoder.state_dict(), destination)
 
def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, EPOCHS):
    model.train()
    loss_sum = 0
    total_step = len(train_loader)
    for idx, batch in enumerate(train_loader):
        data = batch['rgbs'].to(device)
        labels = batch['labels'].to(device)
        
        output = model(data)
        loss = loss_fn(output, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        loss_sum += loss.item()
        if idx%10==0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, EPOCHS, idx+1, total_step, loss.item()), strftime("%H:%M:%S", gmtime()))
    loss = loss_sum / (idx+1)
    return loss


def test_model(model, test_loader, device):
    model.eval()
    acc = 0
    with torch.no_grad():
        data_len = len(test_loader)
        for idx, batch in enumerate(test_loader):
            data = batch['rgbs'].to(device)
            labels = batch['labels'].to(device)

            output = model(data)
            #acc += iou(output, labels, device)
            acc += accuracy(output, labels)
            
            #output = output.argmax(1)
            
            #print(output)
            
            #class_idx, prediction_counts = np.unique(output, return_counts=True)
            
            #for i in range(len(class_idx)):
            if idx % 50 == 0:
                print('Eval [{}/{}]'.format(idx, data_len), strftime("%H:%M:%S", gmtime()))
    acc /= (idx + 1)
    print(f'{acc * 100:.1f}%')

    return acc


def accuracy(prediction, labels_batch, dim=1):
    pred_index = prediction.argmax(dim)
    return (pred_index == labels_batch).float().mean()
 
 
def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    index = np.argmax(memory_available[:-1])  # Skip the 7th card --- it is reserved for evaluation!!!
    return int(index)
 
 
def get_device():  
    if torch.cuda.is_available():
        gpu = get_free_gpu()
        print('GPU:', gpu)
        device = torch.device(gpu)
    else:
        device = 'cpu'
    return device

class Metrics:
    def __init__(self, num_classes, weights=None, clazz_names=None):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes), 'u8')  # confusion matrix
        self.tps = np.zeros(num_classes, dtype='u8')  # true positives
        self.fps = np.zeros(num_classes, dtype='u8')  # false positives
        self.fns = np.zeros(num_classes, dtype='u8')  # false negatives
        self.weights = weights if weights is not None else np.ones(num_classes)  # Weights of each class for mean IOU
        self.clazz_names = clazz_names if clazz_names is not None else np.arange(num_classes)  # for nicer printing
        self.miou = 0.0

    def update(self, labels, predictions, verbose=True):
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
 
        predictions = np.argmax(predictions, 1)  # first dimension are probabilities/scores
 
        tmp_cm = scipy.sparse.coo_matrix(
            (np.ones(np.prod(labels.shape), 'u8'), (labels.flatten(), predictions.flatten())), shape=(self.num_classes, self.num_classes)
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
        with np.errstate(all='ignore'):  # any division could be by zero, we don't really care about these errors, we know about these
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
        self.miou = miou
        self._print_stats(self.cm, precisions, recalls, ious, weights, miou)
 
    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), 'u8')
        self.tps = np.zeros(self.num_classes, dtype='u8')
        self.fps = np.zeros(self.num_classes, dtype='u8')
        self.fns = np.zeros(self.num_classes, dtype='u8')
        self.miou = 0.0

if __name__ == '__main__':
    main()