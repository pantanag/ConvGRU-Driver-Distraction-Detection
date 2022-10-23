# from DAD import DAD
# from DAD_2 import DAD
from dataset import DAD
from train_model import train
from DAD import plot_video
import torch.nn as nn
import numpy as np
import warnings
from torch.utils.data import ConcatDataset
import subprocess
import torch
import torchvision
from dataset import get_labels_concat
from extra_transformations import ReverseFrames
import random
import spatial_transforms
import extra_transformations
warnings.simplefilter("ignore", UserWarning)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)


def get_weights_concat(concatDataset):
    targets = torch.Tensor(get_labels_concat(concatDataset)).int()
    labels, counts = np.unique(targets, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    samples_weight = [class_weights[class_id] for class_id in targets]
    return samples_weight


def get_pos_weight_concat(concatDataset):
    neg = 0
    pos = 0
    for i in range(len(concatDataset.datasets)):
        dataset = concatDataset.datasets[i]
        if dataset.data[0]['label'] == 0.:
            neg += dataset.__len__()
        else:
            pos += dataset.__len__()
    pos_weight = torch.Tensor([neg/pos])
    return pos_weight


def run():
    # seed = np.random.randint(2147483647)
    # random.seed(seed)
    # torch.manual_seed(seed)
    view = 'top_IR'
    normal_train_trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=(165, 194)),
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize(size=(160, 160)),
                                                   torchvision.transforms.Normalize([0.], [1.])])
    normal_eval_trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=(165, 194)),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Resize(size=(160, 160)),
                                                        torchvision.transforms.Normalize([0.], [1.])])
    flipped_trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=(165, 194)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize(size=(160, 160)),
                                                    torchvision.transforms.RandomHorizontalFlip(p=1.),
                                                    torchvision.transforms.Normalize([0.], [1.])])
    blurred_trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=(165, 194)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize(size=(160, 160)),
                                                    torchvision.transforms.GaussianBlur(kernel_size=(13, 13), sigma=(5, 5)),
                                                    torchvision.transforms.Normalize([0.], [1.])])
    noise_trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=(165, 194)),
                                                  # spatial_transforms.SaltImage(),
                                                  extra_transformations.AddNoise('PIL', 's&p'),
                                                  torchvision.transforms.Resize(size=(160, 160)),
                                                  torchvision.transforms.Normalize([0.], [1.])])

    train_normal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='normal', sample_duration=16, spatial_transform=normal_train_trans)
    train_anormal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='anormal', sample_duration=16, spatial_transform=normal_train_trans)
    reversed_normal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='normal', sample_duration=16, spatial_transform=noise_trans, temporal_transform=ReverseFrames(), k=2500)
    reversed_anormal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='anormal', sample_duration=16, spatial_transform=noise_trans, temporal_transform=ReverseFrames(), k=2500)
    flipped_normal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='normal', sample_duration=16, spatial_transform=flipped_trans, k=8000)
    flipped_anormal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='anormal', sample_duration=16, spatial_transform=flipped_trans, k=4500)
    blurred_normal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='normal', sample_duration=16, spatial_transform=blurred_trans, k=8000)
    blurred_anormal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='anormal', sample_duration=16, spatial_transform=blurred_trans, k=2500)
    noise_normal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='normal', sample_duration=16, spatial_transform=noise_trans, k=12000)
    noise_anormal_ds = DAD(root_path='E:/DAD/', subset='train', view=view, type='anormal', sample_duration=16, spatial_transform=noise_trans, k=10000)
    # noise_normal_ds, noise_anormal_ds,
    whole_ds = ConcatDataset([train_normal_ds, train_anormal_ds, flipped_normal_ds, flipped_anormal_ds, blurred_normal_ds, blurred_anormal_ds, noise_normal_ds, noise_anormal_ds, reversed_normal_ds, reversed_anormal_ds])
    pos_weight = get_pos_weight_concat(whole_ds).cuda()

    weights = get_weights_concat(whole_ds)
    test_ds = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view=view, spatial_transform=normal_eval_trans)
    model_head = 'ConvGRUv4'
    lr = 1e-5
    weight_decay = 1e-6
    optimizer = 'AdaBound'
    loss_fcn = nn.BCEWithLogitsLoss(pos_weight)
    batch_size = 32
    n_workers = 2
    epochs = 50
    version = 'Version_3.0'
    desc = '''Learning Rate: {}
Weight Decay: {}
Optimizer: {}
Batch Size: {}
Loss Function: {}
Whole Dataset
    '''.format(lr, weight_decay, optimizer, batch_size, loss_fcn.__class__.__name__)
    train(model_head=model_head, optim_head=optimizer, lr=lr, loss_fcn=loss_fcn, n_epochs=epochs, train_ds=whole_ds,
          batch_size=batch_size, n_workers=n_workers, version=version, desc=desc, view=view, test_ds=test_ds, weights=weights,
          weight_decay=weight_decay)
    #subprocess.run(["shutdown", "-s"])


if __name__ == '__main__':
    run()
