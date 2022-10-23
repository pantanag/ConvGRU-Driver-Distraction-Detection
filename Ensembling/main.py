import torchvision
from dataset import DAD
import torch
from tqdm import tqdm
import numpy as np
from Saved_Models.ResNextGRU.front_IR import front_IR_v1, front_IR_v2
from Saved_Models.ResNextGRU.front_depth import front_depth_v1, front_depth_v13
from Saved_Models.ResNextGRU.top_depth import top_depth_v1
from Saved_Models.ResNextGRU.top_IR import top_IR_v1, top_IR_v2
from extra_transformations import UnNormalize
from sklearn.metrics import f1_score, auc, roc_curve
from torch.optim import SGD
from Model_Architectures.Stacker import Stacker
from Model_Architectures.MobileGRUStacker import MobileGRUStacker
from Model_Architectures.ConvGRUStacker import ConvGRUStacker
from Optimizers.adabound import AdaBound
from torch.utils.data import ConcatDataset
import torch.nn as nn
from Ensembling.train import train
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data.sampler import WeightedRandomSampler
from dataset import get_labels_concat


def class_imbalance_sampler(labels):
    class_count = torch.bincount(labels)
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels))
    return sampler


def load_semi_model(model, path):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return model


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def run():
    # Transforms
    normal_trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=(165, 194)),
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

    # unorm = UnNormalize(mean=[0.], std=[1.0])

    # Datasets
    normal_ds_front_IR = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='front_IR', spatial_transform=normal_trans)
    normal_ds_front_depth = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='front_depth', spatial_transform=normal_trans)
    normal_ds_top_IR = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='top_IR', spatial_transform=normal_trans)
    normal_ds_top_depth = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='top_depth', spatial_transform=normal_trans)

    anormal_ds_front_IR = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='front_IR', spatial_transform=normal_trans)
    anormal_ds_front_depth = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='front_depth', spatial_transform=normal_trans)
    anormal_ds_top_IR = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='top_IR', spatial_transform=normal_trans)
    anormal_ds_top_depth = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='top_depth', spatial_transform=normal_trans)

    normal_ds_front_IR_flipped = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='front_IR', spatial_transform=flipped_trans, k=10000)
    normal_ds_front_depth_flipped = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='front_depth', spatial_transform=flipped_trans, k=10000)
    normal_ds_top_IR_flipped = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='top_IR', spatial_transform=flipped_trans, k=10000)
    normal_ds_top_depth_flipped = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='top_depth', spatial_transform=flipped_trans, k=10000)

    anormal_ds_front_IR_flipped = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='front_IR', spatial_transform=flipped_trans, k=10000)
    anormal_ds_front_depth_flipped = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='front_depth', spatial_transform=flipped_trans, k=10000)
    anormal_ds_top_IR_flipped = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='top_IR', spatial_transform=flipped_trans, k=10000)
    anormal_ds_top_depth_flipped = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='top_depth', spatial_transform=flipped_trans, k=10000)

    normal_ds_front_IR_blurred = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='front_IR', spatial_transform=blurred_trans, k=10000)
    normal_ds_front_depth_blurred = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='front_depth', spatial_transform=blurred_trans, k=10000)
    normal_ds_top_IR_blurred = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='top_IR', spatial_transform=blurred_trans, k=10000)
    normal_ds_top_depth_blurred = DAD(root_path='E:/DAD/', subset='train', type='normal', sample_duration=16, view='top_depth', spatial_transform=blurred_trans, k=10000)

    anormal_ds_front_IR_blurred = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='front_IR', spatial_transform=blurred_trans, k=10000)
    anormal_ds_front_depth_blurred = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='front_depth', spatial_transform=blurred_trans, k=10000)
    anormal_ds_top_IR_blurred = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='top_IR', spatial_transform=blurred_trans, k=10000)
    anormal_ds_top_depth_blurred = DAD(root_path='E:/DAD/', subset='train', type='anormal', sample_duration=16, view='top_depth', spatial_transform=blurred_trans, k=10000)

    # train_ds_front_IR = ConcatDataset([normal_ds_front_IR, anormal_ds_front_IR, normal_ds_front_IR_flipped, anormal_ds_front_IR_flipped, normal_ds_front_IR_blurred, anormal_ds_front_IR_blurred])
    # train_ds_front_depth = ConcatDataset([normal_ds_front_depth, anormal_ds_front_depth, normal_ds_front_depth_flipped, anormal_ds_front_depth_flipped, normal_ds_front_depth_blurred, anormal_ds_front_depth_blurred])
    # train_ds_top_IR = ConcatDataset([normal_ds_top_IR, anormal_ds_top_IR, normal_ds_top_IR_flipped, anormal_ds_top_IR_flipped, normal_ds_top_IR_blurred, anormal_ds_top_IR_blurred])
    # train_ds_top_depth = ConcatDataset([normal_ds_top_depth, anormal_ds_top_depth, normal_ds_top_depth_flipped, anormal_ds_top_depth_flipped, normal_ds_top_depth_blurred, anormal_ds_top_depth_blurred])

    train_ds_front_IR = ConcatDataset([normal_ds_front_IR, anormal_ds_front_IR, normal_ds_front_IR_flipped, anormal_ds_front_IR_flipped, normal_ds_front_IR_blurred, anormal_ds_front_IR_blurred])
    train_ds_front_depth = ConcatDataset([normal_ds_front_depth, anormal_ds_front_depth, normal_ds_front_depth_flipped, anormal_ds_front_depth_flipped, normal_ds_front_depth_blurred, anormal_ds_front_depth_blurred])
    train_ds_top_IR = ConcatDataset([normal_ds_top_IR, anormal_ds_top_IR, normal_ds_top_IR_flipped, anormal_ds_top_IR_flipped, normal_ds_top_IR_blurred, anormal_ds_top_IR_blurred])
    train_ds_top_depth = ConcatDataset([normal_ds_top_depth, anormal_ds_top_depth, normal_ds_top_depth_flipped, anormal_ds_top_depth_flipped, normal_ds_top_depth_blurred, anormal_ds_top_depth_blurred])

    # evens = list(range(0, len(train_ds_front_IR), 10))
    # train_ds_front_IR = torch.utils.data.Subset(train_ds_front_IR, evens)
    # train_ds_front_depth = torch.utils.data.Subset(train_ds_front_depth, evens)
    # train_ds_top_IR = torch.utils.data.Subset(train_ds_top_IR, evens)
    # train_ds_top_depth = torch.utils.data.Subset(train_ds_top_depth, evens)

    eval_ds_front_IR = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view='front_IR', spatial_transform=normal_trans)
    eval_ds_front_depth = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view='front_depth', spatial_transform=normal_trans)
    eval_ds_top_IR = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view='top_IR', spatial_transform=normal_trans)
    eval_ds_top_depth = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view='top_depth', spatial_transform=normal_trans)

    # # Samplers
    # labels = get_labels_concat(train_ds_front_IR)
    # labels = [int(label) for label in labels]
    # labels = torch.tensor(labels)
    # # sampler = class_imbalance_sampler(labels)
    sampler = torch.utils.data.RandomSampler(train_ds_front_IR)
    #
    # labels_eval = eval_ds_front_IR.get_labels()
    # labels_eval = [int(label) for label in labels_eval]
    # labels_eval = torch.tensor(labels_eval)
    # # sampler_eval = class_imbalance_sampler(labels_eval)
    sampler_eval = torch.utils.data.RandomSampler(eval_ds_front_IR)

    # DataLoaders
    train_dl_front_IR = torch.utils.data.DataLoader(train_ds_front_IR, batch_size=32, num_workers=1, sampler=sampler, pin_memory=True)
    train_dl_front_depth = torch.utils.data.DataLoader(train_ds_front_depth, batch_size=32, num_workers=1, sampler=sampler, pin_memory=True)
    train_dl_top_IR = torch.utils.data.DataLoader(train_ds_top_IR, batch_size=32, num_workers=1, sampler=sampler, pin_memory=True)
    train_dl_top_depth = torch.utils.data.DataLoader(train_ds_top_depth, batch_size=32, num_workers=1, sampler=sampler, pin_memory=True)

    eval_dl_front_IR = torch.utils.data.DataLoader(eval_ds_front_IR, batch_size=32, num_workers=1, sampler=sampler_eval, pin_memory=True)
    eval_dl_front_depth = torch.utils.data.DataLoader(eval_ds_front_depth, batch_size=32, num_workers=1, sampler=sampler_eval, pin_memory=True)
    eval_dl_top_IR = torch.utils.data.DataLoader(eval_ds_top_IR, batch_size=32, num_workers=1, sampler=sampler_eval, pin_memory=True)
    eval_dl_top_depth = torch.utils.data.DataLoader(eval_ds_top_depth, batch_size=32, num_workers=1, sampler=sampler_eval, pin_memory=True)

    dataloaders = [train_dl_front_IR, train_dl_front_depth, train_dl_top_IR, train_dl_top_depth]
    eval_dataloaders = [eval_dl_front_IR, eval_dl_front_depth, eval_dl_top_IR, eval_dl_top_depth]

    stacker = MobileGRUStacker()
    # stacker = Stacker()
    # stacker = ConvGRUStacker()
    stacker.cuda()
    stacker.train()
    lr = 1e-5
    optimizer = SGD(stacker.parameters(), lr=lr, momentum=0.9, nesterov=True)
    # optimizer = AdaBound(stacker.parameters(), lr=lr, weight_decay=1e-6, amsbound=True, final_lr=1e-1)
    epochs = 20
    loss_fcn = nn.BCEWithLogitsLoss()
    threshold = 0.
    version = 'Version_2.0'
    desc = '''Learning Rate: {}
    Optimizer: {}
    Batch Size: {}
    Loss Function: {}'''.format(lr, 'SGD', '32', loss_fcn.__class__.__name__)
    train(stacker=stacker, dataloaders=dataloaders, eval_dataloaders=eval_dataloaders, epochs=epochs, optimizer=optimizer, loss_fcn=loss_fcn, threshold=threshold, version=version, desc=desc)


if __name__ == '__main__':
    run()
