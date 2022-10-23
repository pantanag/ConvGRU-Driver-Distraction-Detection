import torchvision
from dataset import DAD
import torch
from tqdm import tqdm
import numpy as np
# Mobile
from Saved_Models.mobileGRU.front_IR import front_IR_v3 as mobile_front_IR_v3
from Saved_Models.mobileGRU.front_depth import front_depth_v1 as mobile_front_depth_v1
from Saved_Models.mobileGRU.top_IR import top_IR_v1 as mobile_top_IR_v1
from Saved_Models.mobileGRU.top_depth import top_depth_v1 as mobile_top_depth_v1
# ResNeXt
from Saved_Models.ResNextGRU.front_IR import front_IR_v2 as resnext_front_IR_v2
from Saved_Models.ResNextGRU.front_depth import front_depth_v12 as resnext_front_depth_v12
from Saved_Models.ResNextGRU.top_IR import top_IR_v2 as resnext_top_IR_v2
from Saved_Models.ResNextGRU.top_depth import top_depth_v2 as resnext_top_depth_v2
# ConvGRU
from Saved_Models.ConvGRUv4.front_IR import front_IR_v1 as convgru_front_IR_v1
from Saved_Models.ConvGRUv4.front_depth import front_depth_v1 as convgru_front_depth_v1
from Saved_Models.ConvGRUv4.top_IR import top_IR_v2 as convgru_top_IR_v2
from Saved_Models.ConvGRUv4.top_depth import top_depth_v13 as convgru_top_depth_v13
from extra_transformations import UnNormalize
from sklearn.metrics import f1_score, auc, roc_curve
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def run():
    # Transforms
    normal_eval_trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=(165, 194)),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Resize(size=(160, 160)),
                                                        torchvision.transforms.Normalize([0.], [1.])])

    unorm = UnNormalize(mean=[0.], std=[1.0])

    # Datasets
    test_ds_front_IR = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view='front_IR', spatial_transform=normal_eval_trans)
    test_ds_front_depth = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view='front_depth', spatial_transform=normal_eval_trans)
    test_ds_top_IR = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view='top_IR', spatial_transform=normal_eval_trans)
    test_ds_top_depth = DAD(root_path='E:/DAD/', subset='validation', sample_duration=16, view='top_depth', spatial_transform=normal_eval_trans)

    # DataLoaders
    test_dl_front_IR = torch.utils.data.DataLoader(test_ds_front_IR, batch_size=32, num_workers=1, pin_memory=True)
    test_dl_front_depth = torch.utils.data.DataLoader(test_ds_front_depth, batch_size=32, num_workers=1, pin_memory=True)
    test_dl_top_IR = torch.utils.data.DataLoader(test_ds_top_IR, batch_size=32, num_workers=1, pin_memory=True)
    test_dl_top_depth = torch.utils.data.DataLoader(test_ds_top_depth, batch_size=32, num_workers=1, pin_memory=True)

    base_model = 'convgru'
    version = 10
    # ===============================
    #         Thresholds
    # ===============================
    if base_model.lower() == 'mobile':
        # Mobile
        front_IR_thresh = torch.tensor([-0.9820]).cuda()
        front_depth_thresh = torch.tensor([-0.2032]).cuda()
        top_IR_thresh = torch.tensor([-0.9741]).cuda()
        top_depth_thresh = torch.tensor([-1.1021]).cuda()
    elif base_model.lower() == 'resnext':
        # ResNeXt
        front_IR_thresh = torch.tensor([-1.5647]).cuda()
        front_depth_thresh = torch.tensor([-1.6490]).cuda()
        top_IR_thresh = torch.tensor([-1.4795]).cuda()
        top_depth_thresh = torch.tensor([-0.9074]).cuda()
    elif base_model.lower() == 'convgru':
        # ConvGRU
        front_IR_thresh = torch.tensor([-0.9005]).cuda()
        front_depth_thresh = torch.tensor([-0.7038]).cuda()
        top_IR_thresh = torch.tensor([-1.3789]).cuda()
        top_depth_thresh = torch.tensor([-1.0242]).cuda()

    # ===============================
    #           Models
    # ===============================
    if base_model.lower() == 'mobile':
        # Mobile
        front_IR_model = mobile_front_IR_v3.mobileGRU().cuda()
        front_depth_model = mobile_front_depth_v1.mobileGRU().cuda()
        top_IR_model = mobile_top_IR_v1.mobileGRU().cuda()
        top_depth_model = mobile_top_depth_v1.mobileGRU().cuda()
        front_IR_version = 'front_IR_v3'
        front_depth_version = 'front_depth_v1'
        top_IR_version = 'top_IR_v1'
        top_depth_version = 'top_depth_v1'
        front_IR_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/mobileGRU/front_IR/front_IR_v3.pth'
        front_depth_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/mobileGRU/front_depth/front_depth_v1.pth'
        top_IR_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/mobileGRU/top_IR/top_IR_v1.pth'
        top_depth_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/mobileGRU/top_depth/top_depth_v1.pth'
    elif base_model.lower() == 'resnext':
        # ResNeXt
        front_IR_model = resnext_front_IR_v2.ResNextGRU().cuda()
        front_depth_model = resnext_front_depth_v12.ResNextGRU().cuda()
        top_IR_model = resnext_top_IR_v2.ResNextGRU().cuda()
        top_depth_model = resnext_top_depth_v2.ResNextGRU().cuda()
        front_IR_version = 'front_IR_v2'
        front_depth_version = 'front_depth_v12'
        top_IR_version = 'top_IR_v2'
        top_depth_version = 'top_depth_v2'
        front_IR_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/front_IR/front_IR_v2.pth'
        front_depth_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/front_depth/front_depth_v12.pth'
        top_IR_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/top_IR/top_IR_v2.pth'
        top_depth_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/top_depth/top_depth_v2.pth'
    elif base_model.lower() == 'convgru':
        front_IR_model = convgru_front_IR_v1.ConvGRUv4().eval().cuda()
        front_depth_model = convgru_front_depth_v1.ConvGRUv4().eval().cuda()
        top_IR_model = convgru_top_IR_v2.ConvGRUv4().eval().cuda()
        top_depth_model = convgru_top_depth_v13.ConvGRUv4().eval().cuda()
        front_IR_version = 'front_IR_v1'
        front_depth_version = 'front_depth_v1'
        top_IR_version = 'top_IR_v2'
        top_depth_version = 'top_depth_v13'
        front_IR_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ConvGRUv4/front_IR/front_IR_v1.pth'
        front_depth_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ConvGRUv4/front_depth/front_depth_v1.pth'
        top_IR_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ConvGRUv4/top_IR/top_IR_v2.pth'
        top_depth_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ConvGRUv4/top_depth/top_depth_v13.pth'

    # Load Models
    front_IR_model.load_state_dict(torch.load(front_IR_path))
    front_IR_model.eval()
    front_depth_model.load_state_dict(torch.load(front_depth_path))
    front_depth_model.eval()
    top_IR_model.load_state_dict(torch.load(top_IR_path))
    top_IR_model.eval()
    top_depth_model.load_state_dict(torch.load(top_depth_path))
    top_depth_model.eval()

    # Ground Truth
    total_labels = np.empty(shape=(0, ))

    # Total Predictions
    total_pred_front_IR = np.empty(shape=(0,))
    total_pred_front_depth = np.empty(shape=(0,))
    total_pred_top_IR = np.empty(shape=(0,))
    total_pred_top_depth = np.empty(shape=(0,))

    # Total Scores
    scores_front_IR = np.empty(shape=(0,))
    scores_front_depth = np.empty(shape=(0,))
    scores_top_IR = np.empty(shape=(0,))
    scores_top_depth = np.empty(shape=(0,))

    for data1, data2, data3, data4 in tqdm(zip(test_dl_front_IR, test_dl_front_depth, test_dl_top_IR, test_dl_top_depth), total=len(test_dl_front_IR), desc='Calculating Scores '):
        inputs_front_IR, labels = data1
        inputs_front_depth, _ = data2
        inputs_top_IR, _ = data3
        inputs_top_depth, _ = data4
        inputs_front_IR, inputs_front_depth, inputs_top_IR, inputs_top_depth = inputs_front_IR.cuda(), inputs_front_depth.cuda(), inputs_top_IR.cuda(), inputs_top_depth.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs_front_IR = front_IR_model(inputs_front_IR)
            outputs_front_depth = front_depth_model(inputs_front_depth)
            outputs_top_IR = top_IR_model(inputs_top_IR)
            outputs_top_depth = top_depth_model(inputs_top_depth)
            pred_front_IR = (outputs_front_IR > front_IR_thresh).float()
            total_pred_front_IR = np.concatenate((total_pred_front_IR, pred_front_IR.detach().cpu()))
            scores_front_IR = np.concatenate((scores_front_IR, outputs_front_IR.detach().cpu()))
            pred_front_depth = (outputs_front_depth > front_depth_thresh).float()
            scores_front_depth = np.concatenate((scores_front_depth, outputs_front_depth.detach().cpu()))
            total_pred_front_depth = np.concatenate((total_pred_front_depth, pred_front_depth.detach().cpu()))
            pred_top_IR = (outputs_top_IR > top_IR_thresh).float()
            scores_top_IR = np.concatenate((scores_top_IR, outputs_top_IR.detach().cpu()))
            total_pred_top_IR = np.concatenate((total_pred_top_IR, pred_top_IR.detach().cpu()))
            pred_top_depth = (outputs_top_depth > top_depth_thresh).float()
            scores_top_depth = np.concatenate((scores_top_depth, outputs_top_depth.detach().cpu()))
            total_pred_top_depth = np.concatenate((total_pred_top_depth, pred_top_depth.detach().cpu()))
            total_labels = np.concatenate((total_labels, labels.detach().cpu()))

    total_sum = total_pred_front_IR + total_pred_front_depth + total_pred_top_depth + total_pred_top_IR
    total_sum = np.where(total_sum >= 3., 1., total_pred_top_IR)  # Change it to best camera
    acc = 100 * (np.sum(total_sum == total_labels) / total_labels.shape[0])
    f1 = f1_score(total_labels, total_sum)
    print('Total Test Acc: {}, F1 Score: {}'.format(acc, f1))
    cf_matrix = confusion_matrix(total_labels, total_sum)
    create_confusion_matrix(base_model, 'Majority', cf_matrix)
    plt.close()
    # 2nd way
    total_scores = ((scores_front_IR + scores_front_depth) / 2 + (scores_top_depth + scores_top_IR) / 2) / 2
    thresh = ((front_IR_thresh + front_depth_thresh) / 2 + (top_depth_thresh + top_IR_thresh) / 2) / 2
    total_pred = np.where(total_scores > thresh.item(), 1., 0.)
    acc_avg = 100 * (np.sum(total_pred == total_labels) / total_labels.shape[0])
    f1_avg = f1_score(total_labels, total_pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(total_labels, sigmoid(total_scores))
    auc_score_avg = auc(false_positive_rate, true_positive_rate)
    print('Total Test Acc: {}, F1 Score: {}, AUC: {}'.format(acc_avg, f1_avg, auc_score_avg))
    cf_matrix2 = confusion_matrix(total_labels, total_pred)
    create_confusion_matrix(base_model, 'Average', cf_matrix2)
    plt.close()
    save_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Evaluating/Results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + 'Ensemble_v' + str(version) + '.txt', 'w') as f:
        f.write('====================================================' + '\n')
        f.write("                   Models Used                      " + '\n')
        f.write('====================================================' + '\n')
        f.write(base_model + '/' + front_IR_version + '\n' + base_model + '/' + front_depth_version + '\n' + base_model + '/' + top_IR_version + '\n' + base_model + '/' + top_depth_version + '\n\n')
        f.write("====================================================" + '\n')
        f.write('                 Majority Voting                    ' + '\n')
        f.write("====================================================" + '\n')
        f.write('Total Test Acc: {}, F1 Score: {}'.format(acc, f1) + '\n\n')
        f.write("====================================================" + '\n')
        f.write('                Decision on Average                 ' + '\n')
        f.write("====================================================" + '\n')
        f.write('Total Test Acc: {}, F1 Score: {}, AUC: {}'.format(acc_avg, f1_avg, auc_score_avg))


def create_confusion_matrix(base_model, method, cf_matrix):
    group_names = ['TrueNeg', 'FalsePos', 'FalseNeg', 'TruePos']
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0: 0.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    os.chdir('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Evaluating/')
    plt.savefig(base_model + '_' + method + '.svg')


if __name__ == '__main__':
    run()
