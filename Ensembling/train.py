import torch
import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc
import os
from tqdm import tqdm
from train_model import find_intersection
from stats import plot_dists_test
import shutil
project_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Ensembling/'


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def find_best_version(model, currF1Score):
    readPath = os.path.join(project_path, 'Saved_Models', model.__class__.__name__).replace('\\', '/')
    items = os.listdir(readPath)
    foundBest = False
    foundNewBest = False
    for item in items:
        if '(BEST)' in item:
            foundBest = True
            bestItem = item
    # Comparing with current best
    if foundBest:
        f = open(readPath + '/' + bestItem, 'r')
        for x in f:
            if 'F1 Score Test' in x:
                texts = x.split(':')
                f1Temp = float(texts[1].replace(' ', '').replace('\n', ''))
                if currF1Score > f1Temp:
                    foundNewBest = True
                    f.close()
                    os.rename(readPath + '/' + bestItem, readPath + '/' + bestItem.replace('_BEST', ''))
                    break
    # First run
    else:
        foundNewBest = True
    return foundNewBest


def create_version_info(version, model, desc, f1Score):
    """
        Function to write info about the model.

        Args:
            version: The current version we are testing, to keep info
            about the training phase and the model architecture (dtype= str)
            model_head: The name of the model we are currently training (dtype= str)
            desc: Description about the model and metrics in the training (dtype= str)
    """
    save_path = os.path.join(project_path, 'Saved_Models').replace("\\", "/")
    version = version.replace('.', '_').split('_')
    version = version[1]
    version_path = os.path.join(save_path, model.__class__.__name__).replace("\\", "/")
    if not os.path.exists(version_path):
        os.makedirs(version_path)
    os.chdir(version_path)
    if find_best_version(model, f1Score):
        version = version + '_(BEST)'
    with open(model.__class__.__name__ + '_v' + version + '.txt', 'w') as f:
        f.write(desc)
    f.close()


def save_model(model, version):
    """
        Function to save the model when its called.

        Args:
            model: The Model we want to save (dtype=PyTorch Model)
            version: The version of the training we are currently on (dtype= str)
            view: The view of the Dataset we have use (dtype= str)
            kfold: Optional value, if kfold cross-validation was used (dtype= bool)
    """
    path = os.path.join(project_path, 'Saved_Models').replace("\\", "/")

    savePath = os.path.join(path, model.__class__.__name__.replace("\\", "/"))
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    version = version.replace('.', '_').split('_')
    version = version[1]
    version = model.__class__.__name__ + '_v' + version + ".pth"
    os.chdir(savePath)
    torch.save(model.state_dict(), './' + version)


def copy_file(model, version):
    """
        Function to copy the architecture file to another location
        so that the main architecture can be altered as many times
        as we want.

        Args:
            model: The model we are currently testing, to get its name and architecture (dtype= PyTorch Model)
            version: The version of the experiment we are conducting (dtype= str)
            view: The view of the Dataset we use for training and testing (dtype= str)
            kfold: Optional value if we used kfold cross-validation (dtype= bool)

    """
    print('Saving Architecture of Model...')
    original = os.path.join(project_path.replace('Ensembling',''), 'Model_Architectures').replace("\\", "/")
    arch = os.path.join(original, model.__class__.__name__ + '.py').replace("\\", "/")
    path = os.path.join(project_path, 'Saved_Models').replace("\\", "/")
    savePath = os.path.join(path, model.__class__.__name__).replace("\\", "/")
    version = version.replace('.', '_').split('_')
    version = version[1]
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savePath = os.path.join(savePath, model.__class__.__name__ + '_v' + version + '.py').replace("\\", "/")
    shutil.copyfile(arch, savePath)
    print('Finished saving architecture of Model...')


def eval_model(stacker, eval_dataloaders, threshold, version, epoch_idx):
    stacker.eval()
    stacker.cuda()
    test_dl_front_IR, test_dl_front_depth, test_dl_top_IR, test_dl_top_depth = eval_dataloaders
    num_test_examples = 0
    num_test_correct = 0
    total_pred = np.empty(shape=(0,))
    total_true = np.empty(shape=(0,))
    total_outputs = np.empty(shape=(0,))
    negScores = np.empty(shape=(0,))
    posScores = np.empty(shape=(0,))
    with torch.no_grad():
        for data1, data2, data3, data4 in tqdm(zip(test_dl_front_IR, test_dl_front_depth, test_dl_top_IR, test_dl_top_depth), total=len(test_dl_front_IR), desc='Testing '):
            inputs_front_IR, labels = data1
            inputs_front_depth, _ = data2
            inputs_top_IR, _ = data3
            inputs_top_depth, _ = data4
            inputs_front_IR, inputs_front_depth, inputs_top_IR, inputs_top_depth = inputs_front_IR.cuda(), inputs_front_depth.cuda(), inputs_top_IR.cuda(), inputs_top_depth.cuda()
            labels = labels.cuda()
            outputs = stacker(inputs_front_IR, inputs_front_depth, inputs_top_IR, inputs_top_depth)
            pred = (outputs > threshold).float()
            num_test_correct += (pred == labels).sum().item()
            num_test_examples += inputs_front_IR.size(0)
            outputs = outputs.detach().cpu()  # .numpy()
            if outputs.ndim == 0:
                outputs = torch.unsqueeze(outputs, 0)
            tempNeg = outputs[labels == 0.].numpy()
            tempPos = outputs[labels == 1.].numpy()
            negScores = np.concatenate((negScores, tempNeg))
            posScores = np.concatenate((posScores, tempPos))
            total_pred = np.concatenate((total_pred, pred.detach().cpu()))
            total_true = np.concatenate((total_true, labels.detach().cpu()))
            total_outputs = np.concatenate((total_outputs, outputs))
    test_acc = 100 * num_test_correct / num_test_examples
    plot_dists_test(negScores, posScores, threshold, epoch_idx, os.path.join(project_path, 'Results', stacker.__class__.__name__, version, "Test Distributions").replace("\\", "/"), normalize=False)
    f1Score = f1_score(total_true, total_pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(total_true, sigmoid(total_outputs))
    auc_score = auc(false_positive_rate, true_positive_rate)
    return test_acc, f1Score, auc_score


def train(stacker, dataloaders, eval_dataloaders, epochs, optimizer, loss_fcn, threshold, version, desc):
    copy_file(stacker, version)
    train_dl_front_IR, train_dl_front_depth, train_dl_top_IR, train_dl_top_depth = dataloaders
    stacker.cuda()
    stacker.train()
    test_accuracies = []
    test_f1Scores = []
    test_aucScores = []
    patience = 10
    last_f1 = 0
    trigger_times = 0
    for epoch_idx in range(1, epochs + 1):
        num_train_correct = 0
        num_train_examples = 0
        total_pred = np.empty(shape=(0,))
        total_true = np.empty(shape=(0,))
        total_outputs = np.empty(shape=(0,))
        negScores = np.empty(shape=(0,))
        posScores = np.empty(shape=(0,))
        skip = False
        print('============================================= EPOCH {} ============================================='.format(epoch_idx))
        train_loss = 0.
        for data1, data2, data3, data4 in tqdm(zip(train_dl_front_IR, train_dl_front_depth, train_dl_top_IR, train_dl_top_depth), total=len(train_dl_front_IR), desc='Training '):
            inputs_front_IR, labels = data1
            inputs_front_depth, _ = data2
            inputs_top_IR, _ = data3
            inputs_top_depth, _ = data4
            inputs_front_IR, inputs_front_depth, inputs_top_IR, inputs_top_depth = inputs_front_IR.cuda(), inputs_front_depth.cuda(), inputs_top_IR.cuda(), inputs_top_depth.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = stacker(inputs_front_IR, inputs_front_depth, inputs_top_IR, inputs_top_depth)
            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()
            pred = (outputs > threshold).float()
            train_loss += loss.data.item() * inputs_front_IR.size(0)
            num_train_correct += (pred == labels).sum().item()
            num_train_examples += inputs_front_IR.size(0)
            outputs = outputs.detach().cpu()
            if outputs.ndim == 0:
                outputs = torch.unsqueeze(outputs, 0)
            tempNeg = outputs[labels == 0.].numpy()
            tempPos = outputs[labels == 1.].numpy()
            negScores = np.concatenate((negScores, tempNeg))
            posScores = np.concatenate((posScores, tempPos))
            total_outputs = np.concatenate((total_outputs, outputs))
            total_pred = np.concatenate((total_pred, pred.detach().cpu()))
            total_true = np.concatenate((total_true, labels.detach().cpu()))

        train_loss = train_loss / num_train_examples
        train_acc = 100 * num_train_correct / num_train_examples
        f1Score = f1_score(total_true, total_pred)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(total_true, sigmoid(total_outputs))
        auc_score = auc(false_positive_rate, true_positive_rate)

        if f1Score > 0.98:
            skip = True
        old_threshold = threshold
        if not skip:
            threshold = find_intersection(negScores, posScores, os.path.join(project_path, 'Results', stacker.__class__.__name__, version, "Score Distributions").replace("\\", "/"), epoch_idx=epoch_idx)
        test_acc, f1ScoreTest, auc_scoreTest = eval_model(stacker, eval_dataloaders, threshold, version, epoch_idx)
        test_accuracies.append(test_acc / 100)
        test_f1Scores.append(f1ScoreTest)
        test_aucScores.append(auc_scoreTest)
        if f1ScoreTest == max(test_f1Scores):
            best_test_acc = test_acc
            save_model(stacker, version)
            listOfStrings = [desc, 'Test Accuracy: {}'.format(best_test_acc), 'F1 Score Test: {}'.format(f1ScoreTest),
                             'AUC Score Test: {}'.format(auc_scoreTest), 'Optimal Threshold: {}'.format(threshold)]
            txt = "\n".join(listOfStrings)
            create_version_info(version, stacker, txt, f1ScoreTest)
        if f1ScoreTest < last_f1:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopped!')
                break

        else:
            print('trigger times: 0')
            trigger_times = 0
            last_f1 = f1ScoreTest
        if isinstance(old_threshold, torch.Tensor):
            old_threshold = old_threshold.item()
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.item()
        print('Train Accuracy: {}, Train Loss: {}, F1 Score: {}, AUC Score: {}'.format(train_acc, train_loss, f1Score, auc_score))
        print('Test Accuracy: {}, F1 Score Test: {}, AUC Score Test: {}'.format(test_acc, f1ScoreTest, auc_scoreTest))
        print('Threshold changed from {} to {} for next epoch'.format(round(old_threshold, 5), round(threshold, 5)))
