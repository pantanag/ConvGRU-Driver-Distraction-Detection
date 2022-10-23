import torch
from stats import plot_dists_test
from Model_Architectures.ConvGRUv4 import ConvGRUv4
from Model_Architectures.shuffleGRU import shuffleGRU
from Model_Architectures.resnextGRU import ResNextGRU
from Optimizers.adabound import AdaBound
from torch.optim import Adadelta, Adam, SGD
from Optimizers.AdaBelief import AdaBelief
from Optimizers.cosangulargrad import cosangulargrad
from Optimizers.tanangulargrad import tanangulargrad
from torch.optim import RMSprop
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import logging
import warnings
from tqdm import tqdm
from stats import create_distributions_plot, find_intersection
from torchinfo import summary
import numpy as np
import shutil
from sklearn.metrics import roc_curve, auc
from Model_Architectures.mobileGRU import mobileGRU
from torch.utils.data import WeightedRandomSampler
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
project_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4'


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def train_per_epoch(model, optimizer, loss_fcn, train_dl, epoch_idx, threshold):
    """
        Function to train the model per epoch.

        Args:
             model: The model we want to train (dtype= PyTorch Model)
             optimizer: The optimizer we use for the training (dtype= PyTorch Optimizer)
             loss_fcn: The loss function we use for training (dtype= PyTorch Loss Function)
             train_dl: DataLoader for the Train Data (dtype= DataLoader)
             epoch_idx: Index to keep track of epochs (dtype= int)
             threshold: The threshold we use for the training phase (dtype= float)
    """
    model.train()
    train_loss = 0
    num_train_examples = 0
    num_train_correct = 0
    total_pred = np.empty(shape=(0,))
    total_true = np.empty(shape=(0,))
    total_outputs = np.empty(shape=(0,))
    negScores = np.empty(shape=(0,))
    posScores = np.empty(shape=(0,))
    scaler = torch.cuda.amp.GradScaler()
    for i, data in tqdm(enumerate(train_dl), total=len(train_dl), desc='Training '):
        optimizer.zero_grad()
        inputs, labels = data[0].cuda(), data[1].cuda()
        outputs = model(inputs)
        with torch.cuda.amp.autocast():
            loss = loss_fcn(outputs, labels)
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        pred = (outputs > threshold).float()
        train_loss += loss.data.item() * inputs.size(0)
        num_train_correct += (pred == labels).sum().item()
        num_train_examples += inputs.size(0)
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

    # og_threshold = threshold
    train_loss = train_loss / num_train_examples
    train_acc = 100 * num_train_correct / num_train_examples
    f1Score = f1_score(total_true, total_pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(total_true, sigmoid(total_outputs))
    auc_score = auc(false_positive_rate, true_positive_rate)
    return train_loss, train_acc, f1Score, negScores, posScores, auc_score


def test_model(model, test_dl, threshold, epoch_idx, model_head, version, view):
    """
        Function to test the model on the Test Data.

        Args:
             model: The Model we are gonna test (dtype= PyTorch Model)
             test_dl: DataLoader for the Test Data (dtype= PyTorch Data Loader)
             threshold: The threshold we use for the test phase (dtype= float)
    """
    model.eval()
    num_test_examples = 0
    num_test_correct = 0
    total_pred = np.empty(shape=(0,))
    total_true = np.empty(shape=(0,))
    total_outputs = np.empty(shape=(0,))
    negScores = np.empty(shape=(0,))
    posScores = np.empty(shape=(0,))
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dl), total=len(test_dl), desc='Testing '):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            pred = (outputs > threshold).float()
            num_test_correct += (pred == labels).sum().item()
            num_test_examples += inputs.size(0)
            outputs = outputs.detach().cpu()#.numpy()
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
    plot_dists_test(negScores, posScores, threshold, epoch_idx, os.path.join(project_path,'Results', model_head, version, view, "Test Distributions").replace("\\", "/"), normalize=False)
    f1Score = f1_score(total_true, total_pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(total_true, sigmoid(total_outputs))
    auc_score = auc(false_positive_rate, true_positive_rate)
    return test_acc, f1Score, auc_score


def find_best_version(model_head, view, currF1Score):
    readPath = os.path.join(project_path, 'Saved_Models', model_head, view).replace('\\', '/')
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


def create_version_info(version, model_head, view, desc, f1Score):
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
    version_path = os.path.join(save_path, model_head, view).replace("\\", "/")
    if not os.path.exists(version_path):
        os.makedirs(version_path)
    os.chdir(version_path)
    if find_best_version(model_head, view, f1Score):
        version = version + '_(BEST)'
    with open(view + '_v' + version + '.txt', 'w') as f:
        f.write(desc)
    f.close()


def create_graph(model_head, version, y_points, y_label, x_label, n_epochs, view, legend_pos='upper right', annot_step=2, offset=0.05, test=None):
    """
        Function to create a graph for the metrics of the training.

        Args:
            model_head: Name of the Model (dtype= str)
            version: The current version we are testing, to keep info and track
            about the training phase and model architecture (dtype= str)
            y_points: The values of the metrics we want to plot (dtype= List)
            y_label: Name of the metric (dtype= str)
            x_label: Name for the x-axis (dtype= str)
            n_epochs: Total number of epochs of the training to construct the x-axis
            view: View of the Dataset folder, for the save path (dtype= int)
            legend_pos: Determines the position of the legend box, search
            matplotlib for more options (dtype= str)
            annot_step: Optional step for value printing in the plots (dtype= int)
            offset: Optional value to determine the distance between the points
            and their printed values (dtype= float)
            test: Optional value to include a final test metric alongside the
            training metric (dtype= float)
            fold: Optional value if kfold cross-validation is used (dtype= int)
    """
    results_path = os.path.join(project_path, 'Results').replace("\\", "/")
    savePath = os.path.join(results_path, model_head, version, view).replace("\\", "/")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    os.chdir(savePath)
    plt.plot(range(1, n_epochs + 1), y_points, label='Train')
    if test is not None:
        plt.plot(n_epochs, test, marker='o', label='Test')
        plt.annotate(str(round(test, 3)), xy=(n_epochs, test))
    for i, j in zip(range(1, n_epochs + 1), y_points):
        if isinstance(j, torch.Tensor):
            j = j.cpu().item()
        if i % annot_step == 0 and i != n_epochs:
            plt.annotate(str(round(j, 2)), xy=(i, j + offset))
    plt.legend(loc=legend_pos)
    plt.xlim(1, n_epochs + 0.2)
    plt.ylim(0, 1.05)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid()
    plt.savefig(y_label + '.png')
    plt.show()
    plt.close('all')


def create_model(model_head):
    """
        Function to create the model based on the model_head parameter.

        Args:
            model_head: Parameter to construct the model based on the available architectures (dtype= str)
    """
    if model_head == 'ConvGRUv4':
        model = ConvGRUv4()
    elif model_head == 'shuffleGRU':
        model = shuffleGRU()
    elif model_head == 'resnextGRU':
        model = ResNextGRU()
    elif model_head == 'mobileGRU':
        model = mobileGRU()
    summary(model, input_size=(1, 1, 16, 160, 160))
    return model


def create_optimizer(optim_head, model, lr, weight_decay=None):
    """
        Function to create optimizer from the parameter optim_head.

        Args:
            optim_head: Parameter to construct optimizer from it (dtype= str)
            model: The model we are gonna train, to get its parameters (dtype= PyTorch Model)
            lr: Learning Rate for the optimizer (dtype= float)
            weight_decay: Optional value to insert weight decay to the optimizer (dtype= float)
    """
    optim_head = optim_head.lower()
    if optim_head == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_head == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    elif optim_head == 'adabound':
        optimizer = AdaBound(model.parameters(), lr=lr, weight_decay=weight_decay, amsbound=True, final_lr=5e-5)
    elif optim_head == 'adadelta':
        optimizer = Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_head == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim_head == 'cosangulargrad':
        optimizer = cosangulargrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_head == 'tanangulargrad':
        optimizer = tanangulargrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_head == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def save_model(model, version, view):
    """
        Function to save the model when its called.

        Args:
            model: The Model we want to save (dtype=PyTorch Model)
            version: The version of the training we are currently on (dtype= str)
            view: The view of the Dataset we have use (dtype= str)
            kfold: Optional value, if kfold cross-validation was used (dtype= bool)
    """
    path = os.path.join(project_path, 'Saved_Models').replace("\\", "/")

    savePath = os.path.join(path, model.__class__.__name__, view).replace("\\", "/")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    version = version.replace('.', '_').split('_')
    version = version[1]
    version = view + '_v' + version + ".pth"
    os.chdir(savePath)
    torch.save(model.state_dict(), './' + version)


def copy_file(model, version, view):
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
    original = os.path.join(project_path, 'Model_Architectures').replace("\\", "/")
    arch = os.path.join(original, model.__class__.__name__ + '.py').replace("\\", "/")
    path = os.path.join(project_path, 'Saved_Models').replace("\\", "/")
    savePath = os.path.join(path, model.__class__.__name__, view).replace("\\", "/")
    version = version.replace('.', '_').split('_')
    version = version[1]
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savePath = os.path.join(savePath, view + '_v' + version + '.py').replace("\\", "/")
    shutil.copyfile(arch, savePath)
    print('Finished saving architecture of Model...')


def get_labels_concat(concatDataset, subset_indices=None):
    """
        Function to get the labels of a concated Dataset.

        Args:
            concatDataset: The Concated Dataset we want the labels
    """
    total_labels = []
    for i in range(len(concatDataset.datasets)):
        dataset = concatDataset.datasets[i]
        labels = dataset.get_labels()
        total_labels.extend(labels)
    if subset_indices is not None:
        partial = [total_labels[x] for x in subset_indices]
        return partial
    return total_labels


def class_imbalance_sampler(labels):
    class_count = torch.bincount(labels)
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels))
    return sampler


def train(model_head, optim_head, lr, loss_fcn, n_epochs, train_ds, test_ds, batch_size, n_workers, version, desc, view,
          weights=None, weight_decay=None, use_cuda=True):
    # weights = get_weights_concat(train_ds)
    # train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=n_workers, pin_memory=True)
    # labels = get_labels_concat(train_ds)
    # labels = [int(label) for label in labels]
    # labels = torch.tensor(labels)
    # sampler = class_imbalance_sampler(labels)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    model, _, _ = run_train(model_head=model_head, optim_head=optim_head, lr=lr, weight_decay=weight_decay,
                            train_dl=train_dl, test_dl=test_dl, loss_fcn=loss_fcn,
                            n_epochs=n_epochs, view=view, version=version, desc=desc, use_cuda=use_cuda)


def run_train(model_head, optim_head, lr, weight_decay, train_dl, test_dl, loss_fcn, n_epochs, view, version, desc,
              use_cuda, fold=None):
    model = create_model(model_head)
    copy_file(model, version, view)
    if use_cuda:
        model.cuda()
    if weight_decay is not None:
        optimizer = create_optimizer(optim_head, model, lr, weight_decay)
    else:
        optimizer = create_optimizer(optim_head, model, lr)
    accuracies = []
    test_accuracies = []
    losses = []
    f1Scores = []
    test_f1Scores = []
    aucScores = []
    test_aucScores = []
    threshold = torch.tensor([0.], device='cuda')
    thresholds = [threshold]
    # Early stopping
    last_f1 = 0
    patience = 10
    trigger_times = 0
    skip = False
    for epoch_idx in range(1, n_epochs + 1):
        print(
            '============================================= EPOCH {} ============================================='.format(
                epoch_idx))
        train_loss, train_acc, f1Score, negScores, posScores, auc_score = train_per_epoch(model, optimizer, loss_fcn, train_dl, epoch_idx, threshold=threshold)
        if f1Score > 0.98:
            skip = True
        if not skip:
            new_thres = find_intersection(negScores, posScores, os.path.join(project_path, 'Results', model_head, version, view, "Score Distributions").replace("\\", "/"), epoch_idx=epoch_idx)
        thresholds.append(new_thres)
        accuracies.append(train_acc / 100)
        losses.append(train_loss)
        f1Scores.append(f1Score)
        aucScores.append(auc_score)
        old_threshold = threshold
        threshold = new_thres
        test_acc, f1ScoreTest, auc_scoreTest = test_model(model, test_dl, threshold, epoch_idx, model_head, version, view)
        test_accuracies.append(test_acc / 100)
        test_f1Scores.append(f1ScoreTest)
        test_aucScores.append(auc_scoreTest)
        if f1ScoreTest == max(test_f1Scores):
            best_test_acc = test_acc
            save_model(model, version, view)
            listOfStrings = [desc, 'Test Accuracy: {}'.format(best_test_acc), 'F1 Score Test: {}'.format(f1ScoreTest), 'AUC Score Test: {}'.format(auc_scoreTest), 'Optimal Threshold: {}'.format(threshold)]
            txt = "\n".join(listOfStrings)
            create_version_info(version, model.__class__.__name__, view, txt, f1ScoreTest)
        if f1ScoreTest < last_f1:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopped!')
                n_epochs = epoch_idx
                break

        else:
            print('trigger times: 0')
            trigger_times = 0
            last_f1 = f1ScoreTest

        print('Train Accuracy: {}, Train Loss: {}, F1 Score: {}, AUC Score: {}'.format(train_acc, train_loss, f1Score, auc_score))
        print('Test Accuracy: {}, F1 Score Test: {}, AUC Score Test: {}'.format(test_acc, f1ScoreTest, auc_scoreTest))
        print('Threshold changed from {} to {} for next epoch'.format(round(old_threshold.item(), 5), round(new_thres.item(), 5)))
        torch.cuda.empty_cache()

    thresholds.pop()
    thresholds = [item.cpu() for item in thresholds]
    create_graph(model_head, version, accuracies, 'Accuracy', 'Epochs', n_epochs, view, test=test_acc, legend_pos='lower right', offset=-0.05, annot_step=5)
    create_graph(model_head, version, losses, 'Train Loss', 'Epochs', n_epochs, view, offset=+0.05, annot_step=5)
    create_graph(model_head, version, f1Scores, 'F1 Score', 'Epochs', n_epochs, view, legend_pos='lower right', offset=-0.05, annot_step=5)
    create_graph(model_head, version, thresholds, 'Thresholds', 'Epochs', n_epochs, view, offset=+0.05)
    create_graph(model_head, version, aucScores, 'AUC Score', 'Epochs', n_epochs, view, offset=-0.05, legend_pos='lower right', annot_step=5)
    create_graph(model_head, version, test_accuracies, 'Test Accuracy', 'Epochs', n_epochs, view, offset=+0.02, legend_pos='lower right')
    create_graph(model_head, version, test_f1Scores, 'F1 Score Test', 'Epochs', n_epochs, view, offset=+0.02, legend_pos='lower right')
    create_graph(model_head, version, test_aucScores, 'AUC Score Test', 'Epochs', n_epochs, view, offset=+0.02, legend_pos='lower right')
    return model, test_acc, f1ScoreTest
