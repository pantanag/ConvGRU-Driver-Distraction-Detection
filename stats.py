import torch
import matplotlib.pyplot as plt
import scipy.stats as st
import time
import numpy as np
import stopit
from tqdm import tqdm
import warnings
import os
from shapely.geometry import LineString
from matplotlib.patches import Patch


def center_text(listTexts, element):
    """
        Function to center a string for printing according to
        a list of strings.

        Args:
             listTexts: List of all possible arrays to print (dtype= List of str)
             element: The string we want to print (must be
             a member of the previous list) (dtype= str)
    """
    maxLen = len(max(listTexts, key=len))
    elementLen = len(element)
    diff = maxLen - elementLen
    if diff % 2 == 0:
        left = diff // 2
        right = diff // 2
    else:
        left = diff // 2
        right = diff // 2 + 1
    leftStr = ''
    rightStr = ''
    for i in range(left):
        leftStr += ' '
    for k in range(right):
        rightStr += ' '
    finalString = leftStr + element + rightStr
    return finalString


def normalize_tensor(vector):
    '''
        Function to normalize Tensor.

        Args:
            vector: Tensor of scores to be normalized (dtype= Tensor)
    '''
    min_v = torch.min(vector)
    range_v = torch.max(vector) - min_v
    if range_v > 0:
        normalised = (vector - min_v) / range_v
    else:
        normalised = torch.zeros(vector.size())
    return normalised


def find_nearest(array, value):
    '''
        Function to find the index of an item in an array whose value is
        closest to the 'value' parameter.

        Args:
            array: Array like to search for.
            value: The value to search in the array and find its closest.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_smallest_dist(x_arr, y_arr, xy):
    '''
        Function to find the the index of a function (given as values)
        whose value is closest to the one given.

        Args:
             x_arr: Array of the x-values of the function (dtype= Tensor)
             y_arr: Array of the y_values of the function (dtype= Tensor)
             xy: The point (x and y value) to find the smallest distance (dtype= List)
    '''
    x_th, y_th = xy
    dist = np.sqrt(np.power(x_arr - x_th, 2) + np.power(y_arr - y_th, 2))
    return np.argmin(dist)


@stopit.threading_timeoutable(default='not finished')
def fit_dist(data, dist_name, params, dist_results, t, fitting_dist_texts):
    """
        Function to fit specific distribution to data.

        Args:
             data: Scores to be fitted in given distribution (dtype= Tensor)
             dist_name: Name of distribution to fit (dtype= String)
             params: Dictionary to store parameters of fitting to use this
             function multiple times (dtype= Dictionary)
             dist_results: List to append tried distribution with p-value as
             tuple (dtype= List)
             t: tqdm bar to update description with p-values (dtype= tqdm bar)
             fitting_dist_texts: List of strings for all possible descriptions
             (dtype= List of str)
    """
    dist = getattr(st, dist_name)
    param = dist.fit(data)
    t.set_description(center_text(fitting_dist_texts, 'Fitting {}'.format(dist_name)), refresh=True)
    time.sleep(0.4)
    params[dist_name] = param
    # Applying the Kolmogorov-Smirnov test
    D, p = st.kstest(data, dist_name, args=param)
    t.set_description(center_text(fitting_dist_texts, '{} | p-value= {:.3e}'.format(dist_name, p)), refresh=True)
    time.sleep(0.4)
    dist_results.append((dist_name, p))


def get_best_fitting_dist(data):
    """
        Function to find the best fitting distribution to scores based on the
        highest p-value (Kolmogorov-Smirnov test).

        Args:
            data: Scores to be fitted in the below distributions (dtype= Tensor)
    """
    dist_names = ['norm', 'lognorm', 'argus',
                  'alpha', 'gamma', 'dgamma', 'levy', 'laplace', 'burr', 'burr12',
                  'rice', 'pearson3', 'wald', 'beta', 'loggamma', 'cauchy', 'chi',
                  'laplace', 'chi2', 'levy', 'logistic', 'loglaplace', 'erlang',
                  'maxwell', 'weibull_min', 'exponnorm', 'expon', 'exponweib', 'exponpow',
                  'f', 'fisk', 'genlogistic', 'gennorm', 'genpareto', 'gausshyper', 'genextreme',
                  'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic',
                  'halfgennorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb',
                  'johnsonsu', 'kappa4', 'kappa3', 'levy_l', 'laplace_asymmetric', 'loguniform', 'lomax',
                  'mielke', 'moyal', 'nakagami', 'powernorm', 'rdist', 'rice', 'recipinvgauss',
                  'skewnorm', 't']
    fitting_dist_texts = ['Fitting {}'.format(name) for name in dist_names]
    p_values_texts = ['{} | p-value= 0.000e-000'.format(name) for name in dist_names]
    fitting_dist_texts.extend(p_values_texts)
    dist_results = []
    params = {}
    t = tqdm(dist_names, desc='Fitting Distribution ', total=len(dist_names))
    for dist_name in t:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_dist(data, dist_name, params, dist_results, t, fitting_dist_texts, timeout=10)
        except Exception:
            t.set_description('Discarded {}'.format(dist_name))
    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    print('Best Fitted Distribution: {}'.format(best_dist))
    # store the name of the best fit and its p value
    paramsBest = list(params[best_dist])
    x = np.linspace(min(data), max(data), 1000)
    dist = getattr(st, str(best_dist))
    pdf = dist.pdf(x, *paramsBest)
    return pdf, x


def find_optimal_nbins(x):
    '''
        Function to find the optimal number of bins to
        make histogram of data.

        Args:
             x: data (dtype= Tensor)
    '''
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = (x.max() - x.min()) / bin_width
    if np.isnan(bins):
        bins = 200
    else:
        bins = round(bins)
    return bins


def normalize_data(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))


def plot_dists_test(negScores, posScores, threshold, epoch_idx, savePath, normalize=True):
    if normalize:
        negScores = normalize_data(negScores)
        posScores = normalize_data(posScores)
    plt.figure(figsize=(10., 8.), dpi=100)
    _, binsNeg, _ = plt.hist(negScores, density=True, bins=find_optimal_nbins(negScores), alpha=0.5, label='Normal Distribution')
    plt.hist(posScores, density=True, bins=binsNeg, alpha=0.5, label='Anomaly Distribution')
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.detach().cpu().item()
    plt.axvline(threshold)
    plt.grid()
    plt.legend(fontsize=7)
    plt.title('Test Scores Distributions')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    os.chdir(savePath)
    plt.savefig('Epoch_' + str(epoch_idx) + '.svg')
    plt.show()


def create_distributions_plot(posScores, negScores, og_threshold, epoch_idx, savePath, normalize=True):
    '''
        Function to create plots of the distributions and plots of the best fitted ones.

        Args:
             posScores: Scores produced by model for the positive class (dtype= Tensor)
             negScores: Scores produced by model for the negative class (dtype= Tensor)
             og_ threshold: Original Threshold for the model (Binary Classification)
             epoch_idx: The index of the current epoch of training, for saving purposes (dtype= int)
             savePath: The path to save the combined plot (dtype= String)
             normalize: Boolean value to determine if scores should be normalized or not to produce
             the new thrsehold, default is False (dtype= bool)
    '''
    if normalize:
        posScores = normalize_data(posScores)
        negScores = normalize_data(negScores)
    else:
        pass
    pdfPos, xPos = get_best_fitting_dist(posScores)
    pdfNeg, xNeg = get_best_fitting_dist(negScores)
    f = plt.figure(figsize=(10., 4.))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.hist(posScores, density=True, bins=find_optimal_nbins(posScores), alpha=0.55, label='Normal Distribution')
    ax.hist(negScores, density=True, bins=find_optimal_nbins(negScores), alpha=0.55, label='Anomaly Distribution')
    ax.set_title('Scores')
    ax.axis(ymax=20)
    ax.grid()
    ax.legend(fontsize=7)

    idx_og_pos = find_nearest(xPos, og_threshold)
    idx_og_neg = find_nearest(xNeg, og_threshold)
    first_line = LineString(np.column_stack((xPos, pdfPos)))
    second_line = LineString(np.column_stack((xNeg, pdfNeg)))
    intersection = first_line.intersection(second_line)

    if intersection.geom_type == 'MultiPoint':
        x, y = LineString(intersection).xy
        idx_y = np.argmax(y)
        y = max(y)
        threshold = x[idx_y]
    elif intersection.geom_type == 'Point':
        threshold, y = intersection.xy
        threshold = threshold.tolist()[0]
        y = y.tolist()[0]
    else:
        if xPos[-1] > xNeg[-1]:
            threshold = xPos[0]
        else:
            threshold = xNeg[0]
        neg_idx = find_nearest(xNeg, threshold)
        pos_idx = find_nearest(xPos, threshold)
        y = min([pdfPos[pos_idx], pdfNeg[neg_idx]])

    idx_pos = find_smallest_dist(xPos, pdfPos, [threshold, y])
    idx_neg = find_smallest_dist(xNeg, pdfNeg, [threshold, y])
    ax2.plot(xPos[:idx_pos], pdfPos[:idx_pos], color='royalblue')
    ax2.plot(xNeg[idx_neg:], pdfNeg[idx_neg:], 'firebrick')
    ax2.fill_between(xPos, pdfPos, step='pre', alpha=0.55, label='Normal Distribution', color='royalblue')
    ax2.fill_between(xNeg, pdfNeg, step='pre', alpha=0.55, label='Anomaly Distribution', color='firebrick')
    ax2.fill_between(xPos[idx_og_pos:], pdfPos[idx_og_pos:], step='pre', alpha=0.25, hatch='xx', label='False Normal',
                     facecolor='darkblue', edgecolor='black')
    ax2.fill_between(xNeg[:idx_og_neg], pdfNeg[:idx_og_neg], step='pre', alpha=0.25, hatch='xx', label='False Anomaly',
                     facecolor='darkred', edgecolor='black')
    ax2.axvline(x=og_threshold, color='black')
    ax2.text(og_threshold, (30 + 0.05) / 2, 'Original Threshold', ha='center', va='bottom', rotation=90,
             rotation_mode='anchor', transform_rotates_text=True, fontsize=10, color='goldenrod')
    ax2.scatter(threshold, y, color='goldenrod', label='Improved Threshold', zorder=5)
    ax2.set_title('Fitting Distributions')
    ax2.axis(ymin=0)
    ax2.axis(ymax=20)
    ax2.legend(fontsize=7)
    ax2.grid()
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    os.chdir(savePath)
    plt.savefig('Epoch_' + str(epoch_idx) + '.png', dpi=100)
    plt.show()
    plt.close('all')
    return threshold


def inter_arrays(values1, values2):
    if values1.shape[0] > values2.shape[0]:
        temp = values1
        temp2 = values2
    else:
        temp = values2
        temp2 = values1
    smallest_diff = 100
    for i in range(temp.shape[0]):
        for k in range(temp2.shape[0]):
            if abs(temp[i] - temp2[k]) < smallest_diff:
                smallest_diff = abs(temp[i] - temp2[k])
                indexes = [i, k]
    threshold = (temp[indexes[0]] + temp2[indexes[1]])/2
    return threshold


def find_intersection(negScores, posScores, savePath, epoch_idx):
    plt.figure(figsize=(10., 8.))
    valuesNeg, edgesNeg, patchesNeg = plt.hist(negScores, bins=find_optimal_nbins(negScores), alpha=0.5, density=True, color='royalblue')
    valuesPos, edgesPos, patchesPos = plt.hist(posScores, bins=edgesNeg, alpha=0.5, density=True, color='firebrick')
    indexNeg = np.digitize(negScores, edgesNeg)
    indexPos = np.digitize(posScores, edgesPos)
    assert valuesNeg.shape == valuesPos.shape
    maxSum = 0
    for i in range(valuesNeg.shape[0]):
        if i != valuesNeg.shape[0]-1:
            if valuesNeg[i] >= valuesPos[i] and valuesNeg[i+1] <= valuesPos[i+1] and (valuesNeg[i] + valuesPos[i+1]) > maxSum:
                if negScores[indexNeg == i].size != 0 and posScores[indexPos == i + 2].size != 0:
                    iNeg = i
                    iPos = i + 1
                    maxSum = valuesNeg[i] + valuesPos[i+1]
    newvaluesNeg = negScores[indexNeg == iNeg]
    newvaluesPos = posScores[indexPos == iPos]
    threshold = inter_arrays(newvaluesNeg, newvaluesPos)
    plt.axvline(threshold, color='black')
    patch = Patch(facecolor='darkblue')
    patch2 = Patch(facecolor='darkred')
    plt.legend(fontsize=8, handles=[patchesNeg, patchesPos, patch, patch2], labels=['Normal Distribution', 'Anomaly Distribution', 'False Normal', 'False Anomaly'])
    plt.axis(ymin=0.)
    plt.grid()
    plt.title('Train Scores Distributions')
    plt.xlabel('Logits')
    # plt.text(threshold, (12 + 0.05) / 2, 'Improved Threshold', ha='center', va='bottom', rotation=90,
    #          rotation_mode='anchor', transform_rotates_text=True, fontsize=10, color='darkslategrey', weight='bold')
    for i in range(valuesPos.shape[0] - iNeg):
        patchesNeg[iNeg + i].set_color('darkblue')
        patchesNeg[iNeg + i].set_alpha(0.9)
        patchesNeg[iNeg + i].set(zorder=5)
    for i in range(iPos):
        patchesPos[iPos - 1 - i].set_color('darkred')
        patchesPos[iPos - 1 - i].set_alpha(0.9)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    os.chdir(savePath)
    plt.savefig('Epoch_' + str(epoch_idx) + '.svg')
    plt.show()
    threshold = torch.tensor([threshold], device='cuda')
    return threshold
