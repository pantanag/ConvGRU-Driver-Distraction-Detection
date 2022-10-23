import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import torchvision
from collections import Counter
from sklearn.utils import class_weight
import random
import math
import matplotlib.pyplot as plt


def create_video_sample(path, label, tester, action, sample_duration, start_frame, end_frame, is_looped):
    sample = {
        'video_path': path,
        'label': label,
        'tester': tester,
        'action': action,
        'sample_duration': sample_duration,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'looped': is_looped
    }
    return sample


def get_data(root_path, subset, view, sample_duration, k=None, transform=None):
    dataset_info = []
    if transform is None:
        print('Loading {} Data'.format(subset.capitalize()))
    else:
        print('Loading {} Train Data'.format(transform))
    if subset == 'train':
        foldersList = [folder for folder in os.listdir(root_path) if 'Tester' in folder]
        for folder in foldersList:
            normalFolders = [root_path + folder + '/' + name for name in os.listdir(root_path+folder) if 'normal' in name]
            create_videos_from_frames(dataset_info, normalFolders, view, 'normal', sample_duration)
            anormalFolders = [root_path + folder + '/' + name for name in os.listdir(root_path + folder) if 'normal' not in name]
            create_videos_from_frames(dataset_info, anormalFolders, view, 'abnormal', sample_duration)
    elif subset == 'test':
        if 'LABEL.csv' in os.listdir(root_path):
            import csv
            os.chdir(root_path)
            file = open('LABEL.csv')
            csvreader = csv.reader(file)
            for row in csvreader:
                if any(row):
                    for item in row:
                        if 'val' in item:
                            valFolder = item
                        if 'rec' in item:
                            recFolder = item
                    currPath = root_path + valFolder + '/' + recFolder + '/' + view
                    equal = True
                    startPart = int(row[2])
                    endPart = int(row[3])
                    if row[4] == 'A':
                        label = 'abnormal'
                    else:
                        label = 'normal'
                    length = endPart - startPart + 1
                    div = int(length / sample_duration)
                    if length % sample_duration != 0:
                        equal = False
                    # for i in range(div):
                    #     video_sample = create_video_sample(currPath, label, valFolder + '/' + recFolder, label, sample_duration,
                    #                                        startPart + i * sample_duration,
                    #                                        startPart + (i + 1) * sample_duration - 1, False)
                    #     dataset_info.append(video_sample)
                    # if not equal:
                    #     video_sample = create_video_sample(currPath, label, valFolder+'/'+recFolder, label, sample_duration, startPart + div*sample_duration, endPart, True)
                    #     dataset_info.append(video_sample)
                    for i in range((len(os.listdir(currPath + '_IR')) - sample_duration) + 1):
                        if i == 0:
                            temp = sample_duration
                        else:
                            temp = 0
                        video_sample = create_video_sample(currPath, label, valFolder+'/'+recFolder, label, sample_duration, i, temp - 1 + i * sample_duration, False)
                        dataset_info.append(video_sample)
        else:
            print("File LABEL.csv doesn't exist. Try running the programm again "
                  "after the file is in the root path of the Dataset")
    print('Finished Loading Data')
    if k is not None:
        ntalking = sum([1 for i in range(len(dataset_info)) if 'talking' in dataset_info[i]['action'] or 'messaging' in dataset_info[i]['action']])
        nnormal = sum([1 for i in range(len(dataset_info)) if dataset_info[i]['action'] == 'normal'])
        prob = [0.75/ntalking if ('talking' in dataset_info[index]['action'] or 'messaging' in dataset_info[index]['action']) else 0.25/nnormal if dataset_info[index]['action'] == 'normal' else 0. for index in range(len(dataset_info))]
        prob[0] += 1 - sum(prob)
        dataset_info = np.random.choice(dataset_info, replace=False, p=prob, size=k).tolist()
    return dataset_info


def create_videos_from_frames(dataset_info, folders, view, label, sample_duration):
    for folder in folders:
        currPath = folder + '/' + view
        folderName = folder.split('/')
        testerName = folderName[len(folderName) - 2]
        action = folderName[len(folderName) - 1]
        if 'normal' in action:
            action = 'normal'
        equal = True
        div = int(len(os.listdir(currPath + '_IR')) / sample_duration)
        # if len(os.listdir(currPath + '_IR')) % sample_duration != 0:
        #     equal = False
        # for i in range(div):
        #     video_sample = create_video_sample(currPath, label, testerName, action, sample_duration, i * sample_duration, (i + 1) * sample_duration - 1, False)
        #     dataset_info.append(video_sample)
        # if not equal:
        #     video_sample = create_video_sample(currPath, label, testerName, action, sample_duration, div * sample_duration, len(os.listdir(currPath + '_IR')) - 1, True)
        #     dataset_info.append(video_sample)
        for i in range((len(os.listdir(currPath + '_IR')) - sample_duration) + 1):
            if i == 0:
                temp = sample_duration
            else:
                temp = 0
            video_sample = create_video_sample(currPath, label, testerName, action, sample_duration, i, temp - 1 + i*sample_duration, False)
            dataset_info.append(video_sample)


def get_image(img_path):
    img = torch.FloatTensor(np.array(Image.open(img_path)))
    return img


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    scaled = (arr - arr.min()) / np.ptp(arr)
    return scaled


def get_video(video_path, start_frame, end_frame, sample_duration, looped):
    repeat = False
    remain_frames = 0
    if looped:
        remain_frames = sample_duration - (end_frame - start_frame + 1)
        sample_duration = end_frame - start_frame + 1
        repeat = True
    frames_IR = np.array([video_path + '_IR' + '/img_' + str(start_frame + i) + '.png' for i in range(sample_duration)])
    frames_IR = torch.unsqueeze(torch.stack([get_image(image) for image in frames_IR]), dim=1)
    frames_depth = np.array(
        [video_path + '_depth' + '/img_' + str(start_frame + i) + '.png' for i in range(sample_duration)])
    frames_depth = torch.unsqueeze(torch.stack([get_image(image) for image in frames_depth]), dim=1)
    video = torch.cat((frames_IR, frames_depth), dim=1)
    if repeat:
        og_len = video.size(0)
        for i in range(remain_frames):
            if og_len - 1 - i >= 0:
                video = torch.cat((video, torch.unsqueeze(video[og_len - 1 - i, :, :, :], dim=0)), dim=0)
            else:
                video = torch.cat((video, torch.unsqueeze(video[video.size(0) - 1, :, :, :], dim=0)), dim=0)
    return video


def sort_frames(frames):
    return sorted(frames, key=lambda image: int(image.replace('.', '_').split('_')[1]))


def strLabelToNumLabel(label):
    if label == 'normal':
        label = np.float32(0.)
    else:
        label = np.float32(1.)
    return label


def plot_video(video, save_path=None, save_name=None):
    fig, axs = plt.subplots(2, 4, figsize=(11., 6.))
    temp = torch.squeeze(video[0, :, 0, :, :])
    count = 0
    for k in range(8 // 4):
        for i in range(8//2):
            image = temp[count]
            image = image.detach().cpu().numpy()
            axs[k, i].imshow(image, cmap='gray')
            count += 1
    if save_path is not None:
        os.chdir(save_path)
        plt.savefig('./' + save_name + '.png')
    plt.show()


class DAD(data.Dataset):
    def __init__(self, data_path, subset, view, sample_duration, k=None, transform_list=None):
        if transform_list is not None:
            self.transform = torchvision.transforms.Compose(transform_list)
        if len(transform_list) > 2:
            transform = transform_list[2].__class__.__name__
        else:
            transform = None
        self.dataset = get_data(data_path, subset, view, sample_duration, k, transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        video = get_video(self.dataset[index]['video_path'], self.dataset[index]['start_frame'], self.dataset[index]['end_frame'], self.dataset[index]['sample_duration'], self.dataset[index]['looped'])
        temp_IR = []
        temp_depth = []
        for i in range(video.size(0)):
            frame_IR = torch.unsqueeze(video[i, 0, :, :], dim=0)
            frame_IR = self.transform(frame_IR)
            frame_depth = torch.unsqueeze(video[i, 1, :, :], dim=0)
            frame_depth = self.transform(frame_depth)
            temp_IR.append(frame_IR)
            temp_depth.append(frame_depth)
        temp_IR = torch.stack(tuple(temp_IR))
        temp_depth = torch.stack(tuple(temp_depth))
        video = torch.cat((temp_IR, temp_depth), dim=1)
        label = strLabelToNumLabel(self.dataset[index]['label'])
        return video, label

    def get_labels(self, subset_indices=None):
        if subset_indices is None:
            labels = [strLabelToNumLabel(self.dataset[index]['label']) for index in range(len(self.dataset))]
        else:
            labels = [strLabelToNumLabel(self.dataset[index]['label']) for index in subset_indices]
        return labels

    def get_pos_weight(self):
        labels = torch.Tensor([strLabelToNumLabel(self.dataset[index]['label']) for index in range(len(self.dataset))]).int()
        class_count = torch.bincount(labels).detach().cpu().tolist()
        return class_count[0], class_count[1]
