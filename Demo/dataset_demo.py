import torch
import torch.nn as nn
import os
import numpy as np
import csv
from PIL import Image
root_path = 'E:/DAD/'


def pil_loader(path):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: image data
    """
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            #return img.convert('RGB')
            return img.convert('L')


def get_video(video_path, frame_indices):
    """
    generate a video clip which is a list of selected frames
    :param video_path: path of video folder which contains video frames
    :param frame_indices: list of selected indices of frames. e.g. if index is 1, then selected frame's name is "img_1.png"
    :return: a list of selected frames which are PIL.Image or accimage form
    """
    image_reader = pil_loader
    video = []
    for image_index in frame_indices:
        image_name = 'img_' + str(image_index) + '.png'
        image_path = os.path.join(video_path, image_name)
        img = image_reader(image_path)
        video.append(img)
    return video


def get_clips(video_path, video_begin, video_end, label, view, sample_duration):
    """
    be used when validation set is generated. be used to divide a video interval into video clips
    :param video_path: validation data path
    :param video_begin: begin index of frames
    :param video_end: end index of frames
    :param label: 1(normal) / 0(anormal)
    :param view: front_depth / front_IR / top_depth / top_IR
    :param sample_duration: how many frames should one sample contain
    :return: a list which contains  validation video clips
    """
    clips = []
    sample = {
        'video': video_path,
        'label': label,
        'subset': 'validation',
        'view': view,
    }
    interval_len = (video_end - video_begin + 1)
    num = int(interval_len / sample_duration)
    for i in range(num):
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_begin + sample_duration))
        clips.append(sample_)
        video_begin += sample_duration
    if interval_len % sample_duration != 0:
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_end+1)) + [video_end] * (sample_duration - (video_end - video_begin + 1))
        clips.append(sample_)
    return clips


def get_data(val_folder, rec_folder, sample_duration, view):
    csv_path = root_path + 'LABEL.csv'
    dataset = []
    rec_folders = ['rec1', 'rec2', 'rec3', 'rec4', 'rec5', 'rec6']
    idx = rec_folders.index(rec_folder)
    if rec_folder == rec_folders[-1]:
        criterion = rec_folders[0]
    else:
        criterion = rec_folders[idx+1]
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[-1] == '':
                continue
            # if row[0] != '':
            if row[0] == val_folder:
                which_val_path = os.path.join(root_path, row[0].strip())
            # if row[1] != '':
            if row[1] == rec_folder and 'which_val_path' in locals():
                video_path = os.path.join(which_val_path, row[1], view)
            # elif row[1] != rec_folders[-1] and row[1] == rec_folders[idx + 1]:
            elif row[1] == criterion:
                if 'which_val_path' in locals() and 'video_path' in locals():
                    del which_val_path
                if 'video_path' in locals():
                    del video_path
            video_begin = int(row[2])
            video_end = int(row[3])
            if row[4] == 'N':
                label = np.float32(0.)
            elif row[4] == 'A':
                label = np.float32(1.)
            if 'video_path' in locals():
                clips = get_clips(video_path, video_begin, video_end, label, view, sample_duration)
                dataset = dataset + clips
    return dataset


class DAD_demo(torch.utils.data.Dataset):
    def __init__(self, val_folder, rec_folder, sample_duration, view, spatial_transform=None):
        self.data = get_data(val_folder, rec_folder, sample_duration, view)
        self.loader = get_video
        self.spatial_transform = spatial_transform

    def __getitem__(self, index):
        video_path = self.data[index]['video']
        ground_truth = self.data[index]['label']
        frame_indices = self.data[index]['frame_indices']

        clip = self.loader(video_path, frame_indices)

        # self.spatial_transform.randomize_parameters()
        clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, ground_truth

    def __len__(self):
        return len(self.data)
