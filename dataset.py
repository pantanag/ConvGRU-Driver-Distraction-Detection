import torch
import torch.utils.data as data
from PIL import Image
import os
import csv
import numpy as np


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


def accimage_loader(path):
    """
    compared with PIL, accimage loader eliminates useless function within class, so that it is faster than PIL
    :param path: image path
    :return: image data
    """
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    """
    choose accimage as image loader if it is available, PIL otherwise
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def get_video(video_path, frame_indices):
    """
    generate a video clip which is a list of selected frames
    :param video_path: path of video folder which contains video frames
    :param frame_indices: list of selected indices of frames. e.g. if index is 1, then selected frame's name is "img_1.png"
    :return: a list of selected frames which are PIL.Image or accimage form
    """
    image_reader = get_default_image_loader()
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


def listdir(path):
    """
    show every files or folders under the path folder
    """
    for f in os.listdir(path):
            yield f


def make_dataset(root_path, subset, view, sample_duration, type=None, nsamples=None):
    """
    :param nsamples: number of samples for augmentation
    :param root_path: root path of the dataset"
    :param subset: train / validation
    :param view: front_depth / front_IR / top_depth / top_IR
    :param sample_duration: how many frames should one sample contain
    :param type: during training process: type = normal / anormal ; during validation or test process: type = None
    :return: list of data samples, each sample is in form {'video':video_path, 'label': 0/1, 'subset': 'train'/'validation', 'view': 'front_depth' / 'front_IR' / 'top_depth' / 'top_IR', 'action': 'normal' / other anormal actions}
    """
    dataset = []
    if subset == 'train' and type == 'normal':
        # load normal training data
        train_folder_list = list(filter(lambda string: string.find('Tester') != -1, list(listdir(root_path))))

        for train_folder in train_folder_list:
            normal_video_list = list(filter(lambda string: string.split('_')[0] == 'normal', list(listdir(os.path.join(root_path, train_folder)))))

            for normal_video in normal_video_list:
                video_path = os.path.join(root_path, train_folder, normal_video, view)
                if not os.path.exists(video_path):
                    print(f"Video path doesn't exit: {video_path}")
                    continue

                n_frames = len(os.listdir(video_path))
                if n_frames <= 0:
                    print(f"Path {video_path} does't contain any data")
                    continue

                sample = {
                    'video': video_path,
                    'label': np.float32(0.),
                    'subset': 'train',
                    'view': view,
                    'action': 'normal'
                }
                for i in range(0, n_frames, sample_duration):
                    sample_ = sample.copy()
                    sample_['frame_indices'] = list(range(i, min(n_frames, i + sample_duration)))
                    if len(sample_['frame_indices']) < sample_duration:
                        for j in range(sample_duration-len(sample_['frame_indices'])):
                            sample_['frame_indices'].append(sample_['frame_indices'][-1])
                    dataset.append(sample_)

    elif subset == 'train' and type == 'anormal':
        #load anormal training data
        train_folder_list = list(filter(lambda string: string.find('Tester') != -1, list(listdir(root_path))))

        for train_folder in train_folder_list:
            anormal_video_list = list(filter(lambda string: string.split('_')[0] != 'normal', list(listdir(os.path.join(root_path, train_folder)))))

            for anormal_video in anormal_video_list:
                video_path = os.path.join(root_path, train_folder, anormal_video, view)
                if not os.path.exists(video_path):
                    print(f"Video path doesn't exit: {video_path}")
                    continue
                n_frames = len(os.listdir(video_path))
                if n_frames <= 0:
                    print(f"Path {video_path} does't contain any data")
                    continue
                sample = {
                    'video': video_path,
                    'label': np.float32(1.),
                    'subset': 'train',
                    'view': view,
                    'action': anormal_video,
                }

                for i in range(0, n_frames, sample_duration):
                    sample_ = sample.copy()
                    sample_['frame_indices'] = list(range(i, min(n_frames, i + sample_duration)))
                    if len(sample_['frame_indices']) < sample_duration:
                        for j in range(sample_duration-len(sample_['frame_indices'])):
                            sample_['frame_indices'].append(sample_['frame_indices'][-1])

                    dataset.append(sample_)

    elif subset == 'validation' and type == None:
        #load valiation data as well as thier labels
        csv_path = root_path + 'LABEL.csv'
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[-1] == '':
                    continue
                if row[0] != '':
                    which_val_path = os.path.join(root_path, row[0].strip())
                if row[1] != '':
                    video_path = os.path.join(which_val_path, row[1], view)
                video_begin = int(row[2])
                video_end = int(row[3])
                if row[4] == 'N':
                    label = np.float32(0.)
                elif row[4] == 'A':
                    label = np.float32(1.)
                clips = get_clips(video_path, video_begin, video_end, label, view, sample_duration)
                dataset = dataset + clips
    else:
        print('!!!DATA LOADING FAILURE!!!CANT FIND CORRESPONDING DATA!!!PLEASE CHECK INPUT!!!')
    if nsamples is not None:
        ntalking = sum([1 for i in range(len(dataset)) if 'talking' in dataset[i]['action'] or 'messaging' in dataset[i]['action']])
        nnormal = sum([1 for i in range(len(dataset)) if dataset[i]['action'] == 'normal'])
        # prob = [0.65 / 45000 if ('talking' in dataset_info[index]['action'] or 'messaging' in dataset_info[index][
            # 'action']) else 0.35 / 250050 if dataset_info[index]['action'] == 'nornmal' else 0. for index in
            #     range(len(dataset_info))]
        prob = [0.75/ntalking if ('talking' in dataset[index]['action'] or 'messaging' in dataset[index]['action']) else 0.25/nnormal if dataset[index]['action'] == 'normal' else 0. for index in range(len(dataset))]
        prob[0] += 1 - sum(prob)
        dataset = np.random.choice(dataset, replace=False, p=prob, size=nsamples).tolist()
    return dataset


def get_labels_concat(concatDataset, subset_indices=None):
    """
        Function to get the labels of a concated Dataset.
            :param concatDataset:
            :param subset_indices:
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


class DAD(data.Dataset):
    """
    generate normal training/ anormal training/ validation dataset according to requirement
    """
    def __init__(self,
                 root_path,
                 subset,
                 view,
                 sample_duration=16,
                 type=None,
                 get_loader=get_video,
                 spatial_transform=None,
                 temporal_transform=None,
                 k=None):
        self.data = make_dataset(root_path, subset, view, sample_duration, type, k)
        self.sample_duration = sample_duration
        self.subset = subset
        self.loader = get_loader
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, index):
        if self.subset == 'train':
            video_path = self.data[index]['video']
            frame_indices = self.data[index]['frame_indices']
            label = self.data[index]['label']
            #print(frame_indices)
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            #print(frame_indices)
            clip = self.loader(video_path, frame_indices)

            # self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)                 #data with shape (channels, timesteps, height, width)
            return clip, label
        elif self.subset == 'validation':
            video_path = self.data[index]['video']
            ground_truth = self.data[index]['label']
            frame_indices = self.data[index]['frame_indices']

            clip = self.loader(video_path, frame_indices)

            # self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            return clip, ground_truth

        else:
            print('!!!DATA LOADING FAILURE!!!CANT FIND CORRESPONDING DATA!!!PLEASE CHECK INPUT!!!')

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        labels = [self.data[index]['label'] for index in range(len(self.data))]
        return labels

    def get_pos_weight(self):
        labels = torch.Tensor([(self.data[index]['label']) for index in range(len(self.data))]).int()
        class_count = torch.bincount(labels).detach().cpu().tolist()
        return class_count[0], class_count[1]
