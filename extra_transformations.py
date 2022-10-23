import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision


def noisy(noise_type, image):
    if noise_type == "gauss":
        row, col, ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean ,sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
    elif noise_type == "s&p":
        black = 0
        white = 255
        image = np.squeeze(image)
        n_salt = 800
        x_cord = np.random.randint(0, image.shape[1], n_salt)
        y_cord = np.random.randint(0, image.shape[0], n_salt)
        coords = (y_cord, x_cord)
        image[coords] = white
        n_pepper = 800
        x_cord = np.random.randint(0, image.shape[1], n_pepper)
        y_cord = np.random.randint(0, image.shape[0], n_pepper)
        coords = (y_cord, x_cord)
        image[coords] = black
        image = np.expand_dims(image, axis=0)
        noisy = image.copy()
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
    elif noise_type =="speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
    return noisy


class AddNoise(object):
    def __init__(self, imgType, noise_type="gauss"):
        self.img_type = imgType
        self.noise_type = noise_type

    def __call__(self, image):
        if self.img_type == 'Tensor':
            image = image.cpu().detach().numpy()
        elif self.img_type == 'PIL':
            image = np.array(image)
        elif self.img_type == 'ndarray':
            pass
        image = noisy(self.noise_type, image)
        t = torchvision.transforms.ToTensor()
        image = t(image)
        image = torch.permute(image, (1, 2, 0))
        return image


class Brightness(object):
    def __init__(self, factor=1.):
        super(Brightness, self).__init__()
        self.factor = factor

    def __call__(self, img):
        return F.adjust_brightness(img, self.factor)


class ReverseFrames(object):
    def __init__(self):
        super(ReverseFrames, self).__init__()

    def __call__(self, clip):
        clip = list(reversed(clip))
        return clip


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
