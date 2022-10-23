import torch


old_threshold = 0.7
threshold = torch.tensor([-0.5])
if isinstance(old_threshold, torch.Tensor):
    old_threshold = old_threshold.item()
if isinstance(threshold, torch.Tensor):
    threshold = threshold.item()
print('Threshold changed from {} to {} for next epoch'.format(round(old_threshold, 5), round(threshold, 5)))