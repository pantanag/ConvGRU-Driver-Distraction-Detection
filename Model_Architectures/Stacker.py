import torch.nn as nn
import torch
from Model_Architectures.FeatureExtractor import FeatureExtractor


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


class Stacker(nn.Module):
    def __init__(self):
        super(Stacker, self).__init__()
        self.front_IR_model = FeatureExtractor().cuda()
        self.front_depth_model = FeatureExtractor().cuda()
        self.top_IR_model = FeatureExtractor().cuda()
        self.top_depth_model = FeatureExtractor().cuda()
        self.front_IR_model = load_semi_model(self.front_IR_model, 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/front_IR/front_IR_v2.pth')
        self.front_IR_model.eval()
        self.front_depth_model = load_semi_model(self.front_depth_model, 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/front_depth/front_depth_v13.pth')
        self.front_depth_model.eval()
        self.top_IR_model = load_semi_model(self.top_IR_model, 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/top_IR/top_IR_v2.pth')
        self.top_IR_model.eval()
        self.top_depth_model = load_semi_model(self.top_depth_model, 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/top_depth/top_depth_v1.pth')
        self.top_depth_model.eval()
        self.stack = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, inputs_front_IR, inputs_front_depth, inputs_top_IR, inputs_top_depth):
        with torch.no_grad():
            features_front_IR = self.front_IR_model(inputs_front_IR)
            features_front_depth = self.front_depth_model(inputs_front_depth)
            features_top_IR = self.top_depth_model(inputs_top_IR)
            features_top_depth = self.top_IR_model(inputs_top_depth)
        features = torch.cat((features_front_IR, features_front_depth, features_top_IR, features_top_depth), 1)
        x = torch.squeeze(self.stack(features))
        return x

