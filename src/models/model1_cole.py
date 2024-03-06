# Full resolution 3D model
# J. H. Cole et al., “Predicting brain age with deep learning from raw imaging data results in a reliable and heritable
# biomarker,” NeuroImage, vol. 163, pp. 115–124, Dec. 2017, doi: 10.1016/j.neuroimage.2017.07.059.
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.data.build_dataset_functions import AtlasCrop
import src.config as config
import warnings
warnings.filterwarnings('ignore')

# %% Define training parameters
arg = {'model_name' : 'model1-cole-seed_42-110ep.tar',
       'random_seed': 42,
       'learning_rate': 0.0001,
       'lr_decay_percentage': 3,
       'lr_decrease_step': 1,
       'batch_size': 16,
       'p_pad': 0.3,
       'p_shift': 0.3,
       'p_flip': 0.5}

# set seed
torch.manual_seed(arg['random_seed'])
np.random.seed(arg['random_seed'])

# %% Define CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv3d: Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        # groups=1, bias=True, padding_mode='zeros') .... N C D H W = N 1 z y x

        # 5 blocks of:
        # - 3x3x3 conv layer stride=1 + ReLU
        # - 3x3x3 conv + batch normalization layer + ReLU
        # - 2x2x2 max pooling layer stride=2
        # One fully connected layer w 5760 params
        # The no of feature layers is set to 8 in the 1st block and doubled after each max pooling layer
        self.conv1 = nn.Conv3d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(8, affine=True)   # num_features=C from N C D H W

        self.conv3 = nn.Conv3d(8, 16, 3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm3d(16, affine=True)

        self.conv5 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.bnorm3 = nn.BatchNorm3d(32, affine=True)

        self.conv7 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.conv8 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.bnorm4 = nn.BatchNorm3d(64, affine=True)

        self.conv9 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.conv10 = nn.Conv3d(128, 128, 3, stride=1, padding=1)
        self.bnorm5 = nn.BatchNorm3d(128, affine=True)

        # Linear(in_features, out_features)
        # 12800
        self.fc1 = nn.Linear(128*5*5*4, 1)

    def forward(self, x):
        # x.shape = B C Z Y X
        x = F.relu(self.conv1(x))
        x = F.relu(self.bnorm1(self.conv2(x)))
        x = F.max_pool3d(x, 2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.bnorm2(self.conv4(x)))
        x = F.max_pool3d(x, 2, stride=2)

        x = F.relu(self.conv5(x))
        x = F.relu(self.bnorm3(self.conv6(x)))
        x = F.max_pool3d(x, 2, stride=2)

        x = F.relu(self.conv7(x))
        x = F.relu(self.bnorm4(self.conv8(x)))
        x = F.max_pool3d(x, 2, stride=2)

        x = F.relu(self.conv9(x))
        x = F.relu(self.bnorm5(self.conv10(x)))
        x = F.max_pool3d(x, 2, stride=2)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]    # all dimensions except the batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    #%% Define model
    net = Net()
    checkpoint_path = join(config.pytorch_models_path, arg['model_name'])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    #%% Training parameters -- here only for reference
    optimizer = optim.SGD(net.parameters(), lr=arg['learning_rate'], momentum=0.9, nesterov=True)
    gamma = (100 - arg['lr_decay_percentage']) / 100
    scheduler = StepLR(optimizer, step_size=arg['lr_decrease_step'], gamma=gamma)
    criterion = nn.L1Loss()

    #%% Prediction
    input = torch.randn(arg['batch_size'], 1, 193, 229, 193)  # size of MNI152 atlas
    # crop brain tissue from MRI image
    input = AtlasCrop()(input)

    target = torch.randint(18, 100, (arg['batch_size'], 1))
    input, target = input.to(device), target.to(device)

    net.eval()
    with torch.set_grad_enabled(False):
        output = net(input)
    loss = criterion(output, target)
    print('MAE: ', loss.item())



