# Downsampled 3D model
# [1]M. Ueda et al., “An Age Estimation Method Using 3D-CNN From Brain MRI Images,” in 2019 IEEE 16th International
# Symposium on Biomedical Imaging (ISBI 2019), Apr. 2019, pp. 380–383. doi: 10.1109/ISBI.2019.8759392.
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from src.data.build_dataset_functions import Downsample
import src.config as config
import warnings
warnings.filterwarnings('ignore')

# %% Define training parameters
arg = {'model_name' : 'model3-ueda-seed_92911-400ep.tar',
       'random_seed': 42,
       'n_epoch': 400,
       'learning_rate': 0.00005,
       'lr_decay': 0.0005,
       'lr_decrease_step': 1,
       'batch_size': 8,
       'p_pad': 0.3,
       'p_shift': 0.3,
       'p_flip': 0.5}

# set seed
torch.manual_seed(arg['random_seed'])
np.random.seed(arg['random_seed'])


# %% Define CNN
class Net(nn.Module):
    def __init__(self, bn_affine=True):
        super(Net, self).__init__()
        # Conv3d: Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        # groups=1, bias=True, padding_mode='zeros') .... N C D H W = N 1 z y x

        # 5 blocks of:
        # - 3x3x3 conv layer stride=1 + ReLU
        # - 3x3x3 conv + batch normalization layer + ReLU
        # - 2x2x2 max pooling layer stride=2
        # One fully connected layer w 5760 params
        # The no of feature layers is set to 8 in thee 1st block and doubled after each max pooling layer
        self.conv1 = nn.Conv3d(1, 8, 3, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm3d(8, affine=bn_affine)
        self.conv2 = nn.Conv3d(8, 16, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm3d(16, affine=bn_affine)
        self.conv3 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm3d(32, affine=bn_affine)
        self.conv4 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm3d(64, affine=bn_affine)

        # Linear(in_features, out_features)
        self.fc1 = nn.Linear(64 * 5 * 5 * 6, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1_bn(self.conv1(x))
        x = F.relu(F.max_pool3d(x, 2, stride=2))

        x = self.conv2_bn(self.conv2(x))
        x = F.relu(F.max_pool3d(x, 2, stride=2))

        x = self.conv3_bn(self.conv3(x))
        x = F.relu(F.max_pool3d(x, 2, stride=2))

        x = self.conv4_bn(self.conv4(x))
        x = F.relu(F.max_pool3d(x, 2, stride=2))

        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.fc3(x)
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
    checkpoint_path = join(config.project_root_path, 'BASE_models', arg['model_name'])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    #%% Training parameters -- here only for reference
    optimizer = optim.SGD(net.parameters(), lr=arg['learning_rate'], momentum=0.9, nesterov=True,  weight_decay=0.0005)
    lr_lambda = lambda ep: 1 / (1 + arg['lr_decay'] * (ep // arg['n_epoch']))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.L1Loss()

    #%% Prediction
    target = torch.randint(18, 100, (arg['batch_size'], 1))
    input = torch.randn(arg['batch_size'], 1, 193, 229, 193)  # size of MNI152 atlas
    # crop brain tissue from MRI image and Downsample
    transform = Downsample()
    #for each 3D MRI image
    # Iterate over each sample in the batch
    input_dwnsmp = torch.empty((arg['batch_size'], 1, 78, 79, 95))
    for i in range(input.shape[0]):
        sample = input[i].squeeze()   # Get the i-th sample in the batch
        # transform the sample
        input_dwnsmp[i] = transform(sample).unsqueeze(0).unsqueeze(0)

    input_dwnsmp, target = input_dwnsmp.to(device), target.to(device)
    net.eval()
    # predict
    with torch.set_grad_enabled(False):
        output = net(input_dwnsmp)
    loss = criterion(output, target)
    print('MAE: ', loss.item())



