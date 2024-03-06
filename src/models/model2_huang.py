# 2D model
# Huang et al., “Age estimation from brain MRI images using deep learning,” in 2017 IEEE 14th International Symposium
# on Biomedical Imaging (ISBI 2017), Apr. 2017, pp. 849–852. doi: 10.1109/ISBI.2017.7950650.
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR

from src.data.build_dataset_functions import AtlasCrop2D, AxialSlicer
import src.config as config
import warnings
warnings.filterwarnings('ignore')

# %% Define training parameters
arg = {'model_name': 'model2-huang-seed_42-400ep.tar',
       'random_seed': 42,
       'learning_rate': 0.001,
       'lr_decay': 0.0001,
       'lr_decrease_step': 1,
       'n_epoch': 400,
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
        self.conv1 = nn.Conv2d(15, 64, 3, stride=3, padding=(2, 2))
        self.conv1_bn = nn.BatchNorm2d(64, affine=bn_affine)
        self.conv2 = nn.Conv2d(64, 192, 3, stride=1, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(192, affine=bn_affine)
        self.conv3 = nn.Conv2d(192, 384, 3, stride=1, padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(384, affine=bn_affine)
        self.conv4 = nn.Conv2d(384, 512, 3, stride=1, padding=(1, 1))
        self.conv4_bn = nn.BatchNorm2d(512, affine=bn_affine)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=(1, 1))
        self.conv5_bn = nn.BatchNorm2d(512, affine=bn_affine)
        self.conv6 = nn.Conv2d(512, 128, 3, stride=1, padding=(1, 1))
        self.conv6_bn = nn.BatchNorm2d(128, affine=bn_affine)

        # Linear(in_features, out_features)
        self.fc1 = nn.Linear(128 * 3 * 2, 128 * 3 * 2)  # 3*2 from image dimension
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128 * 3 * 2, 128 * 3 * 2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128 * 3 * 2, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 3, stride=2))
        x = self.conv1_bn(x)

        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 3, stride=2))
        x = self.conv2_bn(x)
        # .....
        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)

        x = self.conv4(x)
        x = self.conv4_bn(F.relu(x))
        # ......
        x = self.conv5(x)
        x = F.relu(F.max_pool2d(x, 3, stride=2))
        x = self.conv5_bn(x)

        x = self.conv6(x)
        x = F.relu(F.max_pool2d(x, 3, stride=2))
        x = self.conv6_bn(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
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
    optimizer = optim.SGD(net.parameters(), lr=arg['learning_rate'], momentum=0.9, nesterov=True, weight_decay=0.001)
    lr_lambda = lambda ep: 1 / (1 + arg['lr_decay'] * (ep // arg['n_epoch']))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.L1Loss()

    #%% Prediction
    input = torch.randn(arg['batch_size'], 193, 229, 193)  # size of MNI152 atlas
    # slice 2D slices from 3D MRI and crop brain tissue from MRI image
    transform = transforms.Compose([AxialSlicer((30, 143, 8)),
                                    AtlasCrop2D()])
    input = transform(input)

    target = torch.randint(18, 100, (arg['batch_size'], 1))
    input, target = input.to(device), target.to(device)

    net.eval()
    with torch.set_grad_enabled(False):
        output = net(input)
    loss = criterion(output, target)
    print('MAE: ', loss.item())



