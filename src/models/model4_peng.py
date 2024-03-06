# SFCN -- march 2020
# https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/brain_age/sfcn.py
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.data.build_dataset_functions import AtlasCrop, categorize, discretize_norm
import src.config as config
import warnings
warnings.filterwarnings('ignore')

# %% Define training parameters
arg = {'model_name': 'model4-peng-seed_92911-110ep.tar',
       'random_seed': 42,
       'learning_rate': 0.01,
       'lr_decay_percentage': 70,
       'lr_decrease_step': 30,
       'batch_size': 8,
       'output_dim': 41,
       'p_pad': 0.3,
       'p_shift': 0.3,
       'p_flip': 0.5}

# set seed
torch.manual_seed(arg['random_seed'])
np.random.seed(arg['random_seed'])

# %% Define CNN
class Net(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=41, dropout=True):
        """

        :param channel_number:
        :param output_dim: number of output categories; by default we implement 2-year intervals: (18-100)/2; default 41
        :param dropout:
        """
        super(Net, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        # avg_shape = [5, 6, 5]
        avg_shape = [5, 5, 4]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))
        del i, in_channel, out_channel, avg_shape, n_layer

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU()
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        # in article: x.size = 160x192x160
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.log_softmax(x, dim=1)
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
    optimizer = optim.SGD(net.parameters(), lr=arg['learning_rate'], momentum=0.9, nesterov=True,  weight_decay=0.001)
    gamma = (100 - arg['lr_decay_percentage']) / 100
    scheduler = StepLR(optimizer, step_size=arg['lr_decrease_step'], gamma=gamma)
    criterion = nn.KLDivLoss(reduction='batchmean')   # used as loss function
    mae = nn.L1Loss()

    #%% Prediction
    input = torch.randn(arg['batch_size'], 1, 193, 229, 193)  # size of MNI152 atlas
    # crop brain tissue from MRI image
    input = AtlasCrop()(input)

    target = torch.randint(18, 100, (arg['batch_size'], 1))

    # create age categories (2-year intervals from 18 to 100 years)
    categories = torch.linspace(18, 100, arg['output_dim']+1)
    categories_center = (categories[1:] + categories[:-1])/2
    categories_center = categories_center.view(arg['output_dim'], -1)

    # compute class probabilities of target age
    target_class_prob = discretize_norm(target, categories, scale=1.0)

    # input, target, target_class_prob = input.to(device), target.to(device), target_class_prob.to(device)

    net.eval()
    with torch.set_grad_enabled(False):
        output = net(input)
    loss = criterion(output, target_class_prob)

    # numerical (age in years)
    outputs_num = torch.mm(torch.exp(output.to('cpu')), categories_center)

    print('loss: ', loss.item())
    print('MAE: ', mae(output, outputs_num).item())
