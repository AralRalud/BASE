import os
import random
from os.path import join
import warnings
from itertools import compress, repeat

import scipy.linalg as linalg
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import SimpleITK as itk
from torch.utils.data import Dataset
import src.config as config
from scipy.stats import norm
from scipy.interpolate import interpn

# from src.data.resample_transformed import Resampler

import matplotlib.pyplot as plt

# %% Dataset class
# Abstract class representing a dataset: torch.utils.data.Dataset
# A custom dataset should inherit Dataset and override
# __len__  so that len(dataset) returns the size of the dataset.
# __getitem__ to support the indexing such that dataset[i] can be used to get ith sample

# reading of images should be left to __getitem__ not init!!

# %% Transforms
class Normalize(object):
    """
    Normalize by channel [c],z,y,x image by subtracting mean value and divided by standard deviation.
    Mean and std are computed for each image on the spot.

    Args:
        ttensor (bool): True if input object is a torch.tensor
    """
    def __call__(self, sample):
        image = sample['image']
        d = image.ndim

        mean = np.mean(image, (-3, -2, -1))
        std = np.std(image, (-3, -2, -1))
        while mean.ndim < d:
            mean = np.expand_dims(mean, -1)
            std = np.expand_dims(std, -1)
        image_norm = (image - mean) / std

        sample['image'] = image_norm
        return sample


class AtlasCrop(object):
    """
    Crop the MR image to extract the brain structure location based on brain structure in MNI152 v2009 atlas. The MRI
    must be registered to MNI152 v2009 atlas prior to cropping.
    """
    def __call__(self, image):
        return image[..., 0:170, 18:207, 19:176]    #[N],[c], z, y, x


class Downsample(object):
    """
    T1w fMRI BASE for predicting BrainAGE
    Preprocessing steps:
    # 1. Affine registration to MNI space using RawPreproc_v2 pipeline
    # 2. Crop to cut out braina and remove black space
    # 3. Gaussian smoothing with FHWM = 2
    # 4. Resize to 95 x 79 x 78 voxel
    Normalization is done in the learning transformation before learning

    Based on Ueda et al. where the following steps are used:
    1. Align with the standard ICBM 152 template using VBM2 process proposed by Good et al. in SPM2
       smoothing: 12-mm FWHM isotropic Gaussian kernel
    2. Resize to 95 x 79 x 78 voxel
    3. Normalize to zero mean and unit variance
    """

    def __init__(self):
        self.FHWM = 2  # in mm

    def __call__(self, sample):
        # change to Simple ITK image
        t1w_obj = itk.GetImageFromArray(sample)
        if sample.ndim > 3:
            raise ValueError('Not Implemented for 4D')

        # Crop the image to extract the brain structure location based on brain structure in MNI atlas
        cropped_obj = t1w_obj[19:176, 18:207, 0:170]
        # Downsample and smooth using FHWM Gaussian 12 mm
        # FHWM_mm = sigma_mm * sqrt(8 * ln2)
        sigma_mm = self.FHWM / np.sqrt(8 * np.log(2))
        sigma_px = [sigma_mm / spacing_i for spacing_i in cropped_obj.GetSpacing()]

        sample = itk.GetArrayFromImage(self.Resize(cropped_obj, size=(95, 79, 78), sigma=sigma_px))

        return torch.from_numpy(sample)

    class Downsample(object):
        """
        T1w fMRI BASE for predicting BrainAGE
        Preprocessing steps:
        # 1. Affine registration to MNI space using RawPreproc_v2 pipeline
        # 2. Crop to cut out braina and remove black space
        # 3. Gaussian smoothing with FHWM = 2 # FIXME
        # 4. Resize to 95 x 79 x 78 voxel
        Normalization is done in the learning transformation before learning

        Based on Ueda et al. where the following steps are used:
        1. Align with the standard ICBM 152 template using VBM2 process proposed by Good et al. in SPM2
           smoothing: 12-mm FWHM isotropic Gaussian kernel
        2. Resize to 95 x 79 x 78 voxel
        3. Normalize to zero mean and unit variance
        """

        def __init__(self):
            self.FHWM = 2  # in mm

        def __call__(self, sample, spacing=(1, 1, 1)):
            # change to Simple ITK image
            t1w_obj = sample['image']
            if sample['image'].ndim > 3:
                raise ValueError('Not Implemented for 4D')

            # Crop the image to extract the brain structure location based on brain structure in MNI atlas
            cropped_obj = t1w_obj[0:170, 18:207, 19:176]  # should it be reversed??
            # Downsample and smooth using FHWM Gaussian 12 mm
            # FHWM_mm = sigma_mm * sqrt(8 * ln2)
            sigma_mm = self.FHWM / np.sqrt(8 * np.log(2))
            sigma_px = [sigma_mm / spacing_i for spacing_i in spacing]

            sample['image'] = itk.GetArrayFromImage(self.Resize(cropped_obj, size=(95, 79, 78), sigma=sigma_px))

            return sample
    @staticmethod
    def Resize(image, size, interpolator=itk.sitkLinear, smoothing=True, sigma=1.0):
        """
        Resize image to desired dimensions with optional Gaussian smoothing, keeping the origin and direction

        Args:
            image (SimpleITK.Image): image to be resized
            size (vector int): size of output image -- in pixel
            smoothing (bool)   should Gaussian smoothing be applied?
            sigma (vector double)   sigma of Gaussian filter -- in pixel (Sigma is measured in the units of image spacing)
            interpolator (sitk interpolator):
        """
        # The spatial definition of the images we want to use in a deep learning framework (smaller than the original).
        if image.GetDimension() > 3:
            raise ValueError('Not implemented for 4 dimensional input')
        reference_image = itk.Image(size, image.GetPixelIDValue())
        reference_image.SetOrigin(image.GetOrigin())
        reference_image.SetDirection(image.GetDirection())
        reference_image.SetSpacing(
            [osize * spc / nsize for nsize, osize, spc in
             zip(size, image.GetSize(), image.GetSpacing())])

        if smoothing:
            # Resample after Gaussian smoothing.
            out_image = itk.Resample(itk.SmoothingRecursiveGaussian(image, sigma), reference_image,
                                     itk.Transform(), interpolator)
            # Transform(): By default a 3-d identity transform is constructed.
        else:
            out_image = itk.Resample(image, reference_image, sitk.Transform(), interpolator)

        return out_image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors and by default adds a dimension on the 0 axis unitl dim=4."""
    # To Tensor (itk [x,y,z] -> np [z,y,x] -> tensor [z,y,x])??
    def __init__(self, device=None, add_channel=True, image_session_id=False):
        self.add_channel = add_channel # FIXME: obsolete; to be removed and replaced with following Uns
        self.image_session_id = image_session_id
        self.device = torch.device(device) if device else None

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        target = torch.tensor(sample['target'], dtype=torch.float32)
        image = torch.tensor(image.copy(), dtype=torch.float)  # z, y, x

        # if self.add_channel and image.dim() < 4:  FIXME: obsolete; to be removed
        #     image = image.unsqueeze(0)    # add channel
        if self.device:
            image = image.to(self.device)
            target = target.to(self.device)

        sample['image'] = image
        sample['target'] = target
        return sample


class Unsqueeze(object):
    """Adds a dimension on the 0 axis if ndim<4 WARNING: input must be torch.tensor"""
    def __call__(self, sample):
        if not isinstance(sample['image'], torch.Tensor):
            raise TypeError('sample[`image`] shoud be of type torch.Tensor, not {}'.format(type(sample['image'])) )
        if sample['image'].dim() < 4 :
            image = sample['image'].unsqueeze(0)    # add channel
            sample['image'] = image
        return sample

# %% Transforms 2D; For Model 2 (Huang)
class AtlasCrop2D(object):
    """
    Crop the MR image to extract the brain structure location based on brain structure in MNI152 v2009 atlas. The MRI
    must be registered to MNI152 v2009 atlas prior to cropping.

    The cropping is done on 2D axial stlices of the original 3D MRI, i.e. the image is cropped only on x and y axis.
    """
    def __call__(self, image):
        return image[..., 18:207, 19:176]    #[N], [C], [z], y, x


class AxialSlicer(object):
    """
    Slice 3D images alongside z-axis (Axial axis)
    Args:
        slice (tuple): tuple of the form (from, to, step)
    """

    def __init__(self, slice):
        assert isinstance(slice, tuple)
        assert len(slice) == 3
        self.slice = slice

    def __call__(self, image):
        fr, to, step = self.slice
        image_new = image[..., fr:to:step, :, :]  # z,y,x
        return image_new

#%% Functions
def categorize(num_vector, bin=(18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100),
               int_bin=False):
    """
    Categorize each element of a numerical vector into predefined bins.

    Parameters:
    :param num_vector (array_like, float): A numpy.ndarray (or a structure that can be converted to it) containing
    numerical data to be categorized.
    :param bin (tuple of int/float, optional): A tuple defining the bin edges for categorization. Default is
    (18, 20, 25, 30, 35, ..., 90, 100).
    :param int_bin (bool, optional): If True, categories are returned as integers representing bin indices;
    if False, categories are returned as strings showing bin ranges. Default is False.

    Returns:
    :return: A numpy.ndarray containing the categorized data. The data type of the array is either 'int'
    (if int_bin is True) or 'str' (if int_bin is False).
    """
    bin = np.array(bin, dtype=np.float32)
    num_vector = np.array(num_vector)

    if not np.all(np.sort(bin) == bin):
        raise ValueError("Categories should be in increasing order")
    if len(bin) < 2:
        warnings.warn("Vector remains unchanged; Expected len(bin)>=2.")
        return num_vector

    # Check for out of range values
    if np.any(num_vector < bin[0]) or np.any(num_vector > bin[-1]):
        raise ValueError(f"Values should be within the range [{bin[0]}, {bin[-1]}]")

    # Assign bin
    category_labels = [f"[{int(cat1)},{int(cat2)})" for cat1, cat2 in zip(bin, bin[1:])]
    category_labels[-1] = f"[{int(bin[-2])},{int(bin[-1])}]"

    if int_bin:
        # skip the first left bin edge bin[0] since digitize assumes (-Inf, bin[0])
        # skip the right most bin[-1], to include values x=bin[-1] to the last interval [bin[-2], bin[-1]]
        categorized = np.digitize(num_vector, bin[1:-1], right=False)
    else:
        categorized = np.array([category_labels[i] for i in np.digitize(num_vector, bin[1:-1])])
    return np.array(categorized)

def discretize_norm(loc, endpoints, scale=1.0):
    """

    :param loc: mean location of the distribution
    :param endpoints: n+1 endpoints of n intervals
    :return:
    """
    if type(loc) in [int, float]:
        loc = [loc]

    device = None
    out = torch.zeros(size=(len(loc), len(endpoints)-1))
    for i in range(len(loc)):
        loc_i = loc[i]

        if isinstance(loc_i, torch.Tensor):
            device = loc_i.device
            loc_i = np.array(loc_i.cpu())
        cdf = norm.cdf(endpoints, loc=loc_i, scale=scale)   # comulative distribution function; scale=std

        # assuring the sum will be 1
        cdf[0] = 0.0
        cdf[-1] = 1.0

        cdf_diff = cdf[1:] - cdf[:-1]
        out[i, :] = torch.from_numpy(cdf_diff)
    if device is not None:
        out = out.to(device, dtype=torch.float)
    return out

