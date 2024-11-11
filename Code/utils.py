'''
#
# Description:
#  Util codes for x-sepnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 24.11.11
'''
import os
import math
import numpy as np
import h5py
import scipy.io
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from random import sample
import logging
import logging_helper as logging_helper
import time
import datetime

from math import log10, sqrt
from numpy.fft import fftn, ifftn, fftshift
from skimage.metrics import structural_similarity as ssim

        
def set_global_mean_std(pos_mean, pos_std, neg_mean, neg_std):
    global GLOBAL_POS_MEAN
    global GLOBAL_POS_STD
    global GLOBAL_NEG_MEAN
    global GLOBAL_NEG_STD

    GLOBAL_POS_MEAN = pos_mean
    GLOBAL_POS_STD = pos_std
    GLOBAL_NEG_MEAN = neg_mean
    GLOBAL_NEG_STD = neg_std
    
    
def Concat(x, y):
    return torch.cat((x,y),1)


class Conv3d(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=k_size, stride=1, padding=int(k_size/2), dilation=1)
        self.bn = nn.BatchNorm3d(c_out)
        self.act = nn.ReLU()
        nn.init.xavier_uniform_(self.conv.weight)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    
class Conv(nn.Module):
    def __init__(self, c_in, c_out):
        super(Conv, self).__init__()
        self.conv=nn.Conv3d(c_in, c_out, kernel_size=1, stride=1, padding=0, dilation=1)
        nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self,x):
        return self.conv(x)
    
    
class Pool3d(nn.Module):
    def __init__(self):
        super(Pool3d, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1)
    
    def forward(self,x):
        return self.pool(x)
    
    
class Deconv3d(nn.Module):
    def __init__(self, c_in, c_out):
        super(Deconv3d, self).__init__()
        self.deconv=nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2, padding=0, dilation=1)
        nn.init.xavier_uniform_(self.deconv.weight)
    
    def forward(self,x):
        return self.deconv(x)


def load_pretrained_state_dict(model: nn.Module, model_weights_path: str, logger):
    checkpoint = torch.load(model_weights_path)
    epoch = checkpoint["epoch"]

    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    for key in state_dict.keys():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)

    return model, epoch


def setting_logger(path, file_name, module_type):
    if module_type == 'train':
        logger = logging.getLogger("module.train")
    elif module_type == 'test':
        logger = logging.getLogger("module.test")
        
    logger.setLevel(logging.INFO)
    logging_helper.setup(path, file_name)
    
    nowDate = datetime.datetime.now().strftime('%Y-%m-%d')
    nowTime = datetime.datetime.now().strftime('%H:%M:%S')
    logger.info(f'Date: {nowDate}  {nowTime}')

    return logger


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def setting_seed(seed):
    os.environ['PYTHONHASHargs.seed'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic=True # for reproducibility: True is recommended
    torch.backends.cudnn.benchmark=False # for reproducibility: False is recommended
    g = torch.Generator()
    g.manual_seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
        

def dipole_kernel(matrix_size, voxel_size, B0_dir):
    """
    Args:
        matrix_size (array_like): should be length of 3.
        voxel_size (array_like): should be length of 3.
        B0_dir (array_like): should be length of 3.
        
    Returns:
        D (ndarray): 3D dipole kernel matrix in Fourier domain.  
    """    
    x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
    y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
    z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
    Y, X, Z = np.meshgrid(x, y, z)
    
    X = X/(matrix_size[0]*voxel_size[0])
    Y = Y/(matrix_size[1]*voxel_size[1])
    Z = Z/(matrix_size[2]*voxel_size[2])
    
    D = 1/3 - (X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2/((X**2 + Y**2 + Z**2) + 1e-6)
    D[np.isnan(D)] = 0
    D = fftshift(D)
    return D
    
    
def l1_loss(x, y):
    return torch.abs(x-y).mean()
    
    
def field_model_loss(pred_sus, label_local_f, m, d):
    """
    Args:
        pred_sus (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. predicted susceptability map.
        label_local_f (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. local field map.
        m (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. mask.
        d (ndarray): (s1, s2, s3) size matrix. dipole kernel.
        
    Description:
        local field loss: [ local field map = sus map * dipole kernel ] using FFT, we can use multiplication instead of convolution in image domain.
    """
    num_batch = pred_sus.shape[0]
    device = pred_sus.device;
    
    ### FFT(pred sus map) x dipole kernel ###
    pred_sus = torch.stack((pred_sus, torch.zeros(pred_sus.shape, dtype=pred_sus.dtype, device=device)), dim=-1)
    fft_p = torch.fft(pred_sus, 3)
    
    d = d[np.newaxis, np.newaxis, ...]
    d = torch.tensor(d, dtype=pred_sus.dtype, device=device).repeat(num_batch, 1, 1, 1, 1)
    d = torch.stack((d, torch.zeros(d.shape, dtype=pred_sus.dtype, device=device)), dim=-1)
    
    y = torch.zeros(pred_sus.shape, dtype=pred_sus.dtype, device=device)
    y[..., 0] = fft_p[..., 0] * d[..., 0] - fft_p[..., 1] * d[..., 1] # real part
    y[..., 1] = fft_p[..., 0] * d[..., 1] + fft_p[..., 1] * d[..., 0] # imaginary part
    
    ### IFT results = pred susceptibility map * dipole kernel ###
    y = torch.ifft(y, 3)
    pred_local_f = y[..., 0]
    
    local_f_loss = l1_loss(label_local_f*m, pred_local_f*m)
    
    return local_f_loss


def grad_loss(x, y):
    device = x.device
    x_cen = x[:,:,1:-1,1:-1,1:-1]
    grad_x = torch.zeros(x_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = x[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(x_cen.shape, device=device)
                else:
                    temp = torch.relu(x_slice-x_cen)/s
                grad_x = grad_x + temp
    
    y_cen = y[:,:,1:-1,1:-1,1:-1]
    grad_y = torch.zeros(y_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                y_slice = y[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(y_cen.shape, device=device)
                else:
                    temp = torch.relu(y_slice-y_cen)/s
                grad_y = grad_y + temp
    
    return l1_loss(grad_x, grad_y)    


def total_loss(args, pred, label_x, label_sus, label_r2_, label_r2p, input_field, m,
               x_pos_mean, x_pos_std, x_neg_mean, x_neg_std, sus_mean, sus_std, local_f_mean, local_f_std,
               r2n_mean, r2n_std, r2_mean, r2_std, r2p_mean, r2p_std):
    """
    Args:
        input_type: 
        pred (torch.tensor): (batch_size, 2, s1, s2, s3) size matrix. predicted susceptability map.
        label_x (torch.tensor): (batch_size, 2, s1, s2, s3) size matrix. para, dia susceptability maps (label).
        label_r2_ (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. input r2prime or r2star map (label).
        label_r2 (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. input r2 map (label).
        input_field (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. input local field map.
        m (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. brain mask.

        w_l1 (float): weighting factor for L1 losses
        w_gd (float): weighting factor for gradient losses
        w_md_f (float): weighting factor for 'local field' model loss
        w_md_r (float): weighting factor for 'r2prime' model loss
        w_md_q (float): weighting factor for 'QSM' model loss
        
        x_pos_mean (float): mean value of para susceptibility map
        x_pos_std (float): std value of para susceptibility map
        x_neg_mean (float): mean value of dia susceptibility map
        x_neg_std (float): std value of dia susceptibility map
        local_f_mean (float): mean value of local field map
        local_f_std (float): std value of local field map
        r2n_mean (float): mean value of r2prime or r2star map
        r2n_std (float): std value of r2prime or r2star map
        r2_mean (float): mean value of r2 map
        r2_std (float): std value of r2 map
        
    Returns:
        total_loss (torch.float): total loss. sum of above three losses with weighting factor
        l1loss (torch.float): L1 loss. 
        gdloss (torch.float): gradient loss
        md_f_loss (torch.float): local field model loss
        md_q_loss (torch.float): qsm model l oss
        md_r_loss (torch.float): r2' model loss
    """
    ### Splitting into positive/negative maps & masking ###
    p_pos = pred[:, 0, :, :, :]
    p_neg = pred[:, 1, :, :, :]
    
    pred_x_pos = p_pos[:, np.newaxis, :, :, :] * m
    pred_x_neg = p_neg[:, np.newaxis, :, :, :] * m

    l_pos = label_x[:, 0, :, :, :]
    l_neg = label_x[:, 1, :, :, :]
    
    label_x_pos = l_pos[:, np.newaxis, :, :, :] * m
    label_x_neg = l_neg[:, np.newaxis, :, :, :] * m
    
    pred_x_maps = torch.cat((pred_x_pos, pred_x_neg), 1)
    label_x_maps = torch.cat((label_x_pos, label_x_neg), 1)
    
    '''
    ###############
    ### L1 loss ###
    ###############
    '''
    l1loss = l1_loss(pred_x_maps, label_x_maps)
    
    '''
    #####################
    ### Gradient loss ###
    #####################
    '''
    gdloss_pos = grad_loss(pred_x_pos, label_x_pos)
    gdloss_neg = grad_loss(pred_x_neg, label_x_neg)
    
    gdloss = (gdloss_pos + gdloss_neg)
    
    ### De-normalization ###
    device = pred_x_maps.device
    pred_x_pos = (pred_x_pos * x_pos_std) + x_pos_mean
    pred_x_neg = (pred_x_neg * x_neg_std) + x_neg_mean
    local_f = (input_field * local_f_std) + local_f_mean
    label_sus = (label_sus * sus_std) + sus_mean
    # label_r2 = (label_r2 * r2_std) + r2_mean
    
    if args.INPUT_MAP == 'r2p':
        label_r2p = (label_r2_ * r2n_std) + r2n_mean
        label_r2p = label_r2p * m
    elif args.INPUT_MAP == 'r2s':
        label_r2p = (label_r2p * r2p_std) + r2p_mean
        label_r2p = label_r2p * m
    
    local_f = local_f * m
    label_sus = label_sus * m

    '''
    ####################
    ### Model losses ###
    ####################
    '''
    pred_sus = pred_x_pos - pred_x_neg
    pred_r2p = pred_x_pos + pred_x_neg
    d = dipole_kernel(pred_sus.shape[-3:], voxel_size=(1,1,1), B0_dir=(0,0,1))

    md_f_loss = 0
    md_r_loss = 0
    md_q_loss = 0
    
    if args.W_MD_F_LOSS != 0:
        md_f_loss = field_model_loss(pred_sus, local_f, m, d)
    if args.W_MD_Q_LOSS != 0:
        md_q_loss = l1_loss(pred_sus*m, label_sus*m)
    if args.W_MD_R_LOSS != 0:
        if args.INPUT_MAP == 'r2p':
            md_r_loss = l1_loss(pred_r2p*m, label_r2p*m)
        elif args.INPUT_MAP == 'r2s':
            print('wrong input')
    
    total_loss = (l1loss * args.W_L1LOSS) + (gdloss * args.W_GDLOSS) + (md_f_loss * args.W_MD_F_LOSS) + (md_q_loss * args.W_MD_Q_LOSS) + (md_r_loss * args.W_MD_R_LOSS)
    return total_loss, l1loss, gdloss, md_f_loss, md_q_loss, md_r_loss


def save_model(epoch, model, PATH, TAG):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict()},
        f'{PATH}/{TAG}.pth.tar')
    # torch.save(model, f'{PATH}/model.pt')
    
    
def NRMSE(im1, im2, mask):
    im1 = im1 * mask
    im2 = im2 * mask

    mse = torch.mean((im1[mask>0]-im2[mask>0])**2)
    nrmse = sqrt(mse)/sqrt(torch.mean(im2[mask>0]**2))
    return 100*nrmse


def PSNR(im1, im2, mask):
    im1 = im1 * mask
    im2 = im2 * mask
    
    # mse = torch.mean((im1-im2)**2)
    mse = torch.mean((im1[mask>0]-im2[mask>0])**2)
    
    if mse == 0:
        return 100
    #PIXEL_MAX = max(im2[mask])
    PIXEL_MAX = 1
    return 20 * log10(PIXEL_MAX / sqrt(mse))


def SSIM(im1, im2, mask):
    im1 = im1.cpu().detach().numpy(); im2 = im2.cpu().detach().numpy(); mask = mask.cpu().detach().numpy()
    im1 = im1 * mask; im2 = im2 * mask;
    mask = mask.astype(bool)
    
    min_im = np.min([np.min(im1), np.min(im2)])
    im1[mask] = im1[mask] - min_im
    im2[mask] = im2[mask] - min_im
    
    max_im = np.max([np.max(im1), np.max(im2)])
    im1 = 255 * im1 / max_im
    im2 = 255 * im2 / max_im
    
    if len(im1.shape) == 3:
        ssim_value, ssim_map = ssim(im1, im2, data_range=255, full=True)
    
        return np.mean(ssim_map[mask]), ssim_map
    elif len(im1.shape) == 5:
        im1 = im1.squeeze()
        im2 = im2.squeeze()
        mask = mask.squeeze()
        
        if len(im1.shape) == 3:
            im1 = np.expand_dims(im1, axis=0)
            im2 = np.expand_dims(im2, axis=0)
            mask = np.expand_dims(mask, axis=0)
        
        ssim_maps = np.zeros(im1.shape)

        for i in range(0, im1.shape[0]):
            _, ssim_maps[i, :, :, :] = ssim(im1[i, :, :, :], im2[i, :, :, :], data_range=255, full=True)
        return np.mean(ssim_maps[mask])
    else:
        raise Exception('SSIM - input dimension error')    
    
    
def crop_img_16x(img):
    """
    input: 3D img [H, W, C]
    output: cropped 3D img with a H, W of 16x
    """
    if img.shape[0] % 16 != 0:
        residual = img.shape[0] % 16
        img = img[int(residual/2):int(-(residual/2)), :, :]
        
    if img.shape[1] % 16 != 0:
        residual = img.shape[1] % 16
        img = img[:, int(residual/2):int(-(residual/2)), :]
        
    if img.shape[2] % 16 != 0:
        residual = img.shape[2] % 16
        img = img[:, :, int(residual/2):int(-(residual/2))]
        
    return img


class SSIM_cal_3D(nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size, win_size) / win_size ** 3)
        NP = win_size ** 3
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv3d(X, self.w.to(X.device))
        uy = F.conv3d(Y, self.w.to(X.device))
        uxx = F.conv3d(X * X, self.w.to(X.device))
        uyy = F.conv3d(Y * Y, self.w.to(X.device))
        uxy = F.conv3d(X * Y, self.w.to(X.device))
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return torch.mean(S,[2,3,4],keepdim=False)


class SSIM_cal_2D(nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w.to(X.device))
        uy = F.conv2d(Y, self.w.to(X.device))
        uxx = F.conv2d(X * X, self.w.to(X.device))
        uyy = F.conv2d(Y * Y, self.w.to(X.device))
        uxy = F.conv2d(X * Y, self.w.to(X.device))
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return torch.mean(S,[2,3],keepdim=False)
    
ssim_cal_3d = SSIM_cal_3D()
ssim_cal_2d = SSIM_cal_2D()


def calculate_ssim(img, ref, mask):
    if mask != None:
        mask = mask.to(img.device)
        img = img * mask
        ref = ref * mask
    img = img.unsqueeze(0).unsqueeze(0)
    ref = ref.unsqueeze(0).unsqueeze(0)
    
    ones = torch.ones(ref.shape[0]).to(ref.device)
    
    if len(img.shape) == 5:
        ssim = ssim_cal_3d(img, ref, ones)
    elif len(img.shape) == 4:
        ssim = ssim_cal_2d(img, ref, ones)
    return ssim