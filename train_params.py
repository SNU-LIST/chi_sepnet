'''
#
# Description:
#  Training parameters for x-sepnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 24.09.20
'''

"""
Experiment setting parameters
    GPU_NUM: number of GPU to use.
    SEED: fixed seed for reproducibility.
    INPUT_MAP: type of input map for the networks. (r2p: R2', r2s: R2*)
    INPUT_UNIT: unit of input maps ('Hz', 'ppm', 'radian')
    
    TRAIN_PATH: path of train dataset.
    VALID_PATH: path of valid dataset.
    VALUE_PATH: path of value file including mean, std of dataset.
    TRAIN_FILE: filename of train dataset.
    VALID_FILE: filename of valid dataset.
    VALUE_FILE: filename of value file including mean, std of dataset.
    
    CHECKPOINT_PATH = path of network(x-sepnet) checkpoint.
    
    PRE_NET_CHECKPOINT_PATH: path of pre-network(QSMnet) checkpoint.
    PRE_NET_CHECKPOINT_FILE: filename of pre-network(QSMnet) checkpoint.
    PRE_VALUE_FILE: path of value file including mean, std of dataset for pre-network(QSMnet)

    PRE_R2PNET_CHECKPOINT_PATH: path of pre-network(R2PRIMEnet) checkpoint.
    PRE_R2PNET_CHECKPOINT_FILE: filename of pre-network(R2PRIMEnet) checkpoint.
"""
GPU_NUM = '1'
SEED = 777
INPUT_MAP = 'r2p'
INPUT_UNIT = 'ppm'

TRAIN_PATH = '/fast_storage/minjun/x-sep/Train/'
VALID_PATH = '/fast_storage/minjun/x-sep/Valid/'
VALUE_PATH = './Data/'
TRAIN_FILE = 'xsepnet_train_patch_inplane_largedegree_romeo_arlo.hdf5'
VALID_FILE = 'subj06_DataFor_xsepnet_ppm_COSMOS_6dir_romeo_arlo.mat'
VALUE_FILE = 'xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat'

CHECKPOINT_PATH = './Checkpoint/240908_xsepnet_r2p_abstractArchitecture_all_romeo_arlo_onlymodelloss/'

### QSMnet ###
PRE_NET_CHECKPOINT_PATH = './qsmnet/Checkpoint/240831_qsmnet_expdecay_subj06_romeo_arlo_modellosscorrected/'#/'230102_qsmnet_expdecay_subj06_withSimul
PRE_NET_CHECKPOINT_FILE = 'best_loss_14.pth.tar'
PRE_VALUE_FILE = 'xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat'# 'xsepnetplus_train_patch_norm_factor_inplane_largedegree_sf4_36000.mat'

### R2PRIMEnet ###
PRE_R2PNET_CHECKPOINT_PATH = None#'./r2pnet/Checkpoint/240111_r2pnet_stepdecay_subj06/'#/'230102_qsmnet_expdecay_subj06_withSimul
PRE_R2PNET_CHECKPOINT_FILE = None#'best_loss.pth.tar'


"""
Physics-parameters
    delta_TE: time gap between multi-echo times
    CF: center frequency (used for Hz -> ppm calculation)
    Dr: relaxometric constrant between R2' and susceptibility
    
    * Input of network must be unit of ppm. So If the input map has unit of Hz, you can change it to ppm by entering physics-params below (only in case of inference).
    ** Ref: Shin, Hyeong-Geol, et al. "χ-separation: Magnetic susceptibility source separation toward iron and myelin mapping in the brain." Neuroimage 240 (2021): 118371.
"""
delta_TE = 0.003
CF = 123177385
Dr = 114


"""
Network-parameters
    CHANNEL_IN: number of out-channels for first conv layers
    KERNEL_SIZE: kernel size of conv layers
"""
CHANNEL_IN = 32
KERNEL_SIZE = 3


"""
Hyper-parameters
    TRAIN_EPOCH: number of total epoch for training
    SAVE_STEP: step for saving the epoch during training
    LEARNING_RATE: learning rate
    BATCH_SIZE: batch size
    W_L1LOSS: weight of L1loss
    W_GDLOSS: weight of gradient loss
    W_MD_F_LOSS: weight of field model loss
    W_MD_R_LOSS: weight of R2' model loss
    W_MD_Q_LOSS: weight of QSM model loss
"""
TRAIN_EPOCH = 60
SAVE_STEP = 10
LEARNING_RATE = 0.0003
BATCH_SIZE = 12
W_L1Loss = 1
W_GDLOSS = 0.1
W_MD_F_LOSS = 1
W_MD_R_LOSS = 1
W_MD_Q_LOSS = 1


import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--GPU_NUM", default=GPU_NUM)
    parser.add_argument("--SEED", default=SEED)
    parser.add_argument("--INPUT_MAP", default=INPUT_MAP)
    parser.add_argument("--INPUT_UNIT", default=INPUT_UNIT)
    parser.add_argument("--LEAKY_RELU_TOGGLE", default=LEAKY_RELU_TOGGLE)
    
    parser.add_argument("--TRAIN_PATH", default=TRAIN_PATH)
    parser.add_argument("--VALID_PATH", default=VALID_PATH)
    parser.add_argument("--VALUE_PATH", default=VALUE_PATH)
    parser.add_argument("--TRAIN_FILE", default=TRAIN_FILE)
    parser.add_argument("--VALID_FILE", default=VALID_FILE)
    parser.add_argument("--VALUE_FILE", default=VALUE_FILE)
    parser.add_argument("--PRE_VALUE_FILE", default=PRE_VALUE_FILE)
    
    parser.add_argument("--CHECKPOINT_PATH", default=CHECKPOINT_PATH)
    parser.add_argument("--PRE_NET_CHECKPOINT_PATH", default=PRE_NET_CHECKPOINT_PATH)
    parser.add_argument("--PRE_NET_CHECKPOINT_FILE", default=PRE_NET_CHECKPOINT_FILE)
    parser.add_argument("--PRE_R2PNET_CHECKPOINT_PATH", default=PRE_R2PNET_CHECKPOINT_PATH)
    parser.add_argument("--PRE_R2PNET_CHECKPOINT_FILE", default=PRE_R2PNET_CHECKPOINT_FILE)

    parser.add_argument("--delta_TE", default=delta_TE)
    parser.add_argument("--CF", default=CF)
    parser.add_argument("--Dr", default=Dr)

    parser.add_argument("--CHANNEL_IN", default=CHANNEL_IN)
    parser.add_argument("--KERNEL_SIZE", default=KERNEL_SIZE)
    
    parser.add_argument("--TRAIN_EPOCH", default=TRAIN_EPOCH)
    parser.add_argument("--SAVE_STEP", default=SAVE_STEP)
    parser.add_argument("--LEARNING_RATE", default=LEARNING_RATE)
    parser.add_argument("--LR_EXP_DECAY_GAMMA", default=LR_EXP_DECAY_GAMMA)
    parser.add_argument("--BATCH_SIZE", default=BATCH_SIZE)
    parser.add_argument("--W_L1LOSS", default=W_L1Loss)
    parser.add_argument("--W_GDLOSS", default=W_GDLOSS)
    parser.add_argument("--W_MD_F_LOSS", default=W_MD_F_LOSS)
    parser.add_argument("--W_MD_R_LOSS", default=W_MD_R_LOSS)
    parser.add_argument("--W_MD_Q_LOSS", default=W_MD_Q_LOSS)
    args = parser.parse_args()
    return args
