'''
#
# Description:
#  Test parameters for x-sepnet
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
    INPUT_MAP: type of input map for the networks. (r2p: R2', r2s: R2*)
    INPUT_UNIT: unit of input maps ('radian' or 'Hz')
    LABEL_EXIST: bool value if ground-truth label exist (True: ground-truth exist, False: ground-truth not exist)
    MASK_EXIST: type of mask for evaluation (None: no mask exist, CSF: CSF mask exist, Vessel: vessel mask exist)

    QSM_RES_GEN_TOGGLE: bool value to use resolution generalization method (True: using resolution generalization, False: not using)
    RESULT_SAVE_TOGGLE: bool value to decide saving the results (True: save the results, False: do not save the results)
    
    TEST_PATH: path of test dataset.
    TEST_FILE: filename of test dataset.
    
    CHECKPOINT_PATH: path of network(x-sepnet) checkpoint.
    CHECKPOINT_FILE: filename of network(x-sepnet) checkpoint.
    VALUE_FILE_PATH: path of file of normalization factors used in x-sepnet training
    VALUE_FILE_NAME: filename of normalization factors used in x-sepnet training
    
    QSM_NET_CHECKPOINT_PATH: path of pre-network(QSMnet) checkpoint.
    QSM_NET_CHECKPOINT_FILE: filename of pre-network(QSMnet) checkpoint.
    QSM_NET_VALUE_FILE_PATH: path of file of normalization factors used in QSMnet training
    QSM_NET_VALUE_FILE_NAME: filename of normalization factors used in QSMnet training

    R2P_NET_CHECKPOINT_PATH: path of pre-network(R2PRIMEnet) checkpoint.
    R2P_NET_CHECKPOINT_FILE: filename of pre-network(R2PRIMEnet) checkpoint.
    R2P_NET_VALUE_FILE_PATH: path of file of normalization factors used in R2PRIMEnet training
    R2P_NET_VALUE_FILE_NAME: filename of normalization factors used in R2PRIMEnet training

    RESULT_PATH: path to save the results.
    RESULT_FILE: filename of result file (mat).
"""
GPU_NUM = '0'
INPUT_MAP = 'r2p'
INPUT_UNIT = 'ppm'
LABEL_EXIST = True
MASK_EXIST = 'Vessel'

QSM_RES_GEN_TOGGLE = True
RESULT_SAVE_TOGGLE = True

TEST_PATH = './Data/'
TEST_FILE = ['test_file.mat']

CHECKPOINT_PATH = './Checkpoint/'
CHECKPOINT_FILE = 'xsepnet.pth.tar'
VALUE_FILE_PATH = './Checkpoint/'
VALUE_FILE_NAME = 'xsepnet_norm_factor.mat'

QSM_NET_CHECKPOINT_PATH = './Checkpoint/'
QSM_NET_CHECKPOINT_FILE = 'QSMnet.pth.tar'
QSM_NET_VALUE_FILE_PATH = './Checkpoint/'
QSM_NET_VALUE_FILE_NAME = 'xsepnet_norm_factor.mat'

R2P_NET_CHECKPOINT_PATH = './Checkpoint/'
R2P_NET_CHECKPOINT_FILE = 'R2PRIMEnet.pth.tar'
R2P_NET_VALUE_FILE_PATH = './Checkpoint/'
R2P_NET_VALUE_FILE_NAME = 'xsepnet_norm_factor.mat'

RESULT_PATH = CHECKPOINT_PATH + 'Results/'
RESULT_FILE = 'xsepnet_' + INPUT_MAP + '_' + CHECKPOINT_FILE.split('.')[0] + '_'

"""
Physics-parameters
    delta_TE: time gap between multi-echo times
    CF: center frequency (used for Hz -> ppm calculation)
    Dr: relaxometric constrant between R2' and susceptibility
    
    * Input of network must be unit of ppm. So If the input map has unit of radian or Hz, you can change it to ppm by entering physics-params below (only in case of inference).
    ** Ref: Shin, Hyeong-Geol, et al. "χ-separation: Magnetic susceptibility source separation toward iron and myelin mapping in the brain." Neuroimage 240 (2021): 118371.
"""
delta_TE = 0.0056
CF = 123177385
Dr = 114


"""
Network-parameters
    CHANNEL_IN: number of out-channels for first conv layers
    KERNEL_SIZE: kernel size of conv layers
"""
CHANNEL_IN = 32
KERNEL_SIZE = 3


import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--GPU_NUM", default=GPU_NUM)
    parser.add_argument("--INPUT_MAP", default=INPUT_MAP)
    parser.add_argument("--INPUT_UNIT", default=INPUT_UNIT)
    parser.add_argument("--LABEL_EXIST", default=LABEL_EXIST)
    parser.add_argument("--MASK_EXIST", default=MASK_EXIST)
    parser.add_argument("--QSM_RES_GEN_TOGGLE", default=QSM_RES_GEN_TOGGLE)
    parser.add_argument("--RESULT_SAVE_TOGGLE", default=RESULT_SAVE_TOGGLE)

    parser.add_argument("--TEST_PATH", default=TEST_PATH)
    parser.add_argument("--TEST_FILE", default=TEST_FILE)
    
    parser.add_argument("--CHECKPOINT_PATH", default=CHECKPOINT_PATH)
    parser.add_argument("--CHECKPOINT_FILE", default=CHECKPOINT_FILE)
    
    parser.add_argument("--QSM_NET_CHECKPOINT_PATH", default=QSM_NET_CHECKPOINT_PATH)
    parser.add_argument("--QSM_NET_CHECKPOINT_FILE", default=QSM_NET_CHECKPOINT_FILE)
    
    parser.add_argument("--VALUE_FILE_PATH", default=VALUE_FILE_PATH)
    parser.add_argument("--VALUE_FILE_NAME", default=VALUE_FILE_NAME)
    parser.add_argument("--QSM_NET_VALUE_FILE_PATH", default=QSM_NET_VALUE_FILE_PATH)
    parser.add_argument("--QSM_NET_VALUE_FILE_NAME", default=QSM_NET_VALUE_FILE_NAME)
    
    if INPUT_MAP == 'r2s':
        parser.add_argument("--R2P_NET_CHECKPOINT_PATH", default=R2P_NET_CHECKPOINT_PATH)
        parser.add_argument("--R2P_NET_CHECKPOINT_FILE", default=R2P_NET_CHECKPOINT_FILE)

        parser.add_argument("--R2P_NET_VALUE_FILE_PATH", default=R2P_NET_VALUE_FILE_PATH)
        parser.add_argument("--R2P_NET_VALUE_FILE_NAME", default=R2P_NET_VALUE_FILE_NAME)
    
    parser.add_argument("--RESULT_PATH", default=RESULT_PATH)
    parser.add_argument("--RESULT_FILE", default=RESULT_FILE)

    parser.add_argument("--delta_TE", default=delta_TE)
    parser.add_argument("--CF", default=CF)
    parser.add_argument("--Dr", default=Dr)

    parser.add_argument("--CHANNEL_IN", default=CHANNEL_IN)
    parser.add_argument("--KERNEL_SIZE", default=KERNEL_SIZE)
    
    args = parser.parse_args(args=[])
    return args
