'''
#
# Description:
#  Making patch-dataset for x-sepnet training
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# requirements: '*.mat' files including ['cosmos_4d', 'csf_mask_4d', 'local_ppm_hz_4d', 'mask_4d', 'r2_4d', 'r2star_4d', 'xn_cosmos_nnls_4d', 'xp_cosmos_nnls_4d']
#    cosmos_4d: QSMmap
#    mask_4d: brain mask
#    local_f_ppm_4d: local field map (unit: ppm)
#    r2_4d: r2 map
#    r2star_4d: r2* map
#    xn_cosmos_nnls_4d: x-negative map [label of x-sepnet]
#    xp_cosmos_nnls_4d: x-positive map [label of x-sepnet]
#    (every dataset has 4th dimension with multi-head-orientation)
#
#
# Last update: 24.09.20
'''
import scipy.io
import numpy as np
import h5py
import time
import os
import sys
import math
import shutil
import mat73

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
File Path
    FILE_INPUT_PATH: path including input file
    FILE_NAME: name of input file (.mat)
    FILE_RESULT_PATH: path for result
    RESULT_FILE_NAME: name of result file (.hdf5)
    RESULT_VALUE_FILE_NAME = name of values(mean, std) of result file (.mat)
'''
FILE_INPUT_PATH = './Data/subjects/'
FILE_NAME = 'DataFor_xsepnet_ppm_COSMOS_6dir_romeo_arlo.mat'
FILE_RESULT_PATH = './Data/'
RESULT_FILE_NAME = 'xsepnet_train_patch_inplane_largedegree_romeo_arlo.hdf5'
RESULT_FILE = h5py.File(FILE_RESULT_PATH + RESULT_FILE_NAME, 'w')
RESULT_VALUE_FILE_NAME = 'xsepnet_train_patch_norm_factor_inplane_largedegree_romeo_arlo.mat'


'''
Constant Variables
    PS: Patch size
    dir_num: number of directions
    patch_num: Order of Dimension: [x, y, z]
    
    gyro: gyromagnetic ratio
    delta_TE: time gap between multi-echo times
    CF: center frequency (used for Hz -> ppm calculation)
    Dr: relaxometric constrant between R2' and susceptibility
    
    * Input of network must be unit of ppm. So If the input map has unit of Hz, you can change it to ppm by entering physics-params below (only in case of inference).
    ** Ref: Shin, Hyeong-Geol, et al. "χ-separation: Magnetic susceptibility source separation toward iron and myelin mapping in the brain." Neuroimage 240 (2021): 118371.
'''
PS = 64
patch_num = [6, 8, 7]

Dr = 114

if os.path.isdir(FILE_INPUT_PATH + "/.ipynb_checkpoints"):
    shutil.rmtree(FILE_INPUT_PATH + "/.ipynb_checkpoints")
'''
patches for ['cosmos_4d', 'local_f_ppm_4d', 'mask_4d', 'r2_4d', 'r2star_4d', 'xn_cosmos_nnls_4d', 'xp_cosmos_nnls_4d', 'r2prime_4d']
* r2prime_4d: r2' map (which will be made from r2star - r2)
'''
patches_cosmos_sus = []
patches_field = []
patches_mask = []
patches_r2prime = []
patches_r2 = []
patches_r2star = []
patches_x_pos = []
patches_x_neg = []

subject_list = [folder for folder in os.listdir(FILE_INPUT_PATH) if folder.startswith('subj')]
print("Subjects num:", len(subject_list))
print("Subjects:", subject_list)

print('---------------------------------------')
print("*** Patching start !!! ***")

start_time = time.time()
    
for subject in subject_list:
    print('---------------------------------------')
    print('Subject name:', subject)
    
    if subject[-3:] == 'aug':
        print('its augmented data')
        m = scipy.io.loadmat(FILE_INPUT_PATH + subject + "/" + FILE_NAME_2)
    else:
        print('its original data')
        m = scipy.io.loadmat(FILE_INPUT_PATH + subject + "/" + FILE_NAME)

    cosmos_sus = m['cosmos_4d']
    field = m['local_f_ppm_4d']
    mask = m['mask_4d']
    r2 = m['r2_4d']
    r2star = m['r2star_4d']
    x_pos = m['xp_cosmos_nnls_4d']
    x_neg = m['xn_cosmos_nnls_4d']
    
    r2prime = r2star - r2
    r2prime[np.where(r2prime < 0)] = 0
    
    ### Crop brain region tightly ###
    y_max = np.max(np.where(mask[:, :, :, 0] != 0)[0])
    x_max = np.max(np.where(mask[:, :, :, 0] != 0)[1])
    z_max = np.max(np.where(mask[:, :, :, 0] != 0)[2])

    y_min = np.min(np.where(mask[:, :, :, 0] != 0)[0])
    x_min = np.min(np.where(mask[:, :, :, 0] != 0)[1])
    z_min = np.min(np.where(mask[:, :, :, 0] != 0)[2])

    new_y_min = int(y_min-PS/2)
    new_y_max = int(y_max+PS/2)

    new_x_min = int(x_min-PS/2)
    new_x_max = int(x_max+PS/2)

    new_z_min = int(z_min-PS/2)
    new_z_max = int(z_max+PS/2)
    
    print('new y(min, max):', new_y_min, new_y_max)
    print('new x(min, max):', new_x_min, new_x_max)
    print('new z(min, max):', new_z_min, new_z_max)

    if(new_y_min < 0):
        new_y_min = 0

    if(new_z_min < 0):
        new_z_min = 0

    if(new_x_min < 0):
        new_x_min = 0

    if(new_y_max > mask.shape[0]):
        new_y_max = mask.shape[0]

    if(new_x_max > mask.shape[1]):
        new_x_max = mask.shape[1]

    if(new_z_max > mask.shape[2]):
        new_z_max = mask.shape[2]

    print('new y(min, max):', new_y_min, new_y_max)
    print('new x(min, max):', new_x_min, new_x_max)
    print('new z(min, max):', new_z_min, new_z_max)
        
    origin_mask = mask.copy()
    print('Ori size:', np.shape(origin_mask))
    
    cosmos_sus = cosmos_sus[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
    field = field[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
    mask = mask[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
    r2 = r2[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
    r2star = r2star[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
    r2prime = r2prime[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
    x_pos = x_pos[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
    x_neg = x_neg[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
    
    print('New size:', np.shape(mask))
    
    ### Converting Hz maps to ppm ###
    r2prime_in_ppm = r2prime / Dr
    r2star_in_ppm = r2star / Dr
    r2_in_ppm = r2 / Dr
    
    matrix_size = np.shape(mask)
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)];

    print('Matrix size:', matrix_size)
    print('Strides:', strides)

    for direction in range(matrix_size[-1]):
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                for k in range(patch_num[2]):
                    patches_cosmos_sus.append(cosmos_sus[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   direction])
                    patches_field.append(field[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   direction])
                    patches_mask.append(mask[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   direction])
                    patches_r2.append(r2_in_ppm[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   direction])
                    patches_r2prime.append(r2prime_in_ppm[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   direction])
                    patches_r2star.append(r2star_in_ppm[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   direction])
                    patches_x_pos.append(x_pos[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   direction])
                    patches_x_neg.append(x_neg[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   direction])

    print("patch size : " + str(np.shape(patches_mask)))

    
print('---------------------------------------')
print("*** Patching Done !!! ***")
print('---------------------------------------')
print("*** Saving start !!! ***")

del origin_mask
del cosmos_sus
del field
del mask
del r2
del r2star
del r2prime
del x_pos
del x_neg


patches_mask = np.array(patches_mask, dtype='float32', copy=False)
RESULT_FILE.create_dataset('pMask', data=patches_mask)
print('Mask Done')

patches_cosmos_sus = np.array(patches_cosmos_sus, dtype='float32', copy=False)
temp = patches_cosmos_sus[patches_mask > 0]
cosmos_sus_mean = temp.mean(axis=0)
cosmos_sus_std = temp.std(axis=0)
del temp
RESULT_FILE.create_dataset('pCosmosSus', data=patches_cosmos_sus)
del patches_cosmos_sus
print('COSMOS Done')


patches_field = np.array(patches_field, dtype='float32', copy=False)
temp = patches_field[patches_mask > 0]
field_mean = temp.mean(axis=0)
field_std = temp.std(axis=0)
del temp
RESULT_FILE.create_dataset('pField', data=patches_field)
del patches_field
print('Field Done')


patches_r2 = np.array(patches_r2, dtype='float32', copy=False)
temp = patches_r2[patches_mask > 0]
r2_mean = temp.mean(axis=0)
r2_std = temp.std(axis=0)
del temp
RESULT_FILE.create_dataset('pR2', data=patches_r2)
del patches_r2
print('R2 Done')


patches_r2prime = np.array(patches_r2prime, dtype='float32', copy=False)
temp = patches_r2prime[patches_mask > 0]
r2prime_mean = temp.mean(axis=0)
r2prime_std = temp.std(axis=0)
del temp
RESULT_FILE.create_dataset('pR2prime', data=patches_r2prime)
del patches_r2prime
print('R2p Done')


patches_r2star = np.array(patches_r2star, dtype='float32', copy=False)
temp = patches_r2star[patches_mask > 0]
r2star_mean = temp.mean(axis=0)
r2star_std = temp.std(axis=0)
del temp
RESULT_FILE.create_dataset('pR2star', data=patches_r2star)
del patches_r2star
print('R2s Done')

patches_x_pos = np.array(patches_x_pos, dtype='float32', copy=False)
temp = patches_x_pos[patches_mask > 0]
x_pos_mean = temp.mean(axis=0)
x_pos_std = temp.std(axis=0)
del temp
RESULT_FILE.create_dataset('pXpos', data=patches_x_pos)
del patches_x_pos
print('Xpos Done')


patches_x_neg = np.array(patches_x_neg, dtype='float32', copy=False)
temp = patches_x_neg[patches_mask > 0]
x_neg_mean = temp.mean(axis=0)
x_neg_std = temp.std(axis=0)
del temp
RESULT_FILE.create_dataset('pXneg', data=patches_x_neg)
del patches_x_neg
print('Xneg Done')

n_element = np.sum(patches_mask)
print("Final input data size : " + str(np.shape(patches_mask)))

del patches_mask

scipy.io.savemat(FILE_RESULT_PATH + RESULT_VALUE_FILE_NAME,
                 mdict={'cosmos_sus_mean': cosmos_sus_mean, 'cosmos_sus_std': cosmos_sus_std,
                        'field_mean': field_mean, 'field_std': field_std,
                        'r2_mean': r2_mean, 'r2_std':r2_std,
                        'r2prime_mean': r2prime_mean, 'r2prime_std': r2prime_std,
                        'r2star_mean': r2star_mean, 'r2star_std': r2star_std,
                        'x_pos_mean': x_pos_mean, 'x_pos_std': x_pos_std,
                        'x_neg_mean': x_neg_mean, 'x_neg_std': x_neg_std,
                        'n_element': n_element})

RESULT_FILE.close()
    
print("*** Saving Done !!! ***")
print('---------------------------------------')