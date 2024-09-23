'''
#
# Description:
#  Dataset codes for x-sepnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 24.09.20
'''
import math
import h5py
import scipy.io
import scipy.ndimage
import torch
import mat73
import numpy as np
from utils import *

    
class train_dataset():
    def __init__(self, args):
        data_file = h5py.File(args.TRAIN_PATH + args.TRAIN_FILE, "r")
        value_file = scipy.io.loadmat(args.VALUE_PATH + args.VALUE_FILE)
        pre_value_file = scipy.io.loadmat(args.VALUE_PATH + args.PRE_VALUE_FILE)
        
        self.input_map = args.INPUT_MAP

        self.field = data_file['pField']
        self.r2prime = data_file['pR2prime']
        self.r2star = data_file['pR2star']
        self.r2 = data_file['pR2']
        
        self.x_pos = data_file['pXpos']
        self.x_neg = data_file['pXneg']
        self.mask = data_file['pMask']
        self.sus = data_file['pCosmosSus']
        
        self.field_mean = value_file['field_mean'].item()
        self.field_std = value_file['field_std'].item()
        
        self.r2prime_mean = value_file['r2prime_mean'].item()
        self.r2prime_std = value_file['r2prime_std'].item()
        
        self.r2star_mean = value_file['r2star_mean'].item()
        self.r2star_std = value_file['r2star_std'].item()
        
        self.r2_mean = value_file['r2_mean'].item()
        self.r2_std = value_file['r2_std'].item()
        
        self.x_pos_mean = value_file['x_pos_mean'].item()
        self.x_pos_std = value_file['x_pos_std'].item()
        
        self.x_neg_mean = value_file['x_neg_mean'].item()
        self.x_neg_std = value_file['x_neg_std'].item()
        
        self.sus_mean = value_file['cosmos_sus_mean'].item()
        self.sus_std = value_file['cosmos_sus_std'].item()
        
        self.pre_field_mean = pre_value_file['field_mean'].item()
        self.pre_field_std = pre_value_file['field_std'].item()
        
        self.pre_sus_mean = pre_value_file['cosmos_sus_mean'].item()
        self.pre_sus_std = pre_value_file['cosmos_sus_std'].item()    
        
    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        # dim: [1, 64, 64, 64]
        local_f_batch = torch.tensor(self.field[idx], dtype=torch.float).unsqueeze(0)
        r2p_batch = torch.tensor(self.r2prime[idx], dtype=torch.float).unsqueeze(0)
        r2s_batch = torch.tensor(self.r2star[idx,...], dtype=torch.float).unsqueeze(0)
        r2_batch = torch.tensor(self.r2[idx,...], dtype=torch.float).unsqueeze(0)
        x_pos_batch = torch.tensor(self.x_pos[idx], dtype=torch.float).unsqueeze(0)
        x_neg_batch = torch.tensor(self.x_neg[idx], dtype=torch.float).unsqueeze(0)
        m_batch = torch.tensor(self.mask[idx], dtype=torch.float).unsqueeze(0)
        pre_local_f_batch = torch.tensor(self.field[idx], dtype=torch.float).unsqueeze(0)
        sus_batch = torch.tensor(self.sus[idx], dtype=torch.float).unsqueeze(0)

        ### Normalization ###
        local_f_batch = ((local_f_batch - self.field_mean) / self.field_std)
        r2p_batch = ((r2p_batch - self.r2prime_mean) / self.r2prime_std)
        r2s_batch = ((r2s_batch - self.r2star_mean) / self.r2star_std)
        r2_batch = ((r2_batch - self.r2_mean) / self.r2_std)
        x_pos_batch = ((x_pos_batch - self.x_pos_mean) / self.x_pos_std)
        x_neg_batch = ((x_neg_batch - self.x_neg_mean) / self.x_neg_std)
        pre_local_f_batch = ((pre_local_f_batch - self.pre_field_mean) / self.pre_field_std)
        sus_batch = ((sus_batch - self.sus_mean) / self.sus_std)

        return idx, local_f_batch, r2p_batch, r2s_batch, r2_batch, x_pos_batch, x_neg_batch, m_batch, pre_local_f_batch, sus_batch

        
class valid_dataset():
    def __init__(self, args):
        try:
            data_file = scipy.io.loadmat(args.VALID_PATH + args.VALID_FILE)
        except:
            data_file = mat73.loadmat(args.VALID_PATH + args.VALID_FILE)

        value_file = scipy.io.loadmat(args.VALUE_PATH + args.VALUE_FILE)
        pre_value_file = scipy.io.loadmat(args.VALUE_PATH + args.PRE_VALUE_FILE)
            
        r2 = data_file['r2_4d']
        r2star = data_file['r2star_4d']
        r2prime = r2star - r2
        r2prime[np.where(r2prime < 0)] = 0
        
        Dr = args.Dr
        r2star_in_ppm = r2star / Dr
        r2prime_in_ppm = r2prime / Dr
        r2_in_ppm = r2 / Dr
        
        if args.INPUT_UNIT == 'Hz':
            ### Converting Hz maps to ppm ###
            print('Input map unit has been changed (hz -> ppm)')
            field = data_file['local_f_hz_4d']

            CF = args.CF

            field_in_ppm = field / CF * 1e6
        elif args.INPUT_UNIT == 'radian':
            print('Input map unit has been changed (radian -> ppm)')
            field = data_file['local_f_4d']
            
            delta_TE = args.delta_TE
            CF = args.CF
            
            field_in_ppm = -1 * field / (2*math.pi*delta_TE) / CF * 1e6
        elif args.INPUT_UNIT == 'ppm':
            field = data_file['local_f_ppm_4d']

            field_in_ppm = field
        else:
            raise Exception('The unit of input must be one of [Hz, radian, ppm]. Check the unit of input')
        
        self.field = field_in_ppm
        self.r2prime = r2prime_in_ppm
        self.r2star = r2star_in_ppm
        self.r2 = r2_in_ppm
        self.x_pos = data_file['xp_cosmos_nnls_4d']
        self.x_neg = data_file['xn_cosmos_nnls_4d']
        self.mask = data_file['mask_4d']
        self.sus = data_file['cosmos_4d']
        
        self.field_mean = value_file['field_mean'].item()
        self.field_std = value_file['field_std'].item()
        
        self.r2prime_mean = value_file['r2prime_mean'].item()
        self.r2prime_std = value_file['r2prime_std'].item()
        
        self.r2star_mean = value_file['r2star_mean'].item()
        self.r2star_std = value_file['r2star_std'].item()
        
        self.r2_mean = value_file['r2_mean'].item()
        self.r2_std = value_file['r2_std'].item()
        
        self.x_pos_mean = value_file['x_pos_mean'].item()
        self.x_pos_std = value_file['x_pos_std'].item()
        
        self.x_neg_mean = value_file['x_neg_mean'].item()
        self.x_neg_std = value_file['x_neg_std'].item()
        
        self.sus_mean = value_file['cosmos_sus_mean'].item()
        self.sus_std = value_file['cosmos_sus_std'].item()
        
        self.pre_field_mean = pre_value_file['field_mean'].item()
        self.pre_field_std = pre_value_file['field_std'].item()
        
        self.pre_sus_mean = pre_value_file['cosmos_sus_mean'].item()
        self.pre_sus_std = pre_value_file['cosmos_sus_std'].item()     

        self.matrix_size = self.mask.shape
            

class test_dataset():
    def __init__(self, args):
        value_file = scipy.io.loadmat(args.VALUE_FILE_PATH + args.VALUE_FILE_NAME)
        pre_value_file = scipy.io.loadmat(args.QSM_NET_VALUE_FILE_PATH + args.QSM_NET_VALUE_FILE_NAME)

        self.field = []
        self.r2prime = []
        self.r2star = []
        self.r2 = []
        self.sus = []
        self.susQSMnet = []
        self.x_pos = []
        self.x_neg = []
        self.mask = []
        self.mask_for_eval_pos = []
        self.mask_for_eval_neg = []
        self.matrix_size = []
        
        self.field_mean = value_file['field_mean'].item()
        self.field_std = value_file['field_std'].item()
        
        self.r2prime_mean = value_file['r2prime_mean'].item()
        self.r2prime_std = value_file['r2prime_std'].item()
        
        self.r2star_mean = value_file['r2star_mean'].item()
        self.r2star_std = value_file['r2star_std'].item()
        
        self.x_pos_mean = value_file['x_pos_mean'].item()
        self.x_pos_std = value_file['x_pos_std'].item()
        
        self.x_neg_mean = value_file['x_neg_mean'].item()
        self.x_neg_std = value_file['x_neg_std'].item()
        
        self.sus_mean = pre_value_file['cosmos_sus_mean'].item()
        self.sus_std = pre_value_file['cosmos_sus_std'].item()
        
        self.pre_field_mean = pre_value_file['field_mean'].item()
        self.pre_field_std = pre_value_file['field_std'].item()
        
        self.pre_sus_mean = pre_value_file['cosmos_sus_mean'].item()
        self.pre_sus_std = pre_value_file['cosmos_sus_std'].item()
                
        for i in range(0, len(args.TEST_FILE)):
            try:
                data_file = scipy.io.loadmat(args.TEST_PATH + args.TEST_FILE[i])
            except:
                data_file = mat73.loadmat(args.TEST_PATH + args.TEST_FILE[i])
        
            if args.MASK_EXIST == 'CSF' or args.MASK_EXIST == 'None' or args.MASK_EXIST == 'Vessel':
                r2 = data_file['r2_4d']
                r2star = data_file['r2star_4d']
                r2prime = r2star - r2
                r2prime[np.where(r2prime < 0)] = 0
                
            if args.INPUT_UNIT == 'Hz':
                ### Converting Hz to ppm ###
                print('Input map unit has been changed (hz -> ppm)')
                field = data_file['local_f_hz_4d']

                CF = args.CF
                Dr = args.Dr

                field_in_ppm = field / CF * 1e6
                r2star_in_ppm = r2star / Dr
                r2prime_in_ppm = r2prime / Dr
            elif args.INPUT_UNIT == 'radian':
                ### Converting radian to ppm ###
                print('Input map unit has been changed (radian -> ppm)')
                field = data_file['local_f_4d']

                delta_TE = args.delta_TE
                CF = args.CF
                Dr = args.Dr

                field_in_ppm = field / (2*math.pi*delta_TE) / CF * 1e6
                r2star_in_ppm = r2star / Dr
                r2prime_in_ppm = r2prime / Dr
            elif args.INPUT_UNIT == 'ppm':
                field = data_file['local_f_ppm_4d']

                Dr = args.Dr

                field_in_ppm = field
                if args.MASK_EXIST == 'CSF' or args.MASK_EXIST == 'None' or args.MASK_EXIST == 'Vessel':
                    r2star_in_ppm = r2star / Dr
                r2prime_in_ppm = r2prime / Dr
            else:
                raise Exception('The unit of input must be one of [Hz, radian, ppm]. Check the unit of input')
            
            self.field.append(crop_img_16x(field_in_ppm))
            self.r2prime.append(crop_img_16x(r2prime_in_ppm))
            self.r2star.append(crop_img_16x(r2star_in_ppm))
            self.mask.append(crop_img_16x(data_file['mask_4d']))
            self.mask_for_eval_pos.append(crop_img_16x(data_file['mask_4d']))
            self.mask_for_eval_neg.append(crop_img_16x(data_file['mask_4d']))
            
            if args.QSM_RES_GEN_TOGGLE is True:
                self.susQSMnet.append(crop_img_16x(data_file['cosmosQSMnet_4d']))
            
            if args.LABEL_EXIST is True:
                if args.QSM_RES_GEN_TOGGLE == True:
                    self.sus.append(crop_img_16x(data_file['cosmosQSMnet_4d']))
                else:
                    self.sus.append(crop_img_16x(data_file['cosmos_4d']))
                self.x_pos.append(crop_img_16x(data_file['xp_cosmos_nnls_4d']))
                self.x_neg.append(crop_img_16x(data_file['xn_cosmos_nnls_4d']))

            if args.MASK_EXIST == 'CSF':
                subj_name = args.TEST_FILE[i].split('_')[0]
                csf_mask_file = scipy.io.loadmat(args.TEST_PATH + subj_name + '_csf_mask_for_metric.mat')

                csf_mask_only = crop_img_16x(csf_mask_file['CSF_mask_4d'])
                csf_mask_only = (csf_mask_only == 0)
                mask_wo_csf = self.mask[i] * csf_mask_only
                
                self.mask_for_eval_pos[i] = mask_wo_csf
                self.mask_for_eval_neg[i] = mask_wo_csf
            elif args.MASK_EXIST == 'Vessel':
                subj_name = args.TEST_FILE[i].split('_')[0]

                try:
                    vessel_mask_file = scipy.io.loadmat(args.TEST_PATH + subj_name + '_csf_mask_for_metric_vessel_verFinal.mat')
                except:
                    vessel_mask_file = mat73.loadmat(args.TEST_PATH + subj_name + '_csf_mask_for_metric_vessel_verFinal.mat')
                
                csf_mask_only = crop_img_16x(vessel_mask_file['CSF_mask_4d'])
                csf_mask_only = (csf_mask_only == 0)
                mask_wo_csf = self.mask[i] * csf_mask_only
                
                pos_vessel_mask_only = crop_img_16x(vessel_mask_file['x_pos_vessel_mask_4d'])
                neg_vessel_mask_only = crop_img_16x(vessel_mask_file['x_neg_vessel_mask_4d'])
                
                pos_vessel_mask_only = (pos_vessel_mask_only == 0)
                neg_vessel_mask_only = (neg_vessel_mask_only == 0)
                
                pos_mask_wo_vessel = mask_wo_csf * pos_vessel_mask_only
                neg_mask_wo_vessel = mask_wo_csf * neg_vessel_mask_only

                self.mask_for_eval_pos[i] = pos_mask_wo_vessel
                self.mask_for_eval_neg[i] = neg_mask_wo_vessel
                
            self.matrix_size.append(crop_img_16x(data_file['mask_4d']).shape)