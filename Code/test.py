'''
#
# Description:
#  Test code of x-sepnet
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 24.11.11
'''
import os
import logging
import glob
import time
import math
import shutil
import random
import scipy.io
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
import logging_helper as logging_helper
from collections import OrderedDict

from utils import *
from network import *
from custom_dataset import *
from test_params import parse

def main():
    args = parse()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.GPU_NUM)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    '''
    ######################
    ### Logger setting ###
    ######################
    '''
    logger = setting_logger(path=args.CHECKPOINT_PATH + 'Results', file_name='test_log.txt', module_type='test')
    
    for key, value in vars(args).items():
        logger.info('{:15s}: {}'.format(key,value))
    
    '''
    #####################################
    ### Network & Data loader setting ###
    #####################################
    - qsm_model: QSMnet
    - model: x-sepnet
    - r2p_model: R2PRIMEnet
    #####################################
    '''
    test_set = test_dataset(args)
    
    ### QSMnet ###
    qsm_model = QSMnet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
    qsm_ckpt_path = args.QSM_NET_CHECKPOINT_PATH + args.QSM_NET_CHECKPOINT_FILE
    qsm_model, qsm_best_epoch = load_pretrained_state_dict(qsm_model, qsm_ckpt_path, logger)
    
    ### xsepnet ###
    model = KAInet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
    model_ckpt_path = args.CHECKPOINT_PATH + args.CHECKPOINT_FILE
    model, best_epoch = load_pretrained_state_dict(model, model_ckpt_path, logger)
        
    ### R2PRIMEnet ###
    if args.INPUT_MAP == 'r2s':
        r2p_model = R2PRIMEnet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
        r2p_ckpt_path = args.R2P_NET_CHECKPOINT_PATH + args.R2P_NET_CHECKPOINT_FILE
        r2p_model, r2p_best_epoch = load_pretrained_state_dict(r2p_model, r2p_ckpt_path, logger)
    else:
        r2p_model = None
        
    if torch.cuda.device_count() > 1:
        logger.info(f'Multi GPU - num: {torch.cuda.device_count()} - are used')
        qsm_model = nn.DataParallel(qsm_model).to(device)
        model = nn.DataParallel(model).to(device)
        r2p_model = nn.DataParallel(r2p_model).to(device)
        
    logger.info(f'Best epoch: {best_epoch}\n')

    '''
    #################
    ### Inference ###
    #################
    '''
    inference(args, logger, device, test_set, qsm_model, model, r2p_model)


def inference(args, logger, device, test_set, qsm_model, model, r2p_model):
    createDirectory(args.RESULT_PATH)
    
    loss_total_list = []
    pos_NRMSE_total_list = []
    neg_NRMSE_total_list = []
    pos_PSNR_total_list = []
    neg_PSNR_total_list = []
    pos_SSIM_total_list = []
    neg_SSIM_total_list = []
    time_total_list = []
    
    logger.info("------ Testing is started ------")
    
    for idx in range(0, len(args.TEST_FILE)):
        subj_name = args.TEST_FILE[idx].split('_')[0]
    
        with torch.no_grad():
            qsm_model.eval()
            model.eval()
            if args.INPUT_MAP == 'r2s':
                r2p_model.eval()
    
            loss_list = []
            pos_nrmse_list = []
            pos_psnr_list = []
            pos_ssim_list = []
    
            neg_nrmse_list = []
            neg_psnr_list = []
            neg_ssim_list = []
    
            time_list = []
    
            input_field = test_set.field[idx]        
            input_r2p = test_set.r2prime[idx]
            input_r2s = test_set.r2star[idx]
            input_mask = test_set.mask[idx]
            input_mask_for_eval_pos = test_set.mask_for_eval_pos[idx]
            input_mask_for_eval_neg = test_set.mask_for_eval_neg[idx]
            if args.LABEL_EXIST is True:
                label_qsm = test_set.sus[idx]
                label_x_pos = test_set.x_pos[idx]
                label_x_neg = test_set.x_neg[idx]
            if args.QSM_RES_GEN_TOGGLE is True:
                input_qsm = test_set.susQSMnet[idx]
    
            matrix_size = test_set.matrix_size[idx]
            
            if len(matrix_size) == 3:
                ### Case of single head-orientation: expanding dim ###
                matrix_size_list = list(matrix_size)
                matrix_size_list.append(1)
                matrix_size = tuple(matrix_size_list)
                
                input_field = np.expand_dims(input_field, 3)
                input_r2s = np.expand_dims(input_r2s, 3)
                input_r2p = np.expand_dims(input_r2p, 3)
                input_mask = np.expand_dims(input_mask, 3)
                input_mask_for_eval_pos = np.expand_dims(input_mask_for_eval_pos, 3)
                input_mask_for_eval_neg = np.expand_dims(input_mask_for_eval_neg, 3)
                if args.LABEL_EXIST is True:
                    label_qsm = np.expand_dims(label_qsm, 3)
                    label_x_pos = np.expand_dims(label_x_pos, 3)
                    label_x_neg = np.expand_dims(label_x_neg, 3)
                if args.QSM_RES_GEN_TOGGLE is True:
                    input_qsm = np.expand_dims(input_qsm, 3)
                    
            input_field_map = np.zeros(matrix_size)
            input_r2s_map = np.zeros(matrix_size)
            input_r2p_map = np.zeros(matrix_size)
            pred_sus_map = np.zeros(matrix_size)
            pred_x_pos_map = np.zeros(matrix_size)
            pred_x_neg_map = np.zeros(matrix_size)
            label_qsm_map = np.zeros(matrix_size)
            label_x_pos_map = np.zeros(matrix_size)
            label_x_neg_map = np.zeros(matrix_size)
            mask_map = np.zeros(matrix_size)
            mask_eval_pos_map = np.zeros(matrix_size)
            mask_eval_neg_map = np.zeros(matrix_size)
            pos_ssim_maps = np.zeros(matrix_size)
            neg_ssim_maps = np.zeros(matrix_size)
            
            for direction in range(matrix_size[-1]):
                '''
                #################################################
                ### Setting dataset & normalization & masking ###
                #################################################
                '''
                local_f_batch = torch.tensor(input_field[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                local_f_batch = (local_f_batch - test_set.field_mean) / test_set.field_std
                
                m_batch = torch.tensor(input_mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
                mask_for_eval_pos_batch = torch.tensor(input_mask_for_eval_pos[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
                mask_for_eval_neg_batch = torch.tensor(input_mask_for_eval_neg[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
                
                local_f_batch = local_f_batch * m_batch
    
                r2p_batch = torch.tensor(input_r2p[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                r2p_batch = (r2p_batch - test_set.r2prime_mean) / test_set.r2prime_std
                r2p_batch = r2p_batch * m_batch
                
                r2s_batch = torch.tensor(input_r2s[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                r2s_batch = (r2s_batch - test_set.r2star_mean) / test_set.r2star_std
                r2s_batch = r2s_batch * m_batch
                
                if args.LABEL_EXIST is True:
                    x_pos_batch = torch.tensor(label_x_pos[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                    x_neg_batch = torch.tensor(label_x_neg[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                    x_pos_batch = (x_pos_batch - test_set.x_pos_mean) / test_set.x_pos_std
                    x_neg_batch = (x_neg_batch - test_set.x_neg_mean) / test_set.x_neg_std
                    label_batch = torch.cat((x_pos_batch, x_neg_batch), 1)
                    
                if args.QSM_RES_GEN_TOGGLE is True:
                    input_qsm_batch = torch.tensor(input_qsm[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                    input_qsm_batch = (input_qsm_batch - test_set.pre_sus_mean) / test_set.pre_sus_std
    
                '''
                ####################################
                ### Pre-trained QSMnet Inference ###
                ####################################
                - input: local field map
                - output: COSMOS-quality QSM map
                ####################################
                '''
                if args.QSM_RES_GEN_TOGGLE is True:
                    pred_cosmos = input_qsm_batch
                else:
                    pred_cosmos = qsm_model(local_f_batch)
                pred_cosmos_batch = pred_cosmos * m_batch
                
                '''
                ################################################################
                ###                   x-sepnet Inference                     ###
                ################################################################
                * x-sepnet-R2'
                  - input: concat(QSMmap, local field map, r2star map, r2 map)
                  - output: concat(x-para map, x-dia map)
                * x-sepnet-R2*
                  - input: concat(QSMmap, local field map, r2star map)
                  - output: concat(x-para map, x-dia map)
                ################################################################
                '''
                if args.INPUT_MAP == 'r2s':
                    pred_r2p = r2p_model(r2s_batch)
                    pred_r2p_batch = pred_r2p * m_batch
                    
                if args.INPUT_MAP == 'r2p':
                    input_batch = torch.cat((pred_cosmos_batch, local_f_batch, r2p_batch), 1)
                elif args.INPUT_MAP == 'r2s':
                    input_batch = torch.cat((pred_cosmos_batch, local_f_batch, pred_r2p_batch), 1)
    
                start_time = time.time()
                pred = model(input_batch)
                inferenc_time = time.time() - start_time
                time_list.append(inferenc_time)
                time_total_list.append(inferenc_time)
    
                ### De-normalization ###
                pred_cosmos = ((pred_cosmos.cpu() * test_set.sus_std) + test_set.sus_mean).to(device).squeeze()
                pred_x_pos = ((pred[:, 0, ...].cpu() * test_set.x_pos_std) + test_set.x_pos_mean).to(device).squeeze()
                pred_x_neg = ((pred[:, 1, ...].cpu() * test_set.x_neg_std) + test_set.x_neg_mean).to(device).squeeze()
    
                ### Zero-truncation ###
                pred_x_pos[pred_x_pos < 0] = 0;
                pred_x_neg[pred_x_neg < 0] = 0;
    
                if args.LABEL_EXIST is True:
                    label_qsm_for_metric = torch.tensor(label_qsm[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
                    label_x_pos_for_metric = torch.tensor(label_x_pos[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
                    label_x_neg_for_metric = torch.tensor(label_x_neg[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float).squeeze()
    
                    ### Metric calculation ###
                    l1loss = l1_loss(pred, label_batch)
                    
                    pos_nrmse = NRMSE(pred_x_pos, label_x_pos_for_metric, mask_for_eval_pos_batch)
                    pos_psnr = PSNR(pred_x_pos, label_x_pos_for_metric, mask_for_eval_pos_batch)
                    pos_ssim, pos_ssim_map = SSIM(pred_x_pos, label_x_pos_for_metric, mask_for_eval_pos_batch)
    
                    neg_nrmse = NRMSE(pred_x_neg, label_x_neg_for_metric, mask_for_eval_neg_batch)
                    neg_psnr = PSNR(pred_x_neg, label_x_neg_for_metric, mask_for_eval_neg_batch)
                    neg_ssim, neg_ssim_map = SSIM(pred_x_neg, label_x_neg_for_metric, mask_for_eval_neg_batch)
                    mask_map[..., direction] = m_batch.cpu()
                    mask_eval_pos_map[..., direction] = mask_for_eval_pos_batch.cpu()
                    mask_eval_neg_map[..., direction] = mask_for_eval_neg_batch.cpu()
    
                    loss_list.append(l1loss.item())
                    pos_nrmse_list.append(pos_nrmse)
                    pos_psnr_list.append(pos_psnr)
                    pos_ssim_list.append(pos_ssim)
                    neg_nrmse_list.append(neg_nrmse)
                    neg_psnr_list.append(neg_psnr)
                    neg_ssim_list.append(neg_ssim)
                    
                    loss_total_list.append(l1loss.item())
                    pos_NRMSE_total_list.append(pos_nrmse)
                    pos_PSNR_total_list.append(pos_psnr)
                    pos_SSIM_total_list.append(pos_ssim)
                    neg_NRMSE_total_list.append(neg_nrmse)
                    neg_PSNR_total_list.append(neg_psnr)
                    neg_SSIM_total_list.append(neg_ssim)
                    
                    label_qsm_map[..., direction] = (label_qsm_for_metric.cpu() * m_batch.cpu())
                    label_x_pos_map[..., direction] = (label_x_pos_for_metric.cpu() * m_batch.cpu())
                    label_x_neg_map[..., direction] = (label_x_neg_for_metric.cpu() * m_batch.cpu())
                    
                input_field_map[..., direction] = local_f_batch.squeeze().cpu()
                
                input_field_map[..., direction] = input_field[..., direction]
                input_r2s_map[..., direction] = input_r2s[..., direction]
                input_r2p_map[..., direction] = input_r2p[..., direction]
                pred_sus_map[..., direction] = (pred_cosmos.cpu() * m_batch.cpu())
                pred_x_pos_map[..., direction] = (pred_x_pos.cpu() * m_batch.cpu())
                pred_x_neg_map[..., direction] = (pred_x_neg.cpu() * m_batch.cpu())
                
                torch.cuda.empty_cache();
                
            if args.LABEL_EXIST is True:
                test_loss = np.mean(loss_list)
                pos_NRMSE_mean = np.mean(pos_nrmse_list)
                pos_PSNR_mean = np.mean(pos_psnr_list)
                pos_SSIM_mean = np.mean(pos_ssim_list)
                neg_NRMSE_mean = np.mean(neg_nrmse_list)
                neg_PSNR_mean = np.mean(neg_psnr_list)
                neg_SSIM_mean = np.mean(neg_ssim_list)
    
                pos_NRMSE_std = np.std(pos_nrmse_list)
                pos_PSNR_std = np.std(pos_psnr_list)
                pos_SSIM_std = np.std(pos_ssim_list)
                neg_NRMSE_std = np.std(neg_nrmse_list)
                neg_PSNR_std = np.std(neg_psnr_list)
                neg_SSIM_std = np.std(neg_ssim_list)
                total_time = np.mean(time_list)
    
                logger.info(f'{subj_name}  Xpos - NRMSE: {pos_NRMSE_mean:.4f}, {pos_NRMSE_std:.4f}  PSNR: {pos_PSNR_mean:.4f}, {pos_PSNR_std:.4f}  SSIM: {pos_SSIM_mean:.4f}, {pos_SSIM_std:.4f}')
                logger.info(f'{subj_name}  Xneg - NRMSE: {neg_NRMSE_mean:.4f}, {neg_NRMSE_std:.4f}  PSNR: {neg_PSNR_mean:.4f}, {neg_PSNR_std:.4f}  SSIM: {neg_SSIM_mean:.4f}, {neg_SSIM_std:.4f}')
    
                if args.RESULT_SAVE_TOGGLE == True:
                    scipy.io.savemat(args.RESULT_PATH + args.RESULT_FILE + subj_name + '.mat',
                                     mdict={'mask': mask_map,
                                            'label_qsm': label_qsm_map,
                                            'label_x_pos': label_x_pos_map,
                                            'label_x_neg': label_x_neg_map,
                                            'pred_qsm': pred_sus_map,
                                            'pred_x_pos': pred_x_pos_map,
                                            'pred_x_neg': pred_x_neg_map,
                                            'posNRMSEmean': pos_NRMSE_mean,
                                            'posPSNRmean': pos_PSNR_mean,
                                            'posSSIMmean': pos_SSIM_mean,
                                            'negNRMSEmean': neg_NRMSE_mean,
                                            'negPSNRmean': neg_PSNR_mean,
                                            'negSSIMmean': neg_SSIM_mean,
                                            'posNRMSEstd': pos_NRMSE_std,
                                            'posPSNRstd': pos_PSNR_std,
                                            'posSSIMstd': pos_SSIM_std,
                                            'negNRMSEstd': neg_NRMSE_std,
                                            'negPSNRstd': neg_PSNR_std,
                                            'negSSIMstd': neg_SSIM_std})
    
            elif args.LABEL_EXIST is False:
                total_time = np.mean(time_list)
                if args.RESULT_SAVE_TOGGLE == True:
                    scipy.io.savemat(args.RESULT_PATH + args.RESULT_FILE + subj_name + '.mat',
                                     mdict={'pred_qsm': pred_sus_map,
                                            'pred_x_pos': pred_x_pos_map,
                                            'pred_x_neg': pred_x_neg_map})
    
    if args.LABEL_EXIST == True:
        total_loss_mean = np.mean(loss_total_list)
        total_pos_NRMSE_mean = np.mean(pos_NRMSE_total_list)
        total_pos_PSNR_mean = np.mean(pos_PSNR_total_list)
        total_pos_SSIM_mean = np.mean(pos_SSIM_total_list)
        total_neg_NRMSE_mean = np.mean(neg_NRMSE_total_list)
        total_neg_PSNR_mean = np.mean(neg_PSNR_total_list)
        total_neg_SSIM_mean = np.mean(neg_SSIM_total_list)
        
        total_loss_std = np.std(loss_total_list)
        total_pos_NRMSE_std = np.std(pos_NRMSE_total_list)
        total_pos_PSNR_std = np.std(pos_PSNR_total_list)
        total_pos_SSIM_std = np.std(pos_SSIM_total_list)
        total_neg_NRMSE_std = np.std(neg_NRMSE_total_list)
        total_neg_PSNR_std = np.std(neg_PSNR_total_list)
        total_neg_SSIM_std = np.std(neg_SSIM_total_list)
        
        logger.info(f'Total loss: {total_loss_mean:.4f}, {total_loss_std:.4f}')
        logger.info(f'Total Xpos - NRMSE: {total_pos_NRMSE_mean:.4f}, {total_pos_NRMSE_std:.4f}  PSNR: {total_pos_PSNR_mean:.4f}, {total_pos_PSNR_std:.4f}  SSIM: {total_pos_SSIM_mean:.4f}, {total_pos_SSIM_std:.4f}')
        logger.info(f'Total Xneg - NRMSE: {total_neg_NRMSE_mean:.4f}, {total_neg_NRMSE_std:.4f}  PSNR: {total_neg_PSNR_mean:.4f}, {total_neg_PSNR_std:.4f}  SSIM: {total_neg_SSIM_mean:.4f}, {total_neg_SSIM_std:.4f}')
        
    logger.info(f'Total inference time: {np.mean(time_total_list)}')
    logger.info("------ Testing is finished ------\n\n")
    

if __name__ == "__main__":
    main()