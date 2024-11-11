'''
#
# Description:
#  Training code of x-sepnet concat version
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 24.09.20
'''
import os
import logging
import glob
import time
import math
import shutil
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

import logging_helper as logging_helper
from utils import *
from network import *
from custom_dataset import *
from train_params import parse

def main():
    args = parse()
    writer = SummaryWriter(args.CHECKPOINT_PATH + 'runs/')
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.GPU_NUM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    createDirectory(args.CHECKPOINT_PATH)
    createDirectory(args.CHECKPOINT_PATH+ 'Results')
    
    '''
    ######################
    ### Logger setting ###
    ######################
    '''
    logger = setting_logger(path=args.CHECKPOINT_PATH + 'Results', file_name='train_log.txt', module_type='train')
    
    for key, value in vars(args).items():
        logger.info('{:15s}: {}'.format(key,value))
    
    '''
    ####################
    ### Seed Setting ###
    ####################
    '''
    g = setting_seed(seed=args.SEED)
    
    '''
    #####################################
    ### Network & Data loader setting ###
    #####################################
    '''
    train_set = train_dataset(args)
    valid_set = valid_dataset(args)
        
    pre_model = QSMnet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
    model = KAInet(channel_in=args.CHANNEL_IN, kernel_size=args.KERNEL_SIZE).to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f'Multi GPU - num: {torch.cuda.device_count()} - are used')
        model = nn.DataParallel(model).to(device)
    
    '''
    ###########################################
    ###  Pre-trained net (QSMnet) Setting   ###
    ###########################################
    '''
    if args.PRE_NET_CHECKPOINT_PATH is not None:
        logger.info(f'pre-trained net (QSMnet(+)): {args.PRE_NET_CHECKPOINT_PATH}')
        PRE_NET_WEIGHT_NAME = args.PRE_NET_CHECKPOINT_PATH + args.PRE_NET_CHECKPOINT_FILE

        pre_model, qsm_best_epoch = load_pretrained_state_dict(pre_model, PRE_NET_WEIGHT_NAME, logger)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98, last_epoch=-1)

    train(args, logger, writer, device, train_set, valid_set, pre_model, model, g, optimizer, scheduler)


def train(args, logger, writer, device, train_set, valid_set, pre_model, model, g, optimizer, scheduler):
    '''
    #########################
    ### Variables setting ###
    #########################
    '''
    step = 0
    train_loss = []
    valid_loss = []
    ori_valid_loss = []
    nrmse = []
    psnr = []
    ssim = []
    best_loss = math.inf; ori_best_loss = math.inf; best_nrmse = math.inf; best_psnr = -math.inf; best_ssim = -math.inf;
    best_epoch_loss = 0; ori_best_epoch_loss = 0; best_epoch_nrmse = 0; best_epoch_psnr = 0; best_epoch_ssim = 0;

    ### shuffle#######################################
    train_loader = DataLoader(train_set, batch_size = args.BATCH_SIZE, shuffle = False, num_workers = 1, worker_init_fn = seed_worker, generator = g)
    logger.info(f'Num of batches: {len(train_set)}')
    
    start_time = time.time()
    
    logger.info("------ Training is started ------")
    for epoch in tqdm(range(args.TRAIN_EPOCH)):
        epoch_time = time.time()
        
        train_loss_list = []
        train_l1loss_list = []
        train_gdloss_list = []
        train_md_f_loss_list = []
        train_md_q_loss_list = []
        train_md_r_loss_list = []
        
        valid_loss_list = []
        ori_valid_loss_list = []
        nrmse_list = []
        psnr_list = []
        ssim_list = []
        ori_nrmse_list = []
        pos_nrmse_list = []
        neg_nrmse_list = []
        pos_psnr_list = []
        neg_psnr_list = []
        pos_ssim_list = []
        neg_ssim_list = []
        ori_pos_nrmse_list = []
        ori_neg_nrmse_list = []
        sus_nrmse_list = []

        temp_num = 1
        
        for train_data in tqdm(train_loader):
            temp_num += 1
            if temp_num > 10:
                break
                
            pre_model.eval()
            model.train()
            
            index = train_data[0]
            local_f_batch = train_data[1].to(device)
            r2prime_batch = train_data[2].to(device)
            r2star_batch = train_data[3].to(device)
            r2_batch = train_data[4].to(device)
            x_pos_batch = train_data[5].to(device)
            x_neg_batch = train_data[6].to(device)
            m_batch = train_data[7].to(device)
            pre_local_f_batch = train_data[8].to(device)
            sus_batch = train_data[9].to(device)
            
            ### Brain masking ###
            local_f_batch = local_f_batch * m_batch
            r2prime_batch = r2prime_batch * m_batch
            r2star_batch = r2star_batch * m_batch
            r2_batch = r2_batch * m_batch
            pre_local_f_batch = pre_local_f_batch * m_batch
            sus_batch = sus_batch * m_batch
            
            '''
            #############################################################
            ### Extract cosmos-quality QSMmap from pre-trained QSMNet ###
            #############################################################
            '''
            with torch.no_grad():
                pred_cosmos = pre_model(pre_local_f_batch)
            pred_cosmos_batch = (pred_cosmos * train_set.pre_sus_std) + train_set.pre_sus_mean
            pred_cosmos_batch = pred_cosmos_batch * m_batch
    
            '''
            ################################################################
            ###                   x-sepnet Training                      ###
            ################################################################
            * x-sepnet-R2'
              - input: concat(QSMmap, local field map, r2prime map)
              - output: concat(x-para map, x-dia map)
            * x-sepnet-R2*
              - input: concat(QSMmap, local field map, r2star map)
              - output: concat(x-para map, x-dia map)
            ################################################################
            '''
            pred_cosmos_batch = ((pred_cosmos_batch - train_set.sus_mean) / train_set.sus_std)
            pred_cosmos_batch = pred_cosmos_batch * m_batch
            
            if args.INPUT_MAP == 'r2p':
                input_batch = torch.cat((pred_cosmos_batch, local_f_batch, r2prime_batch), 1)
            elif args.INPUT_MAP == 'r2s':
                input_batch = torch.cat((pred_cosmos_batch, local_f_batch, r2star_batch), 1)
            label_batch = torch.cat((x_pos_batch, x_neg_batch), 1)
    
            pred = model(input_batch)        
    
            if args.INPUT_MAP == 'r2p':
                loss, l1loss, gdloss, md_f_loss, md_q_loss, md_r_loss = total_loss(args, pred, label_batch, pred_cosmos_batch, r2prime_batch, r2prime_batch, local_f_batch, m_batch,
                                                                        train_set.x_pos_mean, train_set.x_pos_std,
                                                                        train_set.x_neg_mean, train_set.x_neg_std,
                                                                        train_set.sus_mean, train_set.sus_std,
                                                                        train_set.field_mean, train_set.field_std,
                                                                        train_set.r2prime_mean, train_set.r2prime_std,
                                                                        train_set.r2_mean, train_set.r2_std,
                                                                        train_set.r2prime_mean, train_set.r2prime_std)
            elif args.INPUT_MAP == 'r2s':
                loss, l1loss, gdloss, md_f_loss, md_q_loss = total_loss_r2s(args, pred, label_batch, pred_cosmos_batch, local_f_batch, m_batch,
                                                             train_set.x_pos_mean, train_set.x_pos_std,
                                                             train_set.x_neg_mean, train_set.x_neg_std,
                                                             train_set.sus_mean, train_set.sus_std,
                                                             train_set.field_mean, train_set.field_std)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            
            train_loss_list.append(loss.item())
            train_l1loss_list.append(l1loss.item())
            train_gdloss_list.append(gdloss.item())
            if args.W_MD_F_LOSS != 0:
                train_md_f_loss_list.append(md_f_loss.item())
            else:
                train_md_f_loss_list.append(0)
            if args.W_MD_Q_LOSS != 0:
                train_md_q_loss_list.append(md_q_loss.item())
            else:
                train_md_q_loss_list.append(0)
            if args.W_MD_R_LOSS != 0:
                train_md_r_loss_list.append(md_r_loss.item())
            else:
                train_md_r_loss_list.append(0)
            
            del(local_f_batch, r2star_batch, r2prime_batch, x_pos_batch, x_neg_batch, m_batch, input_batch, label_batch, loss, l1loss, gdloss, md_f_loss); torch.cuda.empty_cache();
            
        logger.info("Train: EPOCH %03d / %03d | LOSS %.4f | L1_LOSS %.4f | G_LOSS %.4f | M_F_LOSS %.4f | M_Q_LOSS %.4f | M_R_LOSS %.4f | TIME %.1fsec | LR %.8f"
              %(epoch+1, args.TRAIN_EPOCH, np.mean(train_loss_list), np.mean(train_l1loss_list), np.mean(train_gdloss_list), np.mean(train_md_f_loss_list), np.mean(train_md_q_loss_list), np.mean(train_md_r_loss_list), time.time() - epoch_time, optimizer.param_groups[0]['lr']))
        
        '''
        ########################
        ### Model validation ###
        ########################
        '''
        pre_model.eval()
        model.eval()
        
        with torch.no_grad():
            for direction in range(valid_set.matrix_size[-1]):
                local_f_batch = torch.tensor(valid_set.field[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                r2prime_batch = torch.tensor(valid_set.r2prime[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                r2star_batch = torch.tensor(valid_set.r2star[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                r2_batch = torch.tensor(valid_set.r2[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                x_pos_batch = torch.tensor(valid_set.x_pos[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                x_neg_batch = torch.tensor(valid_set.x_neg[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                m_batch = torch.tensor(valid_set.mask[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                pre_local_f_batch = torch.tensor(valid_set.field[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
                sus_batch = torch.tensor(valid_set.sus[np.newaxis, np.newaxis, ..., direction], device=device, dtype=torch.float)
    
                local_f_batch = ((local_f_batch.cpu() - valid_set.field_mean) / valid_set.field_std).to(device)
                r2prime_batch = ((r2prime_batch.cpu() - valid_set.r2prime_mean) / valid_set.r2prime_std).to(device)
                r2star_batch = ((r2star_batch.cpu() - valid_set.r2star_mean) / valid_set.r2star_std).to(device)
                r2_batch = ((r2_batch.cpu() - valid_set.r2_mean) / valid_set.r2_std).to(device)
                x_pos_batch = ((x_pos_batch.cpu() - valid_set.x_pos_mean) / valid_set.x_pos_std).to(device)
                x_neg_batch = ((x_neg_batch.cpu() - valid_set.x_neg_mean) / valid_set.x_neg_std).to(device)
                pre_local_f_batch = ((pre_local_f_batch.cpu() - valid_set.pre_field_mean) / valid_set.pre_field_std).to(device)
                sus_batch = ((sus_batch.cpu() - valid_set.sus_mean) / valid_set.sus_std).to(device)
    
                ### Brain masking ###
                local_f_batch = local_f_batch * m_batch
                r2prime_batch = r2prime_batch * m_batch
                r2star_batch = r2star_batch * m_batch
                r2_batch = r2_batch * m_batch
                pre_local_f_batch = pre_local_f_batch * m_batch
                sus_batch = sus_batch * m_batch
    
                ### Extract cosmos batch from pre-QSM Network ###
                pred_cosmos = pre_model(pre_local_f_batch)
                pred_cosmos_batch = (pred_cosmos * valid_set.pre_sus_std) + valid_set.pre_sus_mean
                pred_cosmos_batch = pred_cosmos_batch * m_batch
                
                pred_cosmos_batch = ((pred_cosmos_batch - valid_set.sus_mean) / valid_set.sus_std)
                pred_cosmos_batch = pred_cosmos_batch * m_batch
                
                if args.INPUT_MAP == 'r2p':
                    input_batch = torch.cat((pred_cosmos_batch, local_f_batch, r2prime_batch), 1)
                elif args.INPUT_MAP == 'r2s':
                    input_batch = torch.cat((pred_cosmos_batch, local_f_batch, r2star_batch), 1)
                label_batch = torch.cat((x_pos_batch, x_neg_batch), 1)
                    
                pred = model(input_batch)
                
                pred[:, 0, ...] = pred[:, 0, ...] * m_batch
                pred[:, 1, ...] = pred[:, 1, ...] * m_batch
                label_batch[:, 0, ...] = label_batch[:, 0, ...] * m_batch
                label_batch[:, 1, ...] = label_batch[:, 1, ...] * m_batch
    
                if args.INPUT_MAP == 'r2p':
                    loss, l1loss, gdloss, md_f_loss, md_q_loss, md_r_loss = total_loss(args, pred, label_batch, pred_cosmos_batch, r2prime_batch, r2prime_batch, local_f_batch, m_batch,
                                                                            valid_set.x_pos_mean, valid_set.x_pos_std,
                                                                            valid_set.x_neg_mean, valid_set.x_neg_std,
                                                                            valid_set.sus_mean, valid_set.sus_std,
                                                                            valid_set.field_mean, valid_set.field_std,
                                                                            valid_set.r2prime_mean, valid_set.r2prime_std,
                                                                            valid_set.r2_mean, valid_set.r2_std,
                                                                            valid_set.r2prime_mean, valid_set.r2prime_std)
                elif args.INPUT_MAP == 'r2s':
                    loss, l1loss, gdloss, md_f_loss, md_q_loss = total_loss_r2s(args, pred, label_batch, pred_cosmos_batch, local_f_batch, m_batch,
                                                                 valid_set.x_pos_mean, valid_set.x_pos_std,
                                                                 valid_set.x_neg_mean, valid_set.x_neg_std,
                                                                valid_set.sus_mean, valid_set.sus_std,
                                                                 valid_set.field_mean, valid_set.field_std)
                                                                 
                
                ### De-normalization ###
                pred_x_pos = ((pred[:, 0, ...].cpu() * valid_set.x_pos_std) + valid_set.x_pos_mean).to(device)
                pred_x_neg = ((pred[:, 1, ...].cpu() * valid_set.x_neg_std) + valid_set.x_neg_mean).to(device)
                label_x_pos = ((x_pos_batch.cpu() * valid_set.x_pos_std) + valid_set.x_pos_mean).to(device)
                label_x_neg = ((x_neg_batch.cpu() * valid_set.x_neg_std) + valid_set.x_neg_mean).to(device)
                
                pred = torch.cat((pred_x_pos[:, np.newaxis, ...], pred_x_neg[:, np.newaxis, ...]), 1)
                label = torch.cat((label_x_pos, label_x_neg), 1)
                m = torch.cat((m_batch, m_batch), 1)
                
                pos_nrmse = NRMSE(pred_x_pos, label_x_pos, m_batch)
                pos_psnr = PSNR(pred_x_pos, label_x_pos, m_batch)
                pos_ssim = SSIM(pred_x_pos, label_x_pos, m_batch)
                
                neg_nrmse = NRMSE(pred_x_neg, label_x_neg, m_batch)
                neg_psnr = PSNR(pred_x_neg, label_x_neg, m_batch)
                neg_ssim = SSIM(pred_x_neg, label_x_neg, m_batch)
                
                total_nrmse = pos_nrmse + neg_nrmse
                total_psnr = pos_psnr + neg_psnr
                total_ssim = pos_ssim + neg_ssim
                
                pred_sus = (pred_cosmos_batch * valid_set.sus_std) + valid_set.sus_mean
                label_sus = (sus_batch * valid_set.sus_std) + valid_set.sus_mean
                sus_nrmse = NRMSE(pred_sus, label_sus, m_batch)
                
                if direction < 6:
                    ori_valid_loss_list.append(loss.item())
                    ori_pos_nrmse_list.append(pos_nrmse)
                    ori_neg_nrmse_list.append(neg_nrmse)
                    ori_nrmse_list.append(total_nrmse)
                
                valid_loss_list.append(loss.item())
                pos_nrmse_list.append(pos_nrmse)
                neg_nrmse_list.append(neg_nrmse)
                pos_psnr_list.append(pos_psnr)
                neg_psnr_list.append(neg_psnr)
                pos_ssim_list.append(pos_ssim)
                neg_ssim_list.append(neg_ssim)
                
                nrmse_list.append(total_nrmse)
                psnr_list.append(total_psnr)
                ssim_list.append(total_ssim)
                
                sus_nrmse_list.append(sus_nrmse)
                
                del(local_f_batch, r2star_batch, r2prime_batch, x_pos_batch, x_neg_batch, m_batch, input_batch, label_batch, l1loss); torch.cuda.empty_cache();
            
            logger.info("Valid: EPOCH %03d / %03d | LOSS %.4f | oriLOSS %.4f | NRMSE(pos, neg) (%.4f, %.4f) | susNRMSE (%.4f) | oriNRMSE (%.4f, %.4f) | PSNR(pos, neg) (%.4f, %.4f) | SSIM(pos, neg) (%.4f, %.4f)\n"
                  %(epoch+1, args.TRAIN_EPOCH, np.mean(valid_loss_list), np.mean(ori_valid_loss_list), np.mean(pos_nrmse_list), np.mean(neg_nrmse_list), np.mean(sus_nrmse_list), np.mean(ori_pos_nrmse_list),  np.mean(ori_neg_nrmse_list), np.mean(pos_psnr_list), np.mean(neg_psnr_list), np.mean(pos_ssim_list), np.mean(neg_ssim_list)))
                
            train_loss.append(np.mean(train_loss_list))
            valid_loss.append(np.mean(valid_loss_list))
            ori_valid_loss.append(np.mean(ori_valid_loss_list))
            nrmse.append(np.mean(nrmse_list))
            psnr.append(np.mean(psnr_list))
            ssim.append(np.mean(ssim_list))
            
            writer.add_scalar("Train loss/epoch", np.mean(train_loss_list), epoch+1)      
            writer.add_scalar("valid loss/epoch", np.mean(valid_loss_list), epoch+1)
            writer.add_scalar("valid ori loss/epoch", np.mean(ori_valid_loss_list), epoch+1)
            writer.add_scalar("valid nrmse/epoch", np.mean(nrmse_list), epoch+1)
            writer.add_scalar("valid psnr/epoch", np.mean(psnr_list), epoch+1)
            writer.add_scalar("valid ssim/epoch", np.mean(ssim_list), epoch+1)
    
            if np.mean(valid_loss_list) < best_loss:
                save_model(epoch+1, model, args.CHECKPOINT_PATH, 'best_loss')
                best_loss = np.mean(valid_loss_list)
                best_epoch_loss = epoch+1
            if np.mean(ori_valid_loss_list) < ori_best_loss:
                save_model(epoch+1, model, args.CHECKPOINT_PATH, 'ori_best_loss')
                ori_best_loss = np.mean(ori_valid_loss_list)
                ori_best_epoch_loss = epoch+1
            if np.mean(nrmse_list) < best_nrmse:
                save_model(epoch+1, model, args.CHECKPOINT_PATH, 'best_nrmse')
                best_nrmse = np.mean(nrmse_list)
                best_epoch_nrmse = epoch+1
            if np.mean(psnr_list) > best_psnr:
                save_model(epoch+1, model, args.CHECKPOINT_PATH, 'best_psnr')
                best_psnr = np.mean(psnr_list)
                best_epoch_psnr = epoch+1
            if np.mean(ssim_list) > best_ssim:
                save_model(epoch+1, model, args.CHECKPOINT_PATH, 'best_ssim')
                best_ssim = np.mean(ssim_list)
                best_epoch_ssim = epoch+1
                
        ### Saving the model ###
        if (epoch+1) % args.SAVE_STEP == 0:
            save_model(epoch+1, model, args.CHECKPOINT_PATH, epoch+1)
            
    logger.info("------ Training is finished ------")
    logger.info(f'[best epochs]\nLoss: {best_epoch_loss}\nNRMSE: {best_epoch_nrmse}\nPSNR: {best_epoch_psnr}\nSSIM: {best_epoch_ssim}')
    logger.info(f'Total training time: {time.time() - start_time}')
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()