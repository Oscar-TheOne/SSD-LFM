import os
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import tifffile

import scipy.io as sio

'''
# ----------------------------------------
# Step--1 (prepare opt)
# ----------------------------------------
'''
print('\nprepare opt\n')

parser = argparse.ArgumentParser()

parser.add_argument('--GPU', type=str, default='0', help='Define GPU')
parser.add_argument('--data_disk', type=str, default='0', help='')
parser.add_argument('--smpl_name', type=str, default='synthetic_tubulins', help='')
parser.add_argument('--min_snr', type=str, default='1', help='')
parser.add_argument('--max_snr', type=str, default='1', help='')
# parser.add_argument('--save_suffix', type=str, default='_rust6', help='')
parser.add_argument('--save_suffix', type=str, default='rust1', help='')
parser.add_argument('--net_type', type=str, default='bf_unet3D', help='network type')
parser.add_argument('--test_patch_size', type=int, default=128, help='')
parser.add_argument('--test_z_size', type=int, default=8, help='')
parser.add_argument('--check_data_flag', type=bool, default=False, help='')
parser.add_argument('--loss_type', type=str, default='', help='')

parser.add_argument('--preload_data_flag', action='store_true', help='')

parser.add_argument('--end_iter', type=int, default=100000, help='')

parser.add_argument('--min_z', type=int, default=0, help='')
parser.add_argument('--max_z', type=int, default=1000, help='')

parser.add_argument('--train_cell_count', type=int, default=-1, help='')

params = parser.parse_args()
# params.preload_data_flag = True

replace_params = {}

if not params.loss_type:
    loss_type = 'l2'
    loss_suffix = ''
else:
    loss_type = params.loss_type
    loss_suffix = '_' + loss_type

snr_suffix = '_' + params.min_snr + '_' + params.max_snr

# if params.train_cell_count>0:
# params.save_suffix=params.save_suffix+'_'+str(params.train_cell_count)

if params.train_cell_count > 0:
    if params.smpl_name:
        replace_params['task'] = ('s2s_' + params.smpl_name + snr_suffix + loss_suffix
                                  + '_' + params.save_suffix + '_' + str(
                    params.train_cell_count) + '_' + params.net_type)
else:
    if params.smpl_name:
        replace_params['task'] = ('s2s_' + params.smpl_name + snr_suffix + loss_suffix
                                  + '_' + params.save_suffix + '_' + params.net_type)

if params.data_disk == '0':
    replace_params['dataroot'] = (
                r'D:\SSD-LFM\data process\temp'
                + '\\' + params.smpl_name)
# F:\image_denoise\My_RCAN3D\My_RCAN3D\data_process\Zebrafish embryo membrane LR
# 'F:\image_denoise\My_RCAN3D\My_RCAN3D\data_process\My Synthetic beads HR'
# E:\image_denoise\My_RCAN3D\My_RCAN3D\data_process\My Synthetic tubulins HR

if params.GPU:
    replace_params['gpu_ids'] = [int(params.GPU)]

if params.net_type == 'uformer':
    opt_name = 'train_' + params.net_type + '_s2s' + '.json'
    opt = option.parse('options' + '\\' + opt_name, is_train=True, replace_params=replace_params)
else:
    opt_name = 'train_' + params.net_type + '_3D' + '_s2s' + '.json'
    opt = option.parse('options' + '\\' + opt_name, is_train=True, replace_params=replace_params)

opt['train']['G_lossfn_type'] = loss_type

if params.net_type == 'uformer' or params.net_type == 'sformer':
    params.test_patch_size = opt['netG']['patch_size']

opt['datasets']['train']['min_snr'] = params.min_snr
opt['datasets']['train']['max_snr'] = params.max_snr

opt['datasets']['test']['min_snr'] = params.min_snr
opt['datasets']['test']['max_snr'] = params.max_snr

opt['datasets']['train']['min_z'] = params.min_z
opt['datasets']['train']['max_z'] = params.max_z

opt['datasets']['train']['preload_data_flag'] = params.preload_data_flag
opt['datasets']['test']['preload_data_flag'] = False

opt['datasets']['train']['train_cell_count'] = params.train_cell_count
opt['datasets']['test']['test_cell_count'] = 1

opt['datasets']['train']['rust_suffix'] = params.save_suffix

torch.cuda.set_device(int(params.GPU))
util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

# ----------------------------------------
# update opt
# ----------------------------------------
# -->-->-->-->-->-->-->-->-->-->-->-->-->-
init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
opt['path']['pretrained_netG'] = init_path_G
current_step = init_iter

# --<--<--<--<--<--<--<--<--<--<--<--<--<-

# ----------------------------------------
# save opt to  a '../option.json' file
# ----------------------------------------
option.save(opt)

# ----------------------------------------
# return None for missing key
# ----------------------------------------
opt = option.dict_to_nonedict(opt)
if opt['sleep_time'] >= 1:
    print('sleep {:.2f} hours'.format(opt['sleep_time'] / 3600))
    time.sleep(opt['sleep_time'])

# ----------------------------------------
# configure logger
# ----------------------------------------
logger_name = 'train'
utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
logger = logging.getLogger(logger_name)
logger.info(option.dict2str(opt))
writer = SummaryWriter(os.path.join(opt['path']['log']))

# ----------------------------------------
# seed
# ----------------------------------------
seed = opt['train']['manual_seed']
if seed is None:
    seed = random.randint(1, 10000)
logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

'''
# ----------------------------------------
# Step--2 (creat dataloader)
# ----------------------------------------
'''
print('\ncreat dataloader\n')

# ----------------------------------------
# 1) create_dataset
# 2) creat_dataloader for train and test
# ----------------------------------------
dataset_type = opt['datasets']['train']['dataset_type']
for phase, dataset_opt in opt['datasets'].items():
    if phase == 'train':
        train_set = define_Dataset(dataset_opt)
        train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        train_loader = DataLoader(train_set,
                                  batch_size=dataset_opt['dataloader_batch_size'],
                                  shuffle=dataset_opt['dataloader_shuffle'],
                                  # num_workers=dataset_opt['dataloader_num_workers'],
                                  num_workers=0,
                                  drop_last=True,  # use or abandon the last minibatch
                                  pin_memory=True)  # using swamp memory
    elif phase == 'test':
        test_set = define_Dataset(dataset_opt)
        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=0,
                                 drop_last=False, pin_memory=True)
    else:
        raise NotImplementedError("Phase [%s] is not recognized." % phase)

# tmp1 = train_set.__getitem__(0)
# tmp2 = test_set.__getitem__(0)

'''G_regularizer_orthstep
# ----------------------------------------
# Step--3 (initialize model)
# ----------------------------------------
'''
print('\ninitialize model\n')

model = define_Model(opt)

logger.info(model.info_network())
model.init_train()
logger.info(model.info_params())

'''
# ----------------------------------------
# Step--4 (main training)
# ----------------------------------------
'''
print('\nmain training\n')

# tmp = train_set.__getitem__(0)

for epoch in range(opt['epoch_num']):  # keep running
    for i, train_data in enumerate(train_loader):

        current_step += 1

        # -------------------------------
        # 1) update learning rate
        # -------------------------------
        model.update_learning_rate(current_step)

        # -------------------------------
        # 2) feed patch pairs
        # -------------------------------
        model.feed_data(train_data)

        # -------------------------------
        # 3) optimize parameters
        # -------------------------------
        model.optimize_parameters(current_step)

        # -------------------------------
        # 4) training information
        # -------------------------------
        if current_step % opt['train']['checkpoint_print'] == 0:
            logs = model.current_log()  # such as loss
            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                      model.current_learning_rate())
            for k, v in logs.items():  # merge log information into message
                message += '{:s}: {:.3e} '.format(k, v)
                writer.add_scalar('{:s}'.format(k), v, current_step)
            logger.info(message)
            writer.add_scalar('train_loss', model.log_dict['G_loss'], current_step)
            writer.add_scalar('lr', model.current_learning_rate(), current_step)

        # if current_step == opt['train']['checkpoint_test'] or current_step % opt['train']['checkpoint_save'] == 0:
        #     training_visuals = model.current_results()
        #     TV_E = make_grid(training_visuals['E'], nrow=4, normalize=True, scale_each=True)
        #     TV_L = make_grid(training_visuals['L'], nrow=4, normalize=True, scale_each=True)
        #     TV_H = make_grid(training_visuals['H'], nrow=4, normalize=True, scale_each=True)
        #     TV_G = make_grid(training_visuals['G'], nrow=4, normalize=True, scale_each=True)
        #     writer.add_image('train - reconstr image', TV_E, epoch)
        #     writer.add_image('train - input image', TV_L, epoch)
        #     writer.add_image('train - target image', TV_H, epoch)
        #     writer.add_image('train - clean image', TV_G, epoch)

        # -------------------------------
        # 5) save model
        # -------------------------------
        if current_step % opt['train']['checkpoint_save'] == 0:
            logger.info('Saving the model.')
            model.save(current_step)

        # -------------------------------
        # 6) testing
        # -------------------------------
        if current_step % opt['train']['checkpoint_test'] == 0:
            # if True:
            idx = 0
            for test_data in test_loader:

                img_name = '%03d' % (idx)

                img_dir = os.path.join(opt['path']['images'], img_name)
                util.mkdir(img_dir)

                model.feed_data(test_data, test_patch_range=[0, params.test_patch_size, 0, params.test_patch_size],
                                test_z_range=[0, params.test_z_size], need_R=True)
                # model.feed_data(test_data,test_patch_size=params.test_patch_size,need_R=True)
                model.test(R_input=True)

                visuals = model.current_visuals(need_R=True)

                E_img = util.tensor2float(visuals['E'])
                R_img = util.tensor2float(visuals['R'])

                E_img_norm = util.tensor2float(visuals['E'], norm_flag=True)
                R_img_norm = util.tensor2float(visuals['R'], norm_flag=True)

                # -----------------------
                # save estimated image E
                # -----------------------
                save_img_path_tif = os.path.join(img_dir, 'E_{:s}_{:d}.tif'.format(img_name, current_step))
                tifffile.imsave(save_img_path_tif, np.uint16(E_img))
                save_img_path_tif = os.path.join(img_dir, 'E_{:s}_{:d}_norm.tif'.format(img_name, current_step))
                tifffile.imsave(save_img_path_tif, np.uint16(E_img_norm * 65535.))

                if current_step == opt['train']['checkpoint_test']:
                    save_img_path_tif = os.path.join(img_dir, 'R_{:s}_{:d}.tif'.format(img_name, current_step))
                    tifffile.imsave(save_img_path_tif, np.uint16(R_img))
                    save_img_path_tif = os.path.join(img_dir, 'R_{:s}_{:d}_norm.tif'.format(img_name, current_step))
                    tifffile.imsave(save_img_path_tif, np.uint16(R_img_norm * 65535.))

                idx += 1

        if current_step == params.end_iter:
            exit()

logger.info('Saving the final model.')
model.save('latest')
logger.info('End of training.')

