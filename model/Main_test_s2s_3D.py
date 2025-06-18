import os.path
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
import scipy.io as sio

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import tifffile

'''
# ----------------------------------------
# Step--1 (prepare opt)
# ----------------------------------------
'''
print('\nprepare opt\n')

# /home/bbnc/Documents/harddrive/PythonCode/SIM-dn2n/result/20211103/r2r_Microtubules_3_3_1_rev_bf_unet/models

parser = argparse.ArgumentParser()

parser.add_argument('--GPU', type=str, default='0', help='Define GPU')
parser.add_argument('--data_disk', type=str, default='0', help='')
parser.add_argument('--model_name', type=str, default='synthetic_tubulins_1_1_rust1', help='')
parser.add_argument('--model_folder', type=str, default='', help='')
parser.add_argument('--smpl_name', type=str, default='synthetic_tubulins', help='')
parser.add_argument('--net_type', type=str, default='bf_unet3D', help='network type') # bf_unet, rcan, etc
parser.add_argument('--test_patch_size1', type=int, default=512, help='')
parser.add_argument('--test_patch_size2', type=int, default=512, help='')
parser.add_argument('--test_z_min', type=int, default=1, help='')
parser.add_argument('--test_z_max', type=int, default=99, help='')
parser.add_argument('--model_patch_size', type=int, default=128, help='')
parser.add_argument('--model_z_size', type=int, default=8, help='')
parser.add_argument('--select_iter', type=int, default=-1, help='')

parser.add_argument('--min_snr', type=int, default=1, help='')
parser.add_argument('--max_snr', type=int, default=4, help='')

parser.add_argument('--test_cell_count', type=int, default=1, help='')

parser.add_argument('--overlap_ratio', type=float, default=0.2, help='')
parser.add_argument('--save_mat_flag', action='store_true', help='')

params = parser.parse_args()
replace_params = {}

replace_params['task'] = 's2s'

replace_params['task'] += '_' + params.model_name
replace_params['task'] += '_' + params.net_type

if params.data_disk=='0':
    replace_params['dataroot'] = (r'D:\SSD-LFM\data process\temp'
                                  + '\\' + params.smpl_name)
#'F:\image_denoise\My_RCAN3D\My_RCAN3D\data_process\My Synthetic beads HR'
#'F:\image_denoise\My_RCAN3D\My_RCAN3D\data_process\Zebrafish embryo membrane LR'
#'E:\image_denoise\My_RCAN3D\My_RCAN3D\data_process\My Synthetic tubulins HR'

if params.GPU:
    replace_params['gpu_ids'] = [int(params.GPU)]


opt_name = 'train_'+ params.net_type + '_3D' +'_s2s.json'

opt = option.parse('options'+'\\'+opt_name, is_train=True, replace_params=replace_params)

if params.model_patch_size < 0:
    params.model_patch_size = min(params.test_patch_size1,params.test_patch_size2)

opt['datasets']['test']['min_snr'] = params.min_snr
opt['datasets']['test']['max_snr'] = params.max_snr
opt['datasets']['test']['test_cell_count'] = params.test_cell_count

torch.cuda.set_device(int(params.GPU))

if params.model_folder:
    model_save_path = ('result' + '\\' + params.model_folder + '\\' + 's2s_' + params.model_name
                   + '_' + params.net_type + r'\models' )
else:
    model_save_path = ('result' + '\\' + 's2s_' + params.model_name
                   + '_' + params.net_type + r'\models' )

# ----------------------------------------
# update opt
# ----------------------------------------
# -->-->-->-->-->-->-->-->-->-->-->-->-->-
if params.select_iter < 0:
    init_iter, init_path_G = option.find_last_checkpoint(model_save_path, net_type='G')
    assert init_iter>0, 'Error: the model has not been trained!'
else:
    init_iter = params.select_iter
    init_path_G = model_save_path + '\\' + "%d"%(params.select_iter) + '_G' + '.pth'
    assert os.path.exists(init_path_G), 'Error: the selected model does not exist!'

opt['path']['pretrained_netG'] = init_path_G
current_step = init_iter


opt = option.dict_to_nonedict(opt)

seed = opt['train']['manual_seed']
if seed is None:
    seed = random.randint(1, 10000)
# logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# load test dataset
dataset_opt = opt['datasets']['test']
test_set = define_Dataset(dataset_opt)
test_loader = DataLoader(test_set, batch_size=1,
                         shuffle=False, num_workers=0,
                         drop_last=False, pin_memory=True)

# tmp = test_set.__getitem__(0)
# load model
model = define_Model(opt)
model.init_train()

save_dir = r'D:\SSD-LFM\model\test_results' + '\\' + params.smpl_name  + '\\' + 's2s_' +  params.model_name + '_' + params.net_type
util.mkdir(save_dir)

f = open( save_dir + '\\'+'qr'+'.txt','w')
idx = 0
for test_data in test_loader:
    idx += 1

    print('Processing ' + '%06d'%(idx) + ' out of ' + '%06d'%(len(test_loader))+' !')

    img_name = '%06d' % (idx)

    img_dir = os.path.join(save_dir, img_name)
    util.mkdir(img_dir)

    if params.test_z_min>=0 and params.test_z_max>=0:
        params.test_z_size = params.test_z_max - params.test_z_min
    else:
        params.test_z_size = test_data['R'].shape[2]

    E_img = np.zeros((params.test_z_size,params.test_patch_size1,params.test_patch_size2), dtype=np.float32)
    R_img = np.zeros((params.test_z_size,params.test_patch_size1,params.test_patch_size2), dtype=np.float32)

    zz_list = list(range(0,params.test_z_size-params.model_z_size+1,int(params.model_z_size/2)))
    if zz_list[-1] != params.test_z_size-params.model_z_size:
        zz_list.append(params.test_z_size-params.model_z_size)

    rr_list = list(range(0,params.test_patch_size1-params.model_patch_size+1,int(params.model_patch_size/2)))
    if rr_list[-1] != params.test_patch_size1-params.model_patch_size:
        rr_list.append(params.test_patch_size1-params.model_patch_size)

    cc_list = list(range(0,params.test_patch_size2-params.model_patch_size+1,int(params.model_patch_size/2)))
    if cc_list[-1] != params.test_patch_size2-params.model_patch_size:
        cc_list.append(params.test_patch_size2-params.model_patch_size)
    print('zz_list:', zz_list)
    print('rr_list:', rr_list)
    print('cc_list:', cc_list)
    for zz in zz_list:
        print('%03d'%(zz)+' out of '+'%03d'%(zz_list[-1]))
        for rr in rr_list:
            for cc in cc_list:
                if zz == 0:
                    zz_min = 0
                    zz_min_patch = 0
                else:
                    zz_min = zz + int(params.model_z_size*params.overlap_ratio)
                    zz_min_patch = int(params.model_z_size*params.overlap_ratio)
                if zz + params.model_z_size == params.test_z_size:
                    zz_max = params.test_z_size
                    zz_max_patch = params.model_z_size
                else:
                    zz_max = zz + params.model_z_size - int(params.model_z_size*params.overlap_ratio)
                    zz_max_patch = params.model_z_size -int(params.model_z_size*params.overlap_ratio)

                if rr == 0:
                    rr_min = 0
                    rr_min_patch = 0
                else:
                    rr_min = rr + int(params.model_patch_size*params.overlap_ratio)
                    rr_min_patch = int(params.model_patch_size*params.overlap_ratio)

                if rr + params.model_patch_size == params.test_patch_size1:
                    rr_max = params.test_patch_size1
                    rr_max_patch = params.model_patch_size
                else:
                    rr_max = rr + params.model_patch_size - int(params.model_patch_size*params.overlap_ratio)
                    rr_max_patch = params.model_patch_size - int(params.model_patch_size*params.overlap_ratio)

                if cc == 0:
                    cc_min = 0
                    cc_min_patch = 0
                else:
                    cc_min = cc + int(params.model_patch_size*params.overlap_ratio)
                    cc_min_patch = int(params.model_patch_size*params.overlap_ratio)

                if cc + params.model_patch_size == params.test_patch_size2:
                    cc_max = params.test_patch_size2
                    cc_max_patch = params.model_patch_size
                else:
                    cc_max = cc + params.model_patch_size - int(params.model_patch_size*params.overlap_ratio)
                    cc_max_patch = params.model_patch_size - int(params.model_patch_size*params.overlap_ratio)

                f.write('zz:'+str(zz)+' zz_min:'+str(zz_min)+' zz_max:'+str(zz_max)+' zz_min_patch:'+str(zz_min_patch)+' zz_max_patch:'+str(zz_max_patch)+'\n')
                f.write('rr:'+str(rr)+' rr_min:'+str(rr_min)+' rr_max:'+str(rr_max)+' rr_min_patch:'+str(rr_min_patch)+' rr_max_patch:'+str(rr_max_patch)+'\n')
                f.write('cc:'+ str(cc)+ ' cc_min:'+str(cc_min)+ ' cc_max:'+str(cc_max)+' cc_min_patch:'+str(cc_min_patch)+' cc_max_patch:'+str(cc_max_patch)+'\n')
                f.write('\n')

                if params.test_z_min>0:
                    test_z_range = [params.test_z_min+zz,params.test_z_min+zz+params.model_z_size]
                else:
                    test_z_range = [zz,zz+params.model_z_size]
                test_patch_range = [rr,rr+params.model_patch_size,cc,cc+params.model_patch_size]


                model.feed_data(test_data,test_patch_size=-1,need_R=True,test_patch_range=test_patch_range,test_z_range=test_z_range)
                model.test(R_input=True)

                visuals = model.current_visuals(need_R=True)

                E_img_p = util.tensor2float(visuals['E'])
                R_img_p = util.tensor2float(visuals['R'])

                E_img[zz_min:zz_max,rr_min:rr_max,cc_min:cc_max] = E_img_p[0,zz_min_patch:zz_max_patch,rr_min_patch:rr_max_patch,cc_min_patch:cc_max_patch]
                R_img[zz_min:zz_max,rr_min:rr_max,cc_min:cc_max] = R_img_p[0,zz_min_patch:zz_max_patch,rr_min_patch:rr_max_patch,cc_min_patch:cc_max_patch]


    if params.save_mat_flag:
        sio.savemat(os.path.join(img_dir,'img_data.mat'), {'E':E_img,'R':R_img})

    E_img_norm = util.my_norm(E_img)
    R_img_norm = util.my_norm(R_img)

    # quantatitive analysis
    # psnr1 = 0.0
    # ssim1 = 0.0
    # for z_id in range(params.test_z_size):
    #     psnr1 += util.calculate_psnr_peak(np.squeeze(R_img_norm[z_id,...]), np.squeeze(E_img_norm[z_id,...]), border=30)
    #     ssim1 += util.calculate_ssim_peak(np.squeeze(R_img_norm[z_id,...]), np.squeeze(E_img_norm[z_id,...]), border=30)
    #
    # psnr1 = psnr1 / params.test_z_size
    # ssim1 = ssim1 / params.test_z_size
    #
    # f.write(img_name + ':' +'\n')
    # f.write('psnr1 = '+ '%.5f'%(psnr1)+'\n')
    # f.write('ssim1 = '+ '%.5f'%(ssim1)+'\n')
    # f.write('\n')
    # f.write('\n')


    # save imgs

    E_img[E_img<0] = 0
    R_img[R_img<0] = 0

    E_img_mip = np.max(E_img,axis=0)
    R_img_mip = np.max(R_img,axis=0)


    save_img_path_tif = os.path.join(img_dir, 'E.tif')
    tifffile.imsave(save_img_path_tif, np.uint16(E_img))
    save_img_path_tif = os.path.join(img_dir, 'E_norm.tif')
    tifffile.imsave(save_img_path_tif, np.uint16(E_img_norm*65535.))
    save_img_path_tif = os.path.join(img_dir, 'E_mip.tif')
    tifffile.imsave(save_img_path_tif, np.uint16(E_img_mip))


    save_img_path_tif = os.path.join(img_dir, 'R.tif')
    tifffile.imsave(save_img_path_tif,np.uint16(R_img))
    save_img_path_tif = os.path.join(img_dir, 'R_norm.tif')
    tifffile.imsave(save_img_path_tif,np.uint16(R_img_norm*65535.))
    save_img_path_tif = os.path.join(img_dir, 'R_mip.tif')
    tifffile.imsave(save_img_path_tif, np.uint16(R_img_mip))

f.close()