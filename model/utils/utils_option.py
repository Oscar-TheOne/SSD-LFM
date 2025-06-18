import os
from collections import OrderedDict
from datetime import datetime
import json
import re
import glob


'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def get_timestamp():
    return datetime.now().strftime('_%y%m%d_%H%M%S')


def parse(opt_path, is_train=True, replace_params =None):

    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    
    for (key, value) in replace_params.items():
        if key == 'net_type':
            opt['netG']['net_type'] = value
        else:
            opt[key] = value

    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    # ----------------------------------------
    # set default
    # ----------------------------------------
    if 'merge_bn' not in opt:
        opt['merge_bn'] = False
        opt['merge_bn_startpoint'] = -1

    if 'scale' not in opt:
        opt['scale'] = 1

    # ----------------------------------------
    # datasets
    # ----------------------------------------
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['n_channels_in'] = opt['n_channels_in']  # broadcast
        dataset['n_channels_out'] = opt['n_channels_out']  # broadcast    
        dataset['dataroot'] = opt['dataroot']
        

    # ----------------------------------------
    # path
    # ----------------------------------------
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)

    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = os.path.join(path_task, 'logs')
    opt['path']['options'] = os.path.join(path_task, 'options')

    if is_train:
        opt['path']['models'] = os.path.join(path_task, 'models')
        opt['path']['images'] = os.path.join(path_task, 'images')
    else:  # test
        opt['path']['images'] = os.path.join(path_task, 'test_images')

    # ----------------------------------------
    # network
    # ----------------------------------------
    opt['netG']['scale'] = opt['scale'] if 'scale' in opt else 1

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


def find_last_checkpoint(save_dir, net_type='G'):
    """
    Args: 
        save_dir: model folder
        net_type: 'G' or 'D'

    Return:
        init_iter: iteration number
        init_path: model path
    """
    file_list = glob.glob(os.path.join(save_dir, '*_{}.pth'.format(net_type)))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, '{}_{}.pth'.format(init_iter, net_type))
    else:
        init_iter = 0
        init_path = None
    return init_iter, init_path


'''
# --------------------------------------------
# convert the opt into json file
# --------------------------------------------
'''


def save(opt):
    opt_path = opt['opt_path']
    opt_path_copy = opt['path']['options']
    dirname, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    dump_path = os.path.join(opt_path_copy, filename+get_timestamp()+ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


'''
# --------------------------------------------
# dict to string for logger
# --------------------------------------------
'''


def dict2str(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


'''
# --------------------------------------------
# convert OrderedDict to NoneDict,
# return None for missing key
# --------------------------------------------
'''


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None
    
def remove_last_word(s):
    return '\\'.join(s.split('/')[:-1])
    

def edit_options_r2r_3d_slice_npy(opt, params):
    opt['datasets']['train']['min_snr'] = params.min_snr 
    opt['datasets']['train']['max_snr'] = params.max_snr 
    opt['datasets']['train']['min_z'] = params.min_z 
    opt['datasets']['train']['max_z'] = params.max_z 
    opt['datasets']['train']['smpl_suffix'] = params.smpl_suffix
    opt['datasets']['train']['dataroot_L'] = remove_last_word(opt['datasets']['train']['dataroot_L'])
    opt['datasets']['train']['dataroot_H'] = remove_last_word(opt['datasets']['train']['dataroot_H'])
    opt['datasets']['train']['pararoot_L'] = remove_last_word(opt['datasets']['train']['pararoot_L'])
    opt['datasets']['train']['check_data_flag'] = params.check_data_flag
    opt['datasets']['train']['mask_flag'] = params.mask_flag 

    
    opt['datasets']['test']['min_snr'] = params.min_snr
    opt['datasets']['test']['max_snr'] = params.max_snr 
    opt['datasets']['test']['min_z'] = params.min_z
    opt['datasets']['test']['max_z'] = params.max_z 
    opt['datasets']['test']['smpl_suffix'] = params.smpl_suffix
    opt['datasets']['test']['dataroot_L'] = remove_last_word(opt['datasets']['test']['dataroot_L'])
    opt['datasets']['test']['dataroot_H'] = remove_last_word(opt['datasets']['test']['dataroot_H'])
    opt['datasets']['test']['dataroot_R'] = remove_last_word(opt['datasets']['test']['dataroot_R'])
    opt['datasets']['test']['pararoot_L'] = remove_last_word(opt['datasets']['test']['pararoot_L'])
    opt['datasets']['test']['check_data_flag'] = False
    opt['datasets']['test']['mask_flag'] = False
    
    return opt


def edit_options_r2r_lattice_npy(opt, params):
    opt['datasets']['train']['min_snr'] = params.min_snr 
    opt['datasets']['train']['max_snr'] = params.max_snr 
    opt['datasets']['train']['min_z'] = params.min_z 
    opt['datasets']['train']['max_z'] = params.max_z 
    opt['datasets']['train']['smpl_suffix'] = params.smpl_suffix
    opt['datasets']['train']['dataroot_L'] = remove_last_word(opt['datasets']['train']['dataroot_L'])
    opt['datasets']['train']['dataroot_H'] = remove_last_word(opt['datasets']['train']['dataroot_H'])
    opt['datasets']['train']['pararoot_L'] = remove_last_word(opt['datasets']['train']['pararoot_L'])
    opt['datasets']['train']['check_data_flag'] = params.check_data_flag 
    opt['datasets']['train']['mask_flag'] = params.mask_flag 

    
    opt['datasets']['test']['min_snr'] = params.min_snr
    opt['datasets']['test']['max_snr'] = params.max_snr 
    opt['datasets']['test']['min_z'] = params.min_z
    opt['datasets']['test']['max_z'] = params.max_z 
    opt['datasets']['test']['smpl_suffix'] = params.smpl_suffix
    opt['datasets']['test']['dataroot_L'] = remove_last_word(opt['datasets']['test']['dataroot_L'])
    opt['datasets']['test']['dataroot_H'] = remove_last_word(opt['datasets']['test']['dataroot_H'])
    opt['datasets']['test']['dataroot_R'] = remove_last_word(opt['datasets']['test']['dataroot_R'])
    opt['datasets']['test']['pararoot_L'] = remove_last_word(opt['datasets']['test']['pararoot_L'])
    opt['datasets']['test']['check_data_flag'] = False
    opt['datasets']['test']['mask_flag'] = False    
    
    return opt

def edit_options_r2r_lattice_tle_npy(opt, params):
    opt['datasets']['train']['min_z'] = params.min_z 
    opt['datasets']['train']['max_z'] = params.max_z 
    opt['datasets']['train']['smpl_suffix'] = params.smpl_suffix
    opt['datasets']['train']['dataroot_L'] = remove_last_word(opt['datasets']['train']['dataroot_L'])
    opt['datasets']['train']['dataroot_H'] = remove_last_word(opt['datasets']['train']['dataroot_H'])
    opt['datasets']['train']['pararoot_L'] = remove_last_word(opt['datasets']['train']['pararoot_L'])
    opt['datasets']['train']['check_data_flag'] = params.check_data_flag
    opt['datasets']['train']['mask_flag'] = params.mask_flag
    

    opt['datasets']['test']['min_z'] = params.min_z
    opt['datasets']['test']['max_z'] = params.max_z 
    opt['datasets']['test']['smpl_suffix'] = params.smpl_suffix
    opt['datasets']['test']['dataroot_L'] = remove_last_word(opt['datasets']['test']['dataroot_L'])
    opt['datasets']['test']['dataroot_H'] = remove_last_word(opt['datasets']['test']['dataroot_H'])
    opt['datasets']['test']['dataroot_R'] = remove_last_word(opt['datasets']['test']['dataroot_R'])
    opt['datasets']['test']['pararoot_L'] = remove_last_word(opt['datasets']['test']['pararoot_L'])
    opt['datasets']['test']['check_data_flag'] = False
    opt['datasets']['test']['min_t'] = params.min_t
    opt['datasets']['test']['max_t'] = params.max_t     
    opt['datasets']['test']['mask_flag'] = params.mask_flag
    
    return opt


def edit_options_r2r_simu_npy(opt, params):
    opt['datasets']['train']['dataroot_L'] = remove_last_word(opt['datasets']['train']['dataroot_L'])
    opt['datasets']['train']['dataroot_H'] = remove_last_word(opt['datasets']['train']['dataroot_H'])
    opt['datasets']['train']['pararoot_L'] = remove_last_word(opt['datasets']['train']['pararoot_L'])
    opt['datasets']['train']['check_data_flag'] = params.check_data_flag
    
    opt['datasets']['test']['dataroot_L'] = remove_last_word(opt['datasets']['test']['dataroot_L'])
    opt['datasets']['test']['dataroot_H'] = remove_last_word(opt['datasets']['test']['dataroot_H'])
    opt['datasets']['test']['dataroot_R'] = remove_last_word(opt['datasets']['test']['dataroot_R'])
    opt['datasets']['test']['pararoot_L'] = remove_last_word(opt['datasets']['test']['pararoot_L'])
    opt['datasets']['test']['check_data_flag'] = False
    
    return opt


def edit_options_r2r_exp_npy(opt, params):
    
    opt['datasets']['test']['dataroot_L'] = remove_last_word(opt['datasets']['test']['dataroot_L'])
    opt['datasets']['test']['dataroot_H'] = remove_last_word(opt['datasets']['test']['dataroot_H'])
    opt['datasets']['test']['dataroot_R'] = remove_last_word(opt['datasets']['test']['dataroot_R'])
    opt['datasets']['test']['pararoot_L'] = remove_last_word(opt['datasets']['test']['pararoot_L'])
    return opt


def edit_options_r2r_3d_tle_npy(opt, params):  
    # tmp = remove_last_word(opt['datasets']['train']['dataroot_L']) + '/' + 'Exp'+ params.smpl_suffix
    tmp = remove_last_word(opt['datasets']['train']['dataroot_L']) 
    opt['datasets']['train']['dataroot'] = tmp
    opt['datasets']['train']['pararoot'] = tmp
    opt['datasets']['train']['min_cell'] = params.min_cell
    opt['datasets']['train']['max_cell'] = params.max_cell
    opt['datasets']['train']['min_z'] = params.min_z
    opt['datasets']['train']['max_z'] = params.max_z
    opt['datasets']['train']['smpl_suffix'] = params.smpl_suffix
    opt['datasets']['train']['mask_flag'] = params.mask_flag
    
    # tmp = remove_last_word(opt['datasets']['test']['dataroot_L']) + '/' + 'Exp'+ params.smpl_suffix
    tmp = remove_last_word(opt['datasets']['test']['dataroot_L']) + '\\'
    opt['datasets']['test']['dataroot'] = tmp
    opt['datasets']['test']['pararoot'] = tmp
    opt['datasets']['test']['min_cell'] = params.min_cell
    opt['datasets']['test']['max_cell'] = params.max_cell
    opt['datasets']['test']['test_cell_count'] = params.test_cell_count
    opt['datasets']['test']['min_z'] = params.min_z
    opt['datasets']['test']['max_z'] = params.max_z
    opt['datasets']['test']['min_t'] = params.min_t
    opt['datasets']['test']['max_t'] = params.max_t
    opt['datasets']['test']['mask_flag'] = False

    return opt


