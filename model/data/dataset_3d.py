import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import torch
import glob
import os
import tifffile

class DatasetPlain(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetPlain, self).__init__()
        print('Get L/H for image-to-image mapping.')
        self.opt = opt
        self.n_channels_in = opt['n_channels_in'] if opt['n_channels_in'] else 1
        self.n_channels_out = opt['n_channels_out'] if opt['n_channels_out'] else 1
        assert self.n_channels_in == self.n_channels_out
        self.n_channels = self.n_channels_in
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64
        self.z_size = self.opt['z_size'] if self.opt['z_size'] else 8
        
        self.phase = self.opt['phase']
        self.preload_data_flag = self.opt['preload_data_flag']
        self.dataroot = self.opt['dataroot']
        
        self.train_cell_count = self.opt['train_cell_count'] if self.opt['train_cell_count'] else -1
        self.test_cell_count = self.opt['test_cell_count'] if self.opt['test_cell_count'] else -1
        
        self.view_count = 2
        self.edge_margin = 10
        # ------------------------------------
        # get the path
        # ------------------------------------
        
        if self.opt['phase'] == 'train':
            self.paths_V = [[] for _ in range(self.view_count)] 
        
        if self.opt['phase'] == 'test':
            self.paths_R = []  
        
        self.min_snr = int(opt['min_snr'])
        self.max_snr = int(opt['max_snr'])
        
        if self.opt['phase'] == 'train':
            self.rust_suffix = opt['rust_suffix']
            snr_dirs = sorted(glob.glob(os.path.join(self.dataroot,self.rust_suffix,'*')))
            snr_dirs = snr_dirs[self.min_snr-1:self.max_snr]
        
        if self.opt['phase'] == 'test':

            print(f"Looking for raw data in: {os.path.join(self.dataroot, 'raw')}")

            snr_dirs = sorted(glob.glob(os.path.join(self.dataroot,'raw','*')))

            print(f"Found {len(snr_dirs)} SNR directories: {snr_dirs}")

            snr_dirs = snr_dirs[self.min_snr-1:self.max_snr]
            
        if self.opt['phase'] == 'train':
            self.min_z = opt['min_z']
            self.max_z = opt['max_z']
        
        if self.opt['phase'] == 'train':
            for snr_dir in snr_dirs:
                for view_id in range(self.view_count):
                    sub_dir = os.path.join(snr_dir,'r'+str(view_id+1))
                    tmp_paths = sorted(glob.glob(sub_dir+r'\*.tif'))
                    if self.train_cell_count>0:
                        self.paths_V[view_id].extend(tmp_paths[:self.train_cell_count])
                    else:
                        self.paths_V[view_id].extend(tmp_paths[:])
        if self.opt['phase'] == 'test':
            print(f"Selected SNR directories (from {self.min_snr} to {self.max_snr}): {snr_dirs}")
            for snr_dir in snr_dirs:
                # print("snr_dir:"+ snr_dir)
                tmp_paths = sorted(glob.glob(snr_dir+r'\*.tif'))
                if self.test_cell_count>0:
                    tmp_paths = tmp_paths[:]
                self.paths_R.extend(tmp_paths[:self.test_cell_count])    

                
        # ------------------------------------
        # check the path
        # ------------------------------------
        if self.opt['phase'] == 'train':
            for view_id in range(self.view_count):            
                assert self.paths_V[view_id], 'Error: V path is needed but it is empty.'
                assert len(self.paths_V[view_id])==len(self.paths_V[0]), 'Error: the file number of each view must be the same.'
                
        if self.opt['phase'] == 'test':
            assert self.paths_R, 'Error: R path is needed but it is empty.'
        
        # self.preload_data_flag = True       
        if self.phase =='train' and self.preload_data_flag:
            self.imgs_V = [[] for _ in range(self.view_count)] 
            
            for view_id in range(self.view_count):
                print('Loading data of view %d, please wait ...'%(view_id+1))

                for path_V in self.paths_V[view_id]:
                    # img_LF = util.read_img_3d(path_V)
                    img_LF = np.expand_dims(tifffile.imread(path_V),3)[self.min_z:self.max_z,...]
                    self.imgs_V[view_id].append(img_LF)

    def __getitem__(self, index):

        if self.opt['phase'] == 'train':
            
            view_mode = np.random.randint(0, 2)
            
            file_id = index // self.view_count            
            
            # ------------------------------------
            # get H L image
            # ------------------------------------
            
            if view_mode == 0:
                H_path = self.paths_V[0][file_id]
                L_path = self.paths_V[1][file_id]
                
                if self.preload_data_flag:
                    img_H = self.imgs_V[0][file_id]
                    img_L = self.imgs_V[1][file_id]
                else:                    
                    img_H = np.expand_dims(tifffile.imread(H_path),3)[self.min_z:self.max_z,...]
                    img_L = np.expand_dims(tifffile.imread(L_path),3)[self.min_z:self.max_z,...]
                                    
            elif view_mode == 1:
                H_path = self.paths_V[1][file_id]
                L_path = self.paths_V[0][file_id]
                if self.preload_data_flag:
                    img_H = self.imgs_V[1][file_id]
                    img_L = self.imgs_V[0][file_id]
                else:                    
                    img_H = np.expand_dims(tifffile.imread(H_path),3)[self.min_z:self.max_z,...]
                    img_L = np.expand_dims(tifffile.imread(L_path),3)[self.min_z:self.max_z,...]
                    
            else:
                raise NotImplementedError  

            Z, H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            
            # rnd_z = random.randint(max(0,self.min_z), max(0, min(Z,self.max_z) - self.z_size))
            rnd_z = random.randint(0, max(0, Z - self.z_size))
            rnd_h = random.randint(0+self.edge_margin, max(0, H - self.patch_size-self.edge_margin))
            rnd_w = random.randint(0+self.edge_margin, max(0, W - self.patch_size-self.edge_margin))
            
            patch_L = img_L[rnd_z:rnd_z + self.z_size, rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_z:rnd_z + self.z_size, rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            patch_L, patch_H = util.augment_img_3d(patch_L, mode=mode), util.augment_img_3d(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.single2tensor3_3d(patch_L), util.single2tensor3_3d(patch_H)
            
            img_R = torch.zeros(img_L.shape,dtype=img_L.dtype)
            img_G = torch.zeros(img_L.shape,dtype=img_L.dtype)
            R_path = L_path
            G_path = L_path
            
        if self.opt['phase'] == 'test':
            # ------------------------------------
            # get R image
            # ------------------------------------
            R_path = self.paths_R[index]
            
            img_R = np.expand_dims(tifffile.imread(R_path),3)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_R = util.single2tensor3_3d(img_R)
            
            img_L = torch.zeros(img_R.shape,dtype=img_R.dtype)
            img_H = torch.zeros(img_R.shape,dtype=img_R.dtype)
            img_G = torch.zeros(img_R.shape,dtype=img_R.dtype)
            L_path = R_path
            H_path = R_path
            G_path = R_path

        return {'L': img_L, 'H': img_H, 'R': img_R, 'G': img_G, 'L_path': L_path, 'H_path': H_path, 'R_path': R_path,'G_path':G_path}

    def __len__(self):
        if self.opt['phase'] == 'train':
            return 2*len(self.paths_V[0])
        else:
            return len(self.paths_R)
