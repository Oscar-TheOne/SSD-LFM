

# train
# python Main_train_s2s_3D.py --smpl_name synthetic_tubulins --min_snr 1 --max_snr 1 --min_z 10 --max_z 80 --GPU 0 --preload_data_flag --save_suffix rust1
# python Main_train_s2s_3D.py --smpl_name synthetic_tubulins --min_snr 2 --max_snr 2 --min_z 10 --max_z 80 --GPU 0 --preload_data_flag --save_suffix rust1


# test
# python Main_test_s2s_3D.py --smpl_name synthetic_tubulins --model_name synthetic_tubulins_1_1_rust1 --min_snr 1 --max_snr 1 --GPU 0