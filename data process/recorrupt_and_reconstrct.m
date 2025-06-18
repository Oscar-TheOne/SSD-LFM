clear;

load("./temp/HR_noise_1.mat");

img_baseline = 100;

photon_count_list =[7,10];

photon_count_num = length(photon_count_list);

% image Preparameters
img_size=184;
img_num=1;
z_num=5;

% deconv Preparameters
GPUcompute=1;        %% GPU accelerator (on/off)
Nnum=13;             %% the number of sensor pixels after each microlens/ the number of angles in one dimension
DAO = 0;             %% digital adaptive optics (on/off)
Nb=1;                %% number of blocks for multi-site AO in one dimension
Nshift=3;  % the sampling points of a single scanning period
maxIter=10; %******% the maximum iteration number of single frame


% PSF
load('PSF/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-20u_zmax20u_zspacing0.4u','psf');

weight=squeeze(sum(sum(sum(psf,1),2),5))./sum(psf(:));
weight=weight-min(weight(:));
weight=weight./max(weight(:)).*0.8;
for u=1:Nnum
    for v=1:Nnum
        if (u-round(Nnum/2))^2+(v-round(Nnum/2))^2>(round(Nnum/3))^2
            weight(u,v)=0;
        end
    end
end

smpl_name = 'synthetic_tubulins';


rust_suffix = 'rust1';
alpha_v_min = 0.8;
alpha_v_max = 0.8;
beta_1_min = 1.5;
beta_1_max = 1.5;
beta_2_min = 24;
beta_2_max = 24;


% for j = 7
save_root_dir = ['./temp/',smpl_name];
save_root_dir_raw = [save_root_dir,'\','raw'];
save_root_dir_rust = [save_root_dir,'\',rust_suffix];


%% key parameters

arg=[alpha_v_min,alpha_v_max,beta_1_min,beta_1_max,beta_2_min,beta_2_max];

gaussian_filter = fspecial('gaussian',[3,3],1.2);

for photon_count_id = 1:1:photon_count_num

    photon_count = photon_count_list(photon_count_id);
    save_cur_dir_raw = [save_root_dir_raw,'\','photon_',num2str(photon_count,'%03d')];
    save_cur_dir_r1 = [save_root_dir_rust,'\','photon_',num2str(photon_count,'%03d'),'\','r1'];
    save_cur_dir_r2 = [save_root_dir_rust,'\','photon_',num2str(photon_count,'%03d'),'\','r2'];
    if ~exist(save_cur_dir_raw,'dir')
        mkdir(save_cur_dir_raw);
    end
    if ~exist(save_cur_dir_r1,'dir')
        mkdir(save_cur_dir_r1);
    end
    if ~exist(save_cur_dir_r2,'dir')
        mkdir(save_cur_dir_r2);
    end


    for i = 1: img_num       
        
        %% process raw
        save_name_raw = [num2str(i,'%03d'),'.tif'];
        img_temp=zeros(img_size,img_size,Nnum,Nnum);
        img_temp = img_noise(1:img_size,1:img_size,:,:,photon_count_id,i);
        
        % deconv rawaaaaaaa
        WDF = img_temp;
        WDF=imresize(WDF,[size(WDF,1)*Nnum/Nshift,size(WDF,2)*Nnum/Nshift]);
        Xguess=ones(size(WDF,1),size(WDF,2),size(psf,5));
        Xguess=Xguess./sum(Xguess(:)).*sum(WDF(:))./(size(WDF,3)*size(WDF,4));
        tic;
        Xguess = deconvRL(maxIter, Xguess,WDF, psf, weight, DAO, Nb, GPUcompute);
        ttime = toc;
        imwriteTFSK(single(gather(Xguess(:,:,:))),[save_cur_dir_raw,'\',save_name_raw]); 
        
        %% process r2r
        img_raw = img_temp+img_baseline;

        h = size(img_raw,1);
        w = size(img_raw,2);
        u = size(img_raw,3);
        v = size(img_raw,4);

        img_r1 = zeros(h,w,u,v);
        img_r2 = zeros(h,w,u,v);

        alpha_v = (alpha_v_max-alpha_v_min)*rand()+alpha_v_min;
        D = 1/alpha_v*ones(h,w);
        D_1 = alpha_v*ones(h,w);

        beta_1 = (beta_1_max-beta_1_min)*rand()+beta_1_min;
        beta_2 = (beta_2_max-beta_2_min)*rand()+beta_2_min;

        for zz=1:1:z_num
            save_name_z = [num2str(i,'%03d'),'_',num2str(zz,'%02d'),'.tif'];
            for uu = 1: 1: u
                for vv = 1: 1: v
                    img_tmp = max(img_raw(:,:,uu,vv)-img_baseline,0);
                    img_tmp = imfilter(img_tmp,gaussian_filter,'replicate','same'); 
                    z_n = normrnd(0,1,[h,w]);

                    Sigma_x = max(beta_1*img_tmp+beta_2,0);

                    img_r1(:,:,uu,vv) = img_raw(:,:,uu,vv) + D .* sqrt(Sigma_x) .* z_n;
                    img_r2(:,:,uu,vv) = img_raw(:,:,uu,vv) - D_1 .* sqrt(Sigma_x) .* z_n;
                end
            end

            % deconv r1
            WDF = img_r1-img_baseline;
            WDF=imresize(WDF,[size(WDF,1)*Nnum/Nshift,size(WDF,2)*Nnum/Nshift]);
            Xguess=ones(size(WDF,1),size(WDF,2),size(psf,5));
            Xguess=Xguess./sum(Xguess(:)).*sum(WDF(:))./(size(WDF,3)*size(WDF,4));
            tic;
            Xguess = deconvRL(maxIter, Xguess,WDF, psf, weight, DAO, Nb, GPUcompute);
            ttime = toc;
            imwriteTFSK(single(gather(Xguess(:,:,:))),[save_cur_dir_r1,'\',save_name_z]);  

            % deconv r2
            WDF = img_r2-img_baseline;
            WDF=imresize(WDF,[size(WDF,1)*Nnum/Nshift,size(WDF,2)*Nnum/Nshift]);
            Xguess=ones(size(WDF,1),size(WDF,2),size(psf,5));
            Xguess=Xguess./sum(Xguess(:)).*sum(WDF(:))./(size(WDF,3)*size(WDF,4));
            tic;
            Xguess = deconvRL(maxIter, Xguess,WDF, psf, weight, DAO, Nb, GPUcompute);
            ttime = toc;
            imwriteTFSK(single(gather(Xguess(:,:,:))),[save_cur_dir_r2,'\',save_name_z]);  
        end
    end
end





