clear;
close all;

addpath('./Solver/');
addpath('./utils/');
addpath('./npy/');
addpath('./CONVNFFT_Folder/CONVNFFT_Folder/');

load('PSF/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-20u_zmax20u_zspacing0.4u','psf');

img_num = 1;
Nnum = 13;
img_size = 800;
z_size = size(psf,5);
Nshift=3;

num_control_points = 2; 
twist_factor = 1e-100; 
space_size = [img_size, img_size, z_size]; 


num_points = 1000; 
t = linspace(0, 1, num_points);

outputFolder= './temp/';
result_path = './temp/HR_raw_1.mat';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

img_raw=zeros(ceil(img_size*Nshift/Nnum),ceil(img_size*Nshift/Nnum),Nnum,Nnum,img_num);

for i=1:img_num
    seeds = cell(1,30);
    sample = zeros(img_size,img_size,z_size);
    for xx = 1:30
      
        seeds{xx} = rand(num_control_points, 3);
    end

    for yy = 1:30
       
        seed = seeds{yy};  
        
        img_name = generate_twisted_bezier_curve(num_control_points, twist_factor, space_size, seed);

        
        x = round(img_name(:, 1));
        y = round(img_name(:, 2));
        z = round(img_name(:, 3));

        
        for k = 1:num_points
            if(x(k)>=1 && x(k)<=img_size && y(k)>=1 && y(k)<=img_size && z(k)>=1 &&z(k)<=z_size)
                sample(x(k), y(k), z(k)) = 100;
            end
        end
    end

    sigma_g = 0.2;
    sample = imgaussfilt(sample, sigma_g);
    

    tic;
    for u=1:Nnum
        for v=1:Nnum
            
            fprintf(['Processing i=',num2str(i,'%02d'),', u=',num2str(u,'%02d'),', v=',num2str(v,'%02d'),'!\n']);
            psf_uv = double(squeeze(psf(:,:,u,v,:)));
            
            projection=zeros(size(sample,1),size(sample,2));

            for zz =1:size(sample,3)
                projection=projection+conv2(sample(:,:,zz),psf_uv(:,:,zz),'same');
            end

            projection_uv=imresize(projection,[size(projection,1)*Nshift/Nnum,size(projection,2)*Nshift/Nnum]);
            
            img_raw(:,:,u,v,i) =projection_uv;

        end
    end
    toc;
end

save(result_path,'img_raw');