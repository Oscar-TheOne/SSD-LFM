clear;
close all;

load("./temp/HR_raw_1.mat");

result_path = "./temp/HR_noise_1.mat";

Nnum = 13;

photon_count_list =[7,10]; 
photon_count_num = length(photon_count_list);

img_size = 800;
noise_intensity = 5;
img_num = 1;
Nshift=3;
lambda=1;

img_raw(img_raw<0) = 0;
img_noise=zeros(ceil(img_size*Nshift/Nnum),ceil(img_size*Nshift/Nnum),Nnum,Nnum,photon_count_num,img_num);

for i =1:1:img_num
    for photon_count_id = 1: 1: photon_count_num
        for u=1:Nnum
            for v=1:Nnum
                photon_count = photon_count_list(photon_count_id);
                img_noise(:,:,u,v,photon_count_id,i) = lambda*poissrnd(img_raw(:,:,u,v,i)/20*photon_count/lambda) + normrnd(0,noise_intensity,...
                    [ceil(img_size*Nshift/Nnum),ceil(img_size*Nshift/Nnum)]);
            end
        end
    end
end
save(result_path,'img_noise');
