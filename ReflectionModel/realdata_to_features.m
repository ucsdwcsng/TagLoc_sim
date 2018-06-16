clear
close all;
%% Load Data
load('channels_Feb27_calibrated_ext.mat')
load('ap_cli_pos_real_27Feb.mat')
channels = [14:64,66:115]; % channels to considered
freq1 = (2020 + (-64:63).*(50/128))*10^6; % all the subcarrier frequencies
freq2 = (2060 + (-64:63).*(50/128))*10^6; % all the subcarrier frequencies
lambda1 = 3e8./freq1;
lambda2 = 3e8./freq2;
n_h = length(h1);
channel = zeros(440,202,4,4);
count=0;
indexess = [];
temp_pos=cli_pos;
for i=1:n_h
    if(~isempty(h1{i}))
        count = count+1;
        indexess = [indexess,i];
        for a=1:202
            for j=1:4
                for k=1:4
                    ind=(j-1)*4+k;
                    if (a<102)
                        channel(count,a,j,k) = h1{i}(ind,channels(a));
                    else
                        channel(count,a,j,k) = h2{i}(ind,channels(a-101));
                    end
                end
            end
        end
        cli_pos(i,:) = temp_pos(indexes(i),:);
    end
end
%% AP2 and Ap3 flipping
ap = ap_pos_corrected;
% ap{2} = flip(ap{2});
% ap{3} = flip(ap{3});%% Indexing
% center_index = find(cli_pos(indexes,1)<0.2 & cli_pos(indexes,1)>=-1 & cli_pos(indexes,2)<1.5 & cli_pos(indexes,2)>=0);
%% Calculating AoA Errors
% n_lambda=2*length(channels);
% n_points = length(indexes);
% n_ap=length(ap);
% % n_ant=length(ap{1});
% theta_vals = -pi/2:0.01:pi/2; %Theta values to consider for multipath profiles
% d_vals = -8:0.1:8;
% opt.freq = [freq1(channels),freq2(channels)];
% opt.ant_sep = abs(ap{1}(2,1)-ap{1}(1,1));
% lambda = 3e8/median(opt.freq);
% calc_AoA = zeros(n_points,n_ap);
% ap_orient = [-1,1,1,-1];
% for i=1:n_points
%     for j=1:n_ap
%         Prel =  compute_multipath_profile2d_fast_edit(squeeze(channel(i,:,j,:)),theta_vals,d_vals,opt);
%         [~,inde] = max(abs(Prel(:)));
%         [angle,tof] = ind2sub(size(Prel),inde);
%         calc_AoA(i,j) = theta_vals(angle);
% 
%     end
% end
% 
% real_AoA=zeros(n_points,n_ap);
% for i=1:n_points
%     for j=1:n_ap
%         ap_pos = mean(ap{j});
%         ap_vec=ap{j}(1,:)-ap{j}(end,:);
%         X=cli_pos(indexes(i),1)-ap_pos(1);
%         Y=cli_pos(indexes(i),2)-ap_pos(2);
%         real_AoA(i,j)=sign(sum([X,Y].*ap_vec))*(pi/2-acos(abs(sum([X,Y].*ap_vec))/norm([X,Y])/norm(ap_vec)));
%     end
% end
% calc_AoA = calc_AoA.*180./pi;
% real_AoA = real_AoA.*180./pi;
% AoA_errors = real_AoA-calc_AoA;
% opt.AoA_comp = -mean(AoA_errors,1).*pi./180;
%% Compensating for the AoA errors
% for i=1:4
%     for j=1:4
%         channel(:,:,i,j) = channel(:,:,i,j).*exp(1j*2*pi*opt.ant_sep*(j-1)*sin(opt.AoA_comp(i))/median(lambda));
%     end
% end
%% Feature Calculation
n_lambda=2*length(channels);
n_points = length(indexess);
n_ap=length(ap);
n_ant=length(ap{1});
theta_vals = -pi/2:0.01:pi/2; %Theta values to consider for multipath profiles
d_vals = -8:0.1:8;
d_vals1 = 0:0.1:5;

feature_idx=1;
opt.freq = [freq1(channels),freq2(channels)];
opt.ant_sep = abs(ap{1}(2,1)-ap{1}(1,1));
d1=-2:0.1:2;
d2=-2:0.1:3;
% features = zeros(n_points,n_ap*(n_ap-1),length(d2),length(d1));
features = zeros(n_points,n_ap,length(d2),length(d1));
i=0;
parfor i=1:n_points
    % Get features
%     features(i,:,:,:) = generate_features_real(squeeze(channel(i,:,:,:)),ap,theta_vals,d_vals,d1,d2,opt,n_lambda);
    features(i,:,:,:) = generate_features_abs(squeeze(channel(i,:,:,:)),ap,theta_vals,d_vals1,d1,d2,opt,n_lambda);
    if(mod(i,1000)==0)
        disp(i);
    end    
end
label_std = 1/sqrt(2*pi);%std(labels_gaussian_all(:));
feature_mean =zeros(1,size(features,2));
feature_std = zeros(1,size(features,2));
for k=1:length(feature_mean)
    dset = features(:,k,:,:);
    feature_mean(k) = mean(dset(:));
    feature_std(k) = std(dset(:));
end
features = (features-repmat(feature_mean,size(features,1),1,size(features,3),size(features,4)))./repmat(feature_std,size(features,1),1,size(features,3),size(features,4));

labels_gaussian_2d=get_gaussian_labels_negative(cli_pos(indexess,:),0.1,0.25,d1,d2);
labels_discrete = round(cli_pos(indexess,:)*10)/10;

save('datasets/dataset2_real27Feb_ext.mat','features','labels_gaussian_2d','labels_discrete','-v7.3');