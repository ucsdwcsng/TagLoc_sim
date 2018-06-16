function features = generate_features_abs_aoa(channels,ap,theta_vals,d1,d2,opt)

n_ap=length(ap);
% n_ant=length(ap{1});

% channels_rel = zeros(n_lambda,n_ap,n_ant,n_ap-1);
features = zeros(n_ap,length(d2),length(d1));
% feature_idx=1;

for j=1:n_ap
    ant_pos = abs(ap{j}(1,opt.ap_orient(j))-ap{j}(:,opt.ap_orient(j)));
    P = compute_multipath_profile(squeeze(channels(:,j,:)),ant_pos,opt.lambda,theta_vals);
    P_out = convert_multipath_to_2d(P,theta_vals,d1,d2,ap{j}(:,:));
    features(j,:,:) = db(abs(P_out));
end