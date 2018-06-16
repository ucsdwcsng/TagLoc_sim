clear
idx=101;
figure,hold on
for idx=[1:50:151,200]
    load(sprintf('/home/roshan/work/tagloc/DeepakSimulation/New_Res/Trainreal_testreal_small_new1/test_real_27net1_%d.mat',idx));
    [n_pts,m,n] = size(predict);
    for i=1:n_pts
        temp = labels(i,:,:);
        [~,l_max]=max(temp(:));
        temp = predict(i,:,:);
        [~,p_max]=max(temp(:));
        [li(i),lj(i)]=ind2sub([m,n],l_max);
        [pi(i),pj(i)]=ind2sub([m,n],p_max);
        err(i)=norm([li(i),lj(i)]-[pi(i),pj(i)])*0.1;
    end
    cdfplot(err)
    disp(median(err));
end


load('/home/roshan/work/tagloc/DeepakSimulation/ReflectionModel/datasets/dataset2_real20Apr_test1.mat');
% ids = [1,2,3,5,6,7,9,10,11,13,14,15];
ids = [4,8,12,16];
predictions = squeeze(sum(features(:,:,:,:),2));
[n_points,m,n] = size(predictions);
for i=1:n_points
    temp = predictions(i,:,:);
    [~,p_max]=max(temp(:));
    temp = labels_gaussian_2d(i,:,:);
    [~,l_max]=max(temp(:));
    [apj(i),api(i)]=ind2sub([m,n],p_max);
    [alj(i),ali(i)]=ind2sub([m,n],l_max);
    err1(i)=norm([api(i),apj(i)]-[ali(i),alj(i)])*0.1;
end
cdfplot(err1);hold off;
disp(median(err1));
% legend('After 1 epochs', 'After 5 Epochs','After 9 Epochs','After 13 Epochs','After 17 Epochs','After 20 Epochs','Before CNN');
legend('After 1 Epoch','After 51 Epochs','After 101 Epochs','After 151 Epochs','After 200 Epochs','Before CNN');
hold off;
title('error CDF for 5x4, trained-1-6 and tested-real for absolute channels, CNN first model. For new data')