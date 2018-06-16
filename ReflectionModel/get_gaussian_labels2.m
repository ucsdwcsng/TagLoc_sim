function labels_gaussian = get_gaussian_labels2(labels,grid_size,sigma,min_x,max_x,min_y,max_y)
n_xlabels = length(min_x:grid_size:max_x);
n_ylabels = length(min_y:grid_size:max_y);
% labels_discrete = round(labels./grid_size);
% n_xlabels = length(unique(labels_discrete(:,1)));
% n_ylabels =length(unique(labels_discrete(:,2)));
labels_gaussian = zeros(size(labels,1),n_ylabels,n_xlabels);
map_X = repmat(min_x:grid_size:max_x,n_ylabels,1);
map_Y = repmat((min_y:grid_size:max_y)',1,n_xlabels);
n_points = size(labels,1);
    for i=1:n_points
        d = (map_X-labels(i,1)).^2+(map_Y-labels(i,2)).^2;
        cur_gaussian = exp(-d/sigma/sigma);%*1/sqrt(2*pi)/output_sigma;        
        labels_gaussian(i,:)=cur_gaussian(:);
    end
end



