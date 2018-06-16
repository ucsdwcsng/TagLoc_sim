clear
%% Description: 
% The goal of this code is to get channels given a physical space. 
% The definition of physical space can include finite reflectors and 
% bounded obstacles.
% Reflectors are obstacles by default
% Color code: Red is for walls, blue is for obstacles and green is for
% reflectors
% For paths, red path is a non-blocked path and black path is a blocked
% path
%% Define the space
max_x = 20;
max_y = 15;
n_ap = 4;
d1 = 0:0.5:max_x;
d2=0:0.5:max_y;
n_total_per_ap = 100000;
n_ant_per_ap = 3;
labels = zeros(n_total_per_ap,2);

for idx=4:6
    model=load_model(idx);
    %model=load_model(1);
    %% Define the setup
    max_x = max(model.walls(:,1));
    max_y = max(model.walls(:,2));
    ant_sep = min(model.lambda)/2;
    ap{1}=[10,0;
        10+ant_sep,0;
        10+2*ant_sep,0];

    ap{2}=[10,15;
        10+ant_sep,15;
        10+2*ant_sep,15];

    ap{3}=[0,7.5-ant_sep;
            0, 7.5;
            0,7.5+ant_sep;];

    ap{4}=[20,7.5-ant_sep;
            20, 7.5;
            20,7.5+ant_sep;];
    ap_orient=[1,1,2,2];
    
    theta_vals = [-pi/2:0.01:pi/2];
    
    %% Generate the points. Ap 1 antenna 1 serves as the reference
    %features=zeros(n_total,length(model.lambda)*length(ap)*n_ant_per_ap*2);
    
    channels = zeros(n_total_per_ap,length(model.lambda),length(ap),n_ant_per_ap);
    rng;

    start_time = now;
    
    for i=1:n_total_per_ap
        point_found=false;
        while(~point_found) % Don't allow points inside obstacles
            pos = [rand()*max(model.walls(:,1)),rand()*max(model.walls(:,2))];
            point_found=true;
            for obst_idx = 1:length(model.obstacles)
                if(inpolygon(pos(1),pos(2),model.obstacles{obst_idx}(:,1),model.obstacles{obst_idx}(:,2)))
                    point_found=false;
                end
            end
        end
        for j=1:length(ap)
            for k=1:n_ant_per_ap
                channels(i,:,j,k)=get_channels_from_model(model,pos,ap{j}(k,:),false);
            end
            channels(i,:,j,:) = awgn(squeeze(channels(i,:,j,:)),30);                               
        end
        labels(i,:) = pos;
        if(mod(i,100)==0)
            disp([i,(now -start_time)*24*60]);
        end
    end
    cur_model = load_model(idx);
    save(sprintf('channels_%d.mat',idx),'cur_model','channels','labels','ap','-v7.3');
end
%delete(gcp('nocreate'));
