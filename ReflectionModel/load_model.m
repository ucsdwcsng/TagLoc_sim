function model = load_model(idx)
%% Define the space
    walls = get_rectangle([0,0],20,15);  % Walls are defined just for plotting. They are non-reflecting, non-blocking                                  
    if(idx==1)
        obstacles{1} = get_rectangle([7,5],2,3); %Obstacles are bounded and opaque to wireless signals
        obstacles{2} = get_rectangle([16,5],2,3); %get rectangle([x,y],w,h) returns a rectangle at x,y with width w and height h
        reflectors{1} = [20,10;20,13]; % Reflectors are linear for ease
        reflectors{2} = [20,1;20,5]; % Reflectors are linear for ease
        reflectors{3} = [2,10;8,13];
    elseif (idx==2)
        obstacles{1} = get_rectangle([2,3],1,2);
        obstacles{2} = get_rectangle([10,11],2,2);
        reflectors{1} = [0,0;7,0];
        reflectors{2} = [0,0;7,7];
        reflectors{3} = [15,15;20,15];
    elseif (idx==3)
        obstacles{1} = get_rectangle([12,3],4,6);
        %obstacles{2} = get_rectangle([10,11],2,2);
        reflectors{1} = [0,15;7,15];
        reflectors{2} = [0,0;7,7];        
    elseif (idx==4)
        obstacles{1} = get_rectangle([9,1],1,1);
        obstacles{2} = get_rectangle([0,11],3,2);
        reflectors{1} = [0,0;3,6];
        reflectors{2} = [0,10;0,15];
        reflectors{3} = [17,3;19,4];
        %reflectors{3} = [15,15;20,15];
    elseif (idx==5)
        obstacles{1} = get_rectangle([18,3],1,5);
        obstacles{2} = get_rectangle([10,11],3,1);
        obstacles{3} = get_rectangle([1,5],2,2);
        reflectors{1} = [20,0;20,5];
        reflectors{2} = [0,9;0,15];
        reflectors{3} = [0,0;3,0];
        reflectors{4} = [16,7;13,10];
        %reflectors{3} = [15,15;20,15];
    elseif (idx==6)
        obstacles{1} = get_rectangle([18,3],1,5);
        obstacles{2} = get_rectangle([10,11],3,1);
        obstacles{3} = get_rectangle([10,5],2,2);
        reflectors{1} = [20,0;20,5];
        reflectors{2} = [0,6;0,8];
        reflectors{3} = [10,10;7,11];
        
    
    end
        
    model.walls = walls;
    model.reflectors =  reflectors;
    model.obstacles = obstacles;
    model.obstacles={};
%     model.reflectors={};
    model.lambda = 3e8./(2.4:0.001:2.48)./1e9;
    model.amps = ones(length(model.lambda),1)*100; % Lambda and amps can be arrays
end