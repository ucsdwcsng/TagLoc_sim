function model = load_model_edit1(idx)
%% Define the space
    walls = get_rectangle([-2,-2],4,5);  % Walls are defined just for plotting. They are non-reflecting, non-blocking                                  
    if(idx==1)
        obstacles{1} = get_rectangle([4,2.5],1,2); %Obstacles are bounded and opaque to wireless signals
        obstacles{2} = get_rectangle([8,2.5],1,2); %get rectangle([x,y],w,h) returns a rectangle at x,y with width w and height h
        reflectors{1} = [2,1;2,2]; % Reflectors are linear for ease
        reflectors{2} = [2,-2;2,0]; % Reflectors are linear for ease
        reflectors{3} = [-0.5,2;1,3];
    elseif (idx==2)
        obstacles{1} = get_rectangle([1,2],0.5,1);
        obstacles{2} = get_rectangle([5,5],1,1);
        reflectors{1} = [-2,-2;0,-2];
        reflectors{2} = [0,0;-0.5,0];
        reflectors{3} = [1.5,3;2,3];
    elseif (idx==3)
        obstacles{1} = get_rectangle([6,1],2,3);
        %obstacles{2} = get_rectangle([10,11],2,2);
        reflectors{1} = [-2,3;-0.5,3];
        reflectors{2} = [-2,-2;-0.5,0];        
    elseif (idx==4)
        obstacles{1} = get_rectangle([4.5,0.5],0.5,0.5);
        obstacles{2} = get_rectangle([0,5],1.5,1);
        reflectors{1} = [-2,-2;-1.5,-1.5];
        reflectors{2} = [-2,2;-2,3];
        reflectors{3} = [1.5,-1;2,-1];
        %reflectors{3} = [15,15;20,15];
    elseif (idx==5)
        obstacles{1} = get_rectangle([9,2],0.5,2.5);
        obstacles{2} = get_rectangle([5,6],1.5,0.5);
        obstacles{3} = get_rectangle([0.5,3],1,1);
        reflectors{1} = [2,-2;2,-1];
        reflectors{2} = [-2,0.5;-2,3];
        reflectors{3} = [-2,-2;-1.5,-2];
        reflectors{4} = [1.5,0;1,1];
        %reflectors{3} = [15,15;20,15];
    elseif (idx==6)
        obstacles{1} = get_rectangle([9,1.5],0.5,2.5);
        obstacles{2} = get_rectangle([5,6],1.5,0.5);
        obstacles{3} = get_rectangle([5,2.5],1,1);
        reflectors{1} = [2,-2;2,-1];
        reflectors{2} = [-2,0;-2,0.5];
        reflectors{3} = [1,1;-0.5,0.5];
    elseif (idx==7)
        obstacles = {};
        reflectors = {};
%     elseif (idx==8)
%         obstacles{1} = get_rectangle([18,3],1,5);
%         obstacles{2} = get_rectangle([10,11],3,1);
%         obstacles{3} = get_rectangle([10,5],2,2);
%         reflectors = {};
%     elseif (idx==9)
%         obstacles = {};
%         reflectors{1} = [20,0;20,5];
%         reflectors{2} = [0,6;0,8];
%         reflectors{3} = [10,10;7,11];
    
    end
        
    model.walls = walls;
    model.reflectors =  reflectors;
    model.obstacles = {};
%     model.obstacles={};
    %model.reflectors={};
%     channels = [14:64,66:115]; % channels to considered
    channels = [1:512]; % channels to considered
%     freq1 = (2020 + (-64:63).*(50/128))*10^6; % all the subcarrier frequencies
%     freq2 = (2060 + (-64:63).*(50/128))*10^6; % all the subcarrier frequencies
    freq1 = (1940 + (-256:255).*(200/512))*10^6; % all the subcarrier frequencies
    freq2 = (2140 + (-256:255).*(200/512))*10^6; % all the subcarrier frequencies
    lambda1 = 3e8./freq1;
    lambda2 = 3e8./freq2;
    model.lambda = [lambda1(channels),lambda2(channels)];
%     model.lambda = 3e8./(2.4:0.001:2.48)./1e9;
    model.amps = ones(length(model.lambda),1)*100; % Lambda and amps can be arrays
end