function model = load_model_edit2(idx)
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
    end
        
    model.walls = walls;
    model.reflectors =  reflectors;
    model.obstacles = {};
    [subbands,subband_order] = sort([-21, -7, 7, 21, -26:-22, -20:-8,-6:-1, 1:6, 8:20, 22:26]); % channels to considered
    freq = (5745 + (subbands).*(20/64))*10^6; % all the subcarrier frequencies
    model.lambda = 3e8./freq(subband_order);
    model.amps = ones(length(model.lambda),1)*100; % Lambda and amps can be arrays
end