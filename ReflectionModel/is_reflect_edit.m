function [p,t]=is_reflect_edit(src,dst, reflector)

if(sum(abs(src-dst))==0)

    fprintf('SRC and DST are the same in is_reflect. The code does bad things for this case. ');

end

% Divide the reflector into 1000 points and check if it reflects

n_pts = 20000;

src_angle = zeros(n_pts-1,1);

dst_angle = zeros(n_pts-1,1);

for i=1:n_pts-1

    cur_point = reflector(1,:)*i/n_pts + reflector(2,:)*(n_pts-i)/n_pts;

    line_slope_1 = local_cart2pol(reflector(1,:)-cur_point);

    line_slope_2 = local_cart2pol(reflector(2,:)-cur_point);

    

    src_slope = local_cart2pol(src-cur_point);

    dst_slope = local_cart2pol(dst -cur_point);

    % find acute angles for src and dst

    if (abs(src_slope - line_slope_1)<abs(src_slope-line_slope_2))

        src_angle(i) = src_slope - line_slope_1;

    else

        src_angle(i) = src_slope - line_slope_2;

    end

    

    if (abs(dst_slope - line_slope_1)<abs(dst_slope-line_slope_2))

        dst_angle(i) = dst_slope - line_slope_1;

    else

        dst_angle(i) = dst_slope - line_slope_2;

    end       

end

[~,idx] = min(abs(src_angle+dst_angle));

t=false; p =[];



if(abs(src_angle(idx)+dst_angle(idx))<0.1)

    

    p =  reflector(1,:)*idx/n_pts + reflector(2,:)*(n_pts-idx)/n_pts;

    t=true;

end



[~,idx] = min(abs(src_angle-dst_angle));

if((abs(src_angle(idx)-pi/2) < 0.1 || abs(src_angle(idx)+pi/2)<0.1)&&(abs(src_angle(idx) - dst_angle(idx)) <0.1))

    p =  reflector(1,:)*idx/n_pts + reflector(2,:)*(n_pts-idx)/n_pts;

    t=true;

end



[coeff,~,~]=polyfit(reflector(:,1),reflector(:,2),1);

m=coeff(1);

b=coeff(2);

if(sign(dst(2)-dst(1)*m-b)~=sign(src(2)-src(1)*m-b))

    %Different sides of reflector

    t=false;

end





end



function t = local_cart2pol(a)

    t=cart2pol(a(1),a(2));

end
