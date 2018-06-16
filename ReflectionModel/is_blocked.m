function t = is_blocked(ray,blockage_ray)
t=false;
if(ray(1,1)==ray(2,1) || blockage_ray(1,1)==blockage_ray(2,1))
   % Deal with infinite slope opjects (x=c)
    if(ray(1,1)==ray(2,1) && blockage_ray(1,1)==blockage_ray(2,1))
        t=false;
        return;
    end
    if(ray(1,1)==ray(2,1))
        x_ray = ray;
        other_ray = blockage_ray;
    end
    if(blockage_ray(1,1)==blockage_ray(2,1))
        x_ray = blockage_ray;
        other_ray = ray;
    end
    
    if(other_ray(2,2)==other_ray(1,2))
        % if the y's are equal
        common = [other_ray(1,2),x_ray(1,1)];
    else
        [m1,b1]=get_slope(x_ray(1,[2,1]),x_ray(2,[2,1]));
        [m2,b2]=get_slope(other_ray(1,[2,1]),other_ray(2,[2,1]));
        common(1) = -(b2-b1)/(m2-m1);
        common(2) = m1*common(1)+b1;
    end
    t = is_point_on_line(common, x_ray(:,[2,1])) & is_point_on_line(common,other_ray(:,[2,1]));
        

else
    
    % Normal operation
    [m_ray, b_ray] = get_slope(ray(1,:), ray(2,:));
    [m_block, b_block] = get_slope(blockage_ray(1,:),blockage_ray(2,:));
    if(abs(m_block - m_ray)<0.0001)
        t=false;    % Parallel lines don't intersect
        return;
    else
        % Compute intersection
        common(1) = -(b_block-b_ray)/(m_block-m_ray);
        common(2) = m_ray*common(1)+b_ray;
        
        % Check that the intersection lies on ray
        t = is_point_on_line(common, ray) & is_point_on_line(common,blockage_ray);
    end
end


    
end

function [m,b]=get_slope(p1,p2)
    m = (p2(2)-p1(2))/(p2(1)-p1(1));
    b = -m * p1(1)+p1(2);    
end

function t = is_point_on_line(p,ray)
    check1 = ((p(1)-ray(2,1))/(ray(1,1)-ray(2,1)));
    check2 = ((p(2)-ray(2,2))/(ray(1,2)-ray(2,2)));
    t = (check1> 0 && check1 <1) || (check2>0 && check2<1);
end