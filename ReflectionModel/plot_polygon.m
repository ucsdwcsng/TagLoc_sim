function h=plot_polygon(coord)
for i=1:size(coord,1)
    start_point = coord(i,:);
    if(i<size(coord,1))
        end_point = coord(i+1,:);
    else
        end_point = coord(1,:);
    end
    h(i)=plot([start_point(1),end_point(1)],[start_point(2),end_point(2)]);
end
end