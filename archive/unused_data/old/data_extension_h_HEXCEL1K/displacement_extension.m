function  extensions = displacement_extension(coords, disps)
    
%     coords = loc_disp(:,1:2);
%     disps = loc_disp(:,3);
    
    % corrdinates of the target points
    coords_target_top = [-10, 10; 0, 10; 10, 10];
    coords_target_bot = [-10, -10; 0, -10; 10, -10];
    
    extensions = zeros(1,3);

    % average within 2mm × 2mm reference areas (target points are the centers)
    for i = 1:3
        % calculate the difference between all coordinates and the target
        coords_dif_top = abs(coords - coords_target_top(i,:));
        coords_dif_bot = abs(coords - coords_target_bot(i,:));
        
        % get the indices in the reference area
        ind_top = all(coords_dif_top <= 1, 2);
        ind_bot = all(coords_dif_bot <= 1, 2);
        
        % get the corresponding displacements
        disps_top = disps(ind_top);
        disps_bot = disps(ind_bot);
        
        % calculate the averaged extensions
        extensions(i) = mean(disps_top) - mean(disps_bot);
    end
    
end