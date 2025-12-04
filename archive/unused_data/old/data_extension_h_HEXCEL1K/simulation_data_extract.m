function simulation_data_extract(direction)
%  extract extension data from simulation results
%   direction:  data from which direction to be used
%               'h' - horizontal extension 
%               'v' - vertical extension
% 

    load("lhs_simulation_data.mat", 'coordinates_undeformed', 'coordinates', ...
         'samples_theta', 'samples_angle', 'samples_load');
    
    extensions = zeros(length(samples_load),3);
    
    loads_angles = [samples_load, samples_angle];
    
    % indices of the surface points
    ind = (abs(coordinates_undeformed(:,end) - 1.163) < eps) ... 
         & (coordinates_undeformed(:,3) <= 12.1) & (coordinates_undeformed(:,3) >= -12.1);
    
    % undeformed coordinates
    coords_0 = coordinates_undeformed(ind, 2:4);
    for iter = 1:length(samples_load)
        
        % deformed coordinates
        coords_1 = coordinates{iter}(ind,2:4);

        % rigid motion correction
        coords_1_corr = correct_aba_rigid_rot(coords_0,coords_1); 
        
        % displacements with rigid motion correction
        disps_corr = (coords_1_corr - coords_0)/2;
        
        % calculate the extension
        if strcmp(direction,'h') % horizontal
            extensions(iter,:) = displacement_extension(coords_0(:,1:2), disps_corr(:,1));
        elseif strcmp(direction,'v') % vertical
            extensions(iter,:) = displacement_extension(coords_0(:,1:2), disps_corr(:,2));
        end
    end

    % write data to files
    writematrix(extensions, 'data_extension_sim.txt');
    writematrix(loads_angles, 'input_load_angle_sim.txt');
    writematrix(samples_theta, 'input_theta_sim.txt');

end

% figure; hold on;
% plot(extension(:,1), Loads/1e3, 'b-o');
% plot(extension(:,2), Loads/1e3, 'r-o');
% plot(extension(:,3), Loads/1e3, 'g-o');
% xlabel('Extension');
% ylabel('Load(kN)');
% legend('left', 'center', 'right')

% figure;
% ind = (coordinates{1}(:,end) <= 1.17) & (1.14 <= coordinates{1}(:,end)) & (coordinates{1}(:,3) <= 12.1) & (coordinates{1}(:,3) >= -12.1);
% scatter(coordinates{1}(ind,2), coordinates{1}(ind,3), 40, displacements{1}(ind,2), 'filled'); shading flat; colorbar;

% ind = (coordinates{1}(:,end) <= 1.17) & (1.11 <= coordinates{1}(:,end)) & (coordinates{1}(:,3) <= 12.1) & (coordinates{1}(:,3) >= -12.1);
% scatter(coordinates{1}(ind,2), coordinates{1}(ind,3), 40, log_strains{1}(ind,2), 'filled'); shading flat; colorbar;
% colormap jet;
% ylim([-12.05 12.05]);
% xlim([-19.05 19.5]);
% box on;
% title('Model prediciton');
