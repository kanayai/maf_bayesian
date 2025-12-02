function experiment_data_extract(angle_values, direction)
% extract extension data from experiment data files
%     angle_values: the correpsonding angles of the data to be used,
%                   three angles are available: 45, 90, 135
%                   e.g.,[45, 135] - use data corresponding to 45 and 135
% 
%     direction:  data from which direction to be used
%                 'h' - horizontal extension
%                 'v' - vertical extension

delete data_extension_exp_*
delete input_load_angle_exp_*

% loading angles
angles = [45, 90, 135];

% files for each loading angle
files_45deg = {'C:\Users\Sinan\Documents\CerTest\data_thick_S6_A45\quasi-static-fail-thickS6.csv', ...
    'C:\Users\Sinan\Documents\CerTest\data_thick_S13_A45\quasi-static-fail-thickS13.csv', ...
    'C:\Users\Sinan\Documents\CerTest\data_thick_S14_A45\quasi-static-fail-thickS14.csv'};

files_90deg = {'C:\Users\Sinan\Documents\CerTest\data_thick_S3_A90\quasi-static-fail.csv', ...
    'C:\Users\Sinan\Documents\CerTest\data_thick_S8_A90\quasi-static-fail.csv', ...
    'C:\Users\Sinan\Documents\CerTest\data_thick_S9_A90\quasi-static-fail.csv'};

files_135deg = {'C:\Users\Sinan\Documents\CerTest\data_thick_S23_A135\quasi-static-fail-IM7thickS23.csv',...
    'C:\Users\Sinan\Documents\CerTest\data_thick_S24_A135\quasi-static-fail-IM7thickS24.csv'};

files_all = {files_45deg, files_90deg, files_135deg};

% all data of extensions and loads extracted
extensions_all = cell(1, length(angles));
loads_all = cell(1, length(angles));

% which angles are used, 1: 45deg,  2: 90deg,  3: 135deg
% angles_index = [1,3]; 
angles_index = find(ismember(angles,angle_values));

ind_data = 1;
for ind_angles = angles_index  
    
    % files corresponding to each angle
    files = files_all{ind_angles};

    extenions_each_angle = cell(1, length(files));
    loads_each_angle = cell(1,length(files));
%     for ind_files = 2
    for ind_files = 1:length(files) % loop for each specimen
        
        % get the specific file
        file = files{ind_files}; 
        df = readtable(file, 'NumHeaderLines', 0); 
        
        % data files 
        file_names_o = table2array(df(:,"File")); 
        loads_o = table2array(df(:,"AnalogInAi0_kN_"));
        
        % upper bound for loads
        for iter = 1:length(loads_o)
            if loads_o(iter) > 10
                ind_upper = iter-1;
                break;
            end
        end
        
        n_sep = 3; % take one data in every 'n_sep' steps
        temp = file_names_o(15:n_sep:ind_upper); % start from index 15
        file_names = [file_names_o(1); temp];
        temp = loads_o(15:n_sep:ind_upper);
        loads = [loads_o(1); temp]; 
        [loads, ia] = unique(loads, 'stable'); % remove repeated elements

        file_names = file_names(ia); % get the elements corresponding to loads
%         loads(1) = 0; % set first load equal 0
        loads = loads - loads(1); % first load is 0
        extensions = zeros(length(loads),3);

        for iter = 1:length(loads)
            % name of data file corresponding to each load
            [~,name] = fileparts(file_names{iter});
            filepath = fileparts(file);
            file_data = [filepath filesep name '.tiff.csv'];
            
            % read data file
            df_data = readtable(file_data, 'NumHeaderLines', 0);
        
            % remove nan values
            [df_data, TF] = rmmissing(df_data);
            
            % get the coordinates (deformed)
            coords = table2array(df_data(:,["x","y","z"]));  
                        
            if iter == 1
                coords_0_0 = coords(:,1:3); % initial coordinates (undeformed)
            end
            coords_0 = coords_0_0(~TF,:);
            coords_1 = coords(:,1:3);

            % rigid motion correction
            coords_1_corr = correct_aba_rigid_rot(coords_0,coords_1); 
            
            % calculate the displacement (corrected deformed coordinates - initial coordinates)
            disp_corr = coords_1_corr - coords_0; 
            
            if strcmp(direction, 'h')
                % reflection in y direction when extracting x displacement
                coords_0(:,2) = - coords_0(:,2); 
                % calculate the extension
                extensions(iter,:) = displacement_extension(coords_0(:,1:2), disp_corr(:,1));
            elseif strcmp(direction, 'v')
                % calculate the extension
                extensions(iter,:) = displacement_extension(coords_0(:,1:2), disp_corr(:,2));
            end
        end
        
        % only keep one zero load-extension
        if ind_files == 1
            extenions_each_angle{ind_files} = extensions;
            loads_each_angle{ind_files} = loads;
        else
            extenions_each_angle{ind_files} = extensions(2:end,:);
            loads_each_angle{ind_files} = loads(2:end);
        end
        
        %% save each data set into a file
        if ind_files == 1
            loads_angle = zeros(length(loads), 2);
            loads_angle(:,1) = loads;
            loads_angle(:,2) = deg2rad(angles(ind_angles));
        else
            loads_angle = zeros(length(loads)-1, 2);
            loads_angle(:,1) = loads(2:end);
            loads_angle(:,2) = deg2rad(angles(ind_angles));
            extensions = extensions(2:end,:);
        end
        
%         ind_not = (1:3 ~= ind_angles);
%         loads_angle_add = [zeros(length(angles(ind_not)),1), deg2rad(angles(ind_not))'];
%         
%         loads_angle = [loads_angle; loads_angle_add];
%         extensions = [extensions; zeros(length(angles(ind_not)),3)];

        writematrix(extensions, ['data_extension_exp_' num2str(ind_data) '.txt']);
        writematrix(loads_angle, ['input_load_angle_exp_' num2str(ind_data) '.txt']);
        ind_data = ind_data + 1;
    end

   extensions_all{ind_angles} = extenions_each_angle;
   loads_all{ind_angles} = loads_each_angle;
end

% for i = angles_index
%     figure; hold on; box on;
%     for j = 1:length(extensions_all{i})
%         scatter(mean(extensions_all{i}{j},2), loads_all{i}{j});
%     end
% 
%     temp = loads_all{i};
%     temp = cell2mat(temp'); 
%     [temp,ind] = sort(temp);
%     
%     temp_ext = extensions_all{i};
%     temp_ext = cell2mat(temp_ext');
%     for j = 1:3 % left, center, right
%         % smooth data from different specimens 
%         temp_ext(:,j) = smooth(temp_ext(ind,j),0.1,'sgolay',2); 
%     end
% 
%     s = scatter(mean(temp_ext,2), temp, 20, 'filled');
%     s.MarkerFaceAlpha = 0.7;
% 
%     xlabel('Extension [mm]');
%     ylabel('Load [kN]');
%     title([num2str(angles(i)) '^{\circ}']);
%     if i ~= 3
%         legend('Data 1', 'Data 2', 'Data 3', 'Smoothed Data')
%     else
%         legend('Data 1', 'Data 2', 'Smoothed Data')
%     end
% end

%% merge multiple data sets into one file 
% angles = deg2rad(angles);
% loads_angle = [];
% extensions_all_f = [];
% for ind_files = angles_index
%     temp = loads_all{ind_files};
%     temp = cell2mat(temp'); % put all loads into one matrix
% %     [temp,ind] = sort(temp);
% 
%     temp(:,end+1) = angles(ind_files); % add angle value
%     loads_angle = [loads_angle; temp];
%     
%     temp_ext = extensions_all{ind_files};
%     temp_ext = cell2mat(temp_ext'); % put all extensions into one matrix
% 
%     extensions_all_f = [extensions_all_f; temp_ext];
% end

%%
% for ind_files = angles_index
%     temp = loads_all{ind_files};
%     temp = cell2mat(temp'); % put all loads into one matrix
% %     temp(:,end+1) = angles(ind_files); % add angle value
% 
%     temp_ext = extensions_all{ind_files};
%     temp_ext = cell2mat(temp_ext'); % put all extensions into one matrix
%     
%     loads_t = loads_all{ind_files}{1};
%     extensions_t = zeros(length(loads_t),3);
%     for i = 1:3 % left, center, right
%         f = fit(temp,temp_ext(:,i),'poly2');
%         extensions_t(:,i) = f(loads_t);
%     end
%     
%     loads_t(:,end+1) = angles(ind_files); % add angle value
%     loads_angle = [loads_angle; loads_t];
%     extensions_all_f = [extensions_all_f; extensions_t];
% end

%% save('data_exp.mat', "loads", "extensions");
% writematrix(extensions_all_f, 'data_extension_exp.txt');
% writematrix(loads_angle, 'input_load_angle_exp.txt');

end