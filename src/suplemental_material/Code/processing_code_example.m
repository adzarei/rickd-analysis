%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT processing_code_example.m
% 
%
% Wrapper script for processing data through the Running Injury Clinic
% pipeline
%
% Main functions in wrapper: GAIT_KINEMATICS, GAIT_STEPS. Please open
% these functions for detailed description of INPUTS and OUTPUTS
%
%  LICENSE
%  -------
%  See file LICENSE.txt
%
% Copyright (C) 2022-2023 Allan Brett and the Running Injury Clinic
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% get user directories

code_folder = uigetdir('C:\','Select folder containing PROCESSING code');

data_folder = uigetdir('C:\','Select folder containing JSON data');

files = dir([data_folder '\*\*.json']);

cur_folder = pwd;

f = waitbar(0,'Please wait...');

%change folder if not on path
cd(code_folder)

%% loop through json files and process

err_log = cell(size(files,1),4);

for i = 1%size(files,1) uncomment to process all files
    
     waitbar(i/size(files,1),f,['Processing subjects ' num2str(i) ' of ' num2str(size(files,1)) ' JSON files']);
    
    try
        %get fully defined path to json data file
        json_file = fullfile(files(i).folder, files(i).name);
        
        %load json file
        fid = fopen(json_file);
        raw = fread(fid,inf);
        str = char(raw');
        fclose(fid);
        out = jsondecode(str);
        
        plots = 0;
        
        %IMPORTANT!%
        %writing to json does not faithfully recreate the structure of
        %out.joints, and out.neutral as stored in original .MAT FILE.
        %Must reformat beforehand for INPUT into gait_kinematics and 
        %gait_steps
        
        fields = fieldnames(out.joints);
             
        for j = 1:size(fields,1)
            
            out.joints.(fields{j,1}) = transpose(out.joints.(fields{j,1}));
            
        end
        
        clear fields
        fields = fieldnames(out.neutral);
        
        for j = 1:size(fields,1)
            
            out.neutral.(fields{j,1}) = transpose(out.neutral.(fields{j,1}));
            
        end
        
        %check for existence of walking data
        if isfield(out,'dv_w') && ~isempty(out.walking)

            [w_angles,w_velocities,w_jc,w_R,w_djc] = gait_kinematics(out.joints,out.neutral,out.walking,out.hz_w,plots);
            [w_norm_ang,w_norm_vel,w_events,w_event,w_DISCRETE_VARIABLES,w_speedoutput,w_eventsflag,w_label] = gait_steps(out.neutral,out.walking,w_angles,w_velocities,out.hz_w,plots);
            
        end
        
        %check for existence of running data
        if isfield(out, 'dv_r') && ~isempty(out.running)
            
            [r_angles,r_velocities,r_jc,r_R,r_djc] = gait_kinematics(out.joints,out.neutral,out.running,out.hz_r,plots);
            [r_norm_ang,r_norm_vel,r_events,r_event,r_DISCRETE_VARIABLES,r_speedoutput,r_eventsflag,r_label] = gait_steps(out.neutral,out.running,r_angles,r_velocities,out.hz_r,plots);           
            
        end
        
        clearvars -except i f files err_log cur_folder
              
    catch ME
        
        err_log{i,1} = i;
        err_log{i,2} = {'Failed processing json file'};
        err_log{i,3} = json_file;
        err_log{i,4} = ME;
        
    end
    
end

%return to original folder
cd(cur_folder)

%% basic error logging (uncomment to use)

% curdate= date;
% today = strrep(curdate, '-', '');
% 
% if size(err_log,1) > 0
%     err_log = err_log(~cellfun(@isempty, err_log(:,1)), :);
%     save([cur_folder '\' today '_errlog'], 'err_log');
% end

