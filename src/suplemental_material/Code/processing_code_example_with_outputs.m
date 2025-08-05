%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT processing_code_example_with_outputs.m
%
%
% Wrapper script for processing ONE  JSON file with RUNNING data through the
% Running Injury Clinic pipeline and saving all outputs for Python testing.
%
% NOTE: If the chosen file is not a running file, the script will not process
% it.
%
% Based of processing_code_example.m found in the same folder.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% get user directories

% code_folder = uigetdir('/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/src/suplemental_material/Code');
code_folder = '/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/src/suplemental_material/Code';

% data_folder = uigetdir('/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset/source_data');
data_folder = '/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset/source_data';

% Output folder for saving results
output_folder = '/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/resources/test_data/matlab_outputs';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

files = dir([data_folder '/*/*.json']);

cur_folder = pwd;

%change folder if not on path
cd(code_folder)

%% Process ONE JSON file and save outputs

% Process only the first file for testing
i = 1;

fprintf('Processing file: %s\n', files(i).name);

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
    
    % Initialize results structure
    results = struct();
    results.subject_id = files(i).name;
    results.json_file = json_file;
    
    % Save input data for Python testing
    results.inputs.joints = out.joints;
    results.inputs.neutral = out.neutral;
    
    %check for existence of walking data
    if isfield(out,'dv_w') && ~isempty(out.walking)
        fprintf('Walking data found...\n');
        error('Walking data detected in file: %s. Aborting processing...', json_file);
    end
    
    %check for existence of running data
    if isfield(out, 'dv_r') && ~isempty(out.running)
        
        fprintf('Processing running data...\n');
        
        % Save running input data
        results.inputs.running = out.running;
        results.inputs.hz_r = out.hz_r;
        
        [r_angles,r_velocities,r_jc,r_R,r_djc] = gait_kinematics(out.joints,out.neutral,out.running,out.hz_r,plots);
        [r_norm_ang,r_norm_vel,r_events,r_event,r_DISCRETE_VARIABLES,r_speedoutput,r_eventsflag,r_label] = gait_steps(out.neutral,out.running,r_angles,r_velocities,out.hz_r,plots);
        
        % Save running outputs
        results.running.gait_kinematics = struct();
        results.running.gait_kinematics.angles = r_angles;
        results.running.gait_kinematics.velocities = r_velocities;
        results.running.gait_kinematics.joint_centers = r_jc;
        results.running.gait_kinematics.rotations = r_R;
        results.running.gait_kinematics.djc = r_djc;
        
        results.running.gait_steps = struct();
        results.running.gait_steps.norm_angles = r_norm_ang;
        results.running.gait_steps.norm_velocities = r_norm_vel;
        results.running.gait_steps.events = r_events;
        results.running.gait_steps.event = r_event;
        results.running.gait_steps.discrete_variables = r_DISCRETE_VARIABLES;
        results.running.gait_steps.speed_output = r_speedoutput;
        results.running.gait_steps.events_flag = r_eventsflag;
        results.running.gait_steps.label = r_label;
        
        fprintf('Running data processed successfully.\n');
        
    end
    
    % Save all results to .mat file
    [~, subject_name, ~] = fileparts(files(i).name);
    output_filename = fullfile(output_folder, [subject_name '_matlab_results.mat']);
    save(output_filename, 'results', '-v7.3');
    
    fprintf('Results saved to: %s\n', output_filename);
    
    % Also save individual components for easier access
    if isfield(results, 'running')
        running_output = fullfile(output_folder, [subject_name '_running_results.mat']);
        running_data = results.running;
        save(running_output, 'running_data', '-v7.3');
        fprintf('Running results saved to: %s\n', running_output);
    end
    
    % Save inputs separately for Python testing
    inputs_output = fullfile(output_folder, [subject_name '_inputs.mat']);
    inputs_data = results.inputs;
    save(inputs_output, 'inputs_data', '-v7.3');
    fprintf('Input data saved to: %s\n', inputs_output);
    
catch ME
    fprintf('Error processing file %s:\n', files(i).name);
    fprintf('Error message: %s\n', ME.message);
    fprintf('Error in: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end

%return to original folder
cd(cur_folder)

fprintf('Processing complete!\n'); 