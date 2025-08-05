%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT processing_code_example_with_outputs.m
%
%
% Wrapper script for processing JSON files through the Running Injury Clinic
% pipeline and saving all outputs for Python testing.
%
% Can process either a single file or all files in the dataset.
% Outputs are saved in both MATLAB format (.mat) and CSV format for
% easy loading into Python dataframes.
%
% Based on processing_code_example.m found in the same folder.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Configuration

% Set processing mode: 'single' or 'all'
PROCESSING_MODE = 'single';  % Change to 'all' to process all files

% Set which file to process if using single mode (1 = first file, 2 = second file, etc.)
SINGLE_FILE_INDEX = 1;

fprintf('Processing mode: %s\n', PROCESSING_MODE);
if strcmp(PROCESSING_MODE, 'single')
    fprintf('File index: %d\n', SINGLE_FILE_INDEX);
end

% code_folder = uigetdir('/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/src/suplemental_material/Code');
code_folder = '/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/src/suplemental_material/Code';

% data_folder = uigetdir('/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset/source_data');
data_folder = '/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset/source_data';

% Output folder for saving results
output_folder = '/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset/processed_data/matlab_output';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Create subfolders for different output formats
csv_folder = fullfile(output_folder, 'csv');
mat_folder = fullfile(output_folder, 'mat');
if ~exist(csv_folder, 'dir'), mkdir(csv_folder); end
if ~exist(mat_folder, 'dir'), mkdir(mat_folder); end

files = dir([data_folder '/*/*.json']);


cur_folder = pwd;

%change folder if not on path
cd(code_folder)

%% Determine which files to process

if strcmp(PROCESSING_MODE, 'single')
    if SINGLE_FILE_INDEX <= length(files)
        file_indices = SINGLE_FILE_INDEX;
        fprintf('Processing single file (index %d): %s\n', SINGLE_FILE_INDEX, files(SINGLE_FILE_INDEX).name);
    else
        error('File index %d exceeds available files (%d)', SINGLE_FILE_INDEX, length(files));
    end
elseif strcmp(PROCESSING_MODE, 'all')
    file_indices = 1:length(files);
    fprintf('Processing all %d files...\n', length(files));
else
    error('Invalid PROCESSING_MODE. Use "single" or "all"');
end

%% Initialize summary tracking
summary_data = cell(length(file_indices), 6);  % Preallocate based on number of files to process
summary_headers = {'ID', 'SubjectID', 'SessionID', 'JsonFile', 'ProcessingStatus', 'ErrorMessage'};
summary_row = 0;  % Track current row

%% Initialize discrete variables tracking
discrete_vars_data = [];
discrete_vars_headers = [];
discrete_vars_initialized = false;

%% Process files

% Initialize waitbar
total_files = length(file_indices);
h = waitbar(0, sprintf('Processing files... (0/%d)', total_files), 'Name', 'MATLAB Processing Progress');

try
    for idx = 1:length(file_indices)
    i = file_indices(idx);
    
    % Update waitbar
    progress_percent = (idx - 1) / total_files;
    waitbar(progress_percent, h, sprintf('Processing file %d/%d: %s', idx, total_files, files(i).name));
    
    fprintf('\n--- Processing file %d/%d: %s ---\n', i, length(files), files(i).name);
    
    try
        %get fully defined path to json data file
        json_file = fullfile(files(i).folder, files(i).name);

        % Extract subject_id and session_id from file path
        [parent_folder, session_id, ext] = fileparts(json_file);
        [~, subject_id] = fileparts(parent_folder);

        % Match id column in Python tables
        id = [subject_id '_' session_id]; 

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
        results.id = id;
        results.session_id = session_id;
        results.subject_id = subject_id;
        results.json_file = json_file;
        
        % Save input data for Python testing
        results.inputs.joints = out.joints;
        results.inputs.neutral = out.neutral;
        
        % Check data availability (only for running data)
        has_running = isfield(out, 'dv_r') && ~isempty(out.running);
        
        fprintf('Data availability: Running=%d\n', has_running);
        
        %check for existence of running data
        if has_running
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
            % results.running.gait_kinematics.rotations = r_R;
            results.running.gait_kinematics.djc = r_djc;
            
            results.running.gait_steps = struct();
            results.running.gait_steps.norm_angles = r_norm_ang;
            results.running.gait_steps.norm_velocities = r_norm_vel;
            % results.running.gait_steps.events = r_events;
            results.running.gait_steps.event = r_event;
            results.running.gait_steps.discrete_variables = r_DISCRETE_VARIABLES;
            results.running.gait_steps.speed_output = r_speedoutput;
            % results.running.gait_steps.events_flag = r_eventsflag;
            results.running.gait_steps.label = r_label;
            
            % Create individual folder for this file
            subject_folder = fullfile(output_folder, id);
            if ~exist(subject_folder, 'dir')
                mkdir(subject_folder);
            end
            
            % Create subfolders for results and inputs
            results_folder = fullfile(subject_folder, 'results');
            inputs_folder = fullfile(subject_folder, 'inputs');
            if ~exist(results_folder, 'dir'), mkdir(results_folder); end
            if ~exist(inputs_folder, 'dir'), mkdir(inputs_folder); end
            
            % Export running data to CSV with new structure
            export_to_csv(results.running, results_folder);
            
            % Export inputs data to CSV
            export_inputs_to_csv(results.inputs, inputs_folder);
            
            % Collect discrete variables for combined file
            [discrete_vars_data, discrete_vars_headers, discrete_vars_initialized] = ...
                collect_discrete_variables(results.running, id, r_speedoutput, r_label, out.hz_r, ...
                                         discrete_vars_data, discrete_vars_headers, discrete_vars_initialized);
            
            fprintf('Running data processed successfully.\n');
        end

        if ~has_running
            fprintf('No running data found in file %s. Skipping...\n', files(i).name);
            % Check for label mismatch: if r_label exists and is 'run', but has_running is false
            if strcmpi(r_label, 'run')
                warning('Label indicates running (r_label = ''run'') but no running data found (has_running = false) in file %s.', files(i).name);
            end
        elseif ~strcmpi(r_label, 'run')
            % If running data exists but label is not 'run', raise a warning
            warning('Running data found (has_running = true) but label is not ''run'' (r_label = ''%s'') in file %s.', r_label, files(i).name);
        end
        
        % Save all results to .mat file
        output_filename = fullfile(mat_folder, [subject_id '_matlab_results.mat']);
        save(output_filename, 'results', '-v7.3');
                
        % Update summary
        summary_row = summary_row + 1;
        summary_data(summary_row, :) = {id, subject_id, session_id, json_file, 'Success', ''};
        
        fprintf('Results saved for subject %s\n', subject_id);
        
    catch ME
        fprintf('Error processing file %s:\n', files(i).name);
        fprintf('Error message: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('Error in: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        
        % Update summary with error
        [~, subject_id, ~] = fileparts(files(i).name);
        summary_row = summary_row + 1;
        summary_data(summary_row, :) = {id, subject_id, session_id, json_file, 'Error', ME.message};
    end
    
    end

    % Complete waitbar and close
    waitbar(1, h, sprintf('Processing complete! (%d/%d files processed)', total_files, total_files));
    pause(1); % Brief pause to show completion
    close(h);
    
catch ME
    % Ensure waitbar is closed even if an error occurs
    if exist('h', 'var') && ishandle(h)
        close(h);
    end
    rethrow(ME);
end

%% Save processing summary
summary_table = cell2table(summary_data, 'VariableNames', summary_headers);
summary_csv = fullfile(output_folder, 'processing_summary.csv');
writetable(summary_table, summary_csv);

%% Save combined discrete variables
if discrete_vars_initialized && ~isempty(discrete_vars_data)
    discrete_vars_table = cell2table(discrete_vars_data, 'VariableNames', discrete_vars_headers);
    discrete_vars_csv = fullfile(output_folder, 'discrete_variables.csv');
    writetable(discrete_vars_table, discrete_vars_csv);
    fprintf('💾 Discrete variables saved as: %s\n', discrete_vars_csv);
end

fprintf('\n📊 Processing Summary:\n');
fprintf('Total files processed: %d\n', height(summary_table));
fprintf('Successful: %d\n', sum(strcmp(summary_table.ProcessingStatus, 'Success')));
fprintf('Errors: %d\n', sum(strcmp(summary_table.ProcessingStatus, 'Error')));

%return to original folder
cd(cur_folder)

fprintf('\nProcessing complete!\n');
fprintf('📁 MAT files saved in: %s\n', mat_folder);
fprintf('📊 Individual CSV files saved in subject folders under: %s\n', output_folder);
fprintf('📋 Summary saved as: %s\n', summary_csv);

%% Helper function to export data to CSV format with new structure
function export_to_csv(gait_data, output_folder)
    % Export gait analysis results to CSV files in individual subject folders
    
    % Export joint angles (one file per joint)
    if isfield(gait_data, 'gait_kinematics') && isfield(gait_data.gait_kinematics, 'angles')
        angles = gait_data.gait_kinematics.angles;
        joints = fieldnames(angles);
        
        for j = 1:length(joints)
            joint_name = joints{j};
            joint_data = angles.(joint_name);
            
            % Ensure joint_data has at least 3 columns, pad with NaN if necessary
            if size(joint_data, 2) < 3
                warning('Joint "%s" angles data has only %d columns instead of expected 3. Padding with NaN.', joint_name, size(joint_data, 2));
                joint_data = [joint_data, NaN(size(joint_data, 1), 3 - size(joint_data, 2))];
            end
            
            % Create table with time index and x,y,z columns
            n_samples = size(joint_data, 1);
            time_idx = (1:n_samples)';
            
            angle_table = table(time_idx, joint_data(:,1), joint_data(:,2), joint_data(:,3), ...
                'VariableNames', {'TimeIndex', 'X_deg', 'Y_deg', 'Z_deg'});
            
            filename = fullfile(output_folder, sprintf('%s_angles.csv', joint_name));
            writetable(angle_table, filename);
        end
    end
    
    % Export joint velocities (one file per joint)
    if isfield(gait_data, 'gait_kinematics') && isfield(gait_data.gait_kinematics, 'velocities')
        velocities = gait_data.gait_kinematics.velocities;
        joints = fieldnames(velocities);
        
        for j = 1:length(joints)
            joint_name = joints{j};
            joint_data = velocities.(joint_name);
            
            % Ensure joint_data has at least 3 columns, pad with NaN if necessary
            if size(joint_data, 2) < 3
                warning('Joint "%s" velocities data has only %d columns instead of expected 3. Padding with NaN.', joint_name, size(joint_data, 2));
                joint_data = [joint_data, NaN(size(joint_data, 1), 3 - size(joint_data, 2))];
            end
            
            n_samples = size(joint_data, 1);
            time_idx = (1:n_samples)';
            
            vel_table = table(time_idx, joint_data(:,1), joint_data(:,2), joint_data(:,3), ...
                'VariableNames', {'TimeIndex', 'X_deg_per_s', 'Y_deg_per_s', 'Z_deg_per_s'});
            
            filename = fullfile(output_folder, sprintf('%s_velocities.csv', joint_name));
            writetable(vel_table, filename);
        end
    end
    
    % Export normalized angles (one file per joint)
    if isfield(gait_data, 'gait_steps') && isfield(gait_data.gait_steps, 'norm_angles')
        norm_angles = gait_data.gait_steps.norm_angles;
        joints = fieldnames(norm_angles);
        
        for j = 1:length(joints)
            joint_name = joints{j};
            joint_data = norm_angles.(joint_name);
            
            % Ensure joint_data has at least 3 columns, pad with NaN if necessary
            if size(joint_data, 2) < 3
                warning('Joint "%s" normalized angles data has only %d columns instead of expected 3. Padding with NaN.', joint_name, size(joint_data, 2));
                joint_data = [joint_data, NaN(size(joint_data, 1), 3 - size(joint_data, 2))];
            end
            
            n_samples = size(joint_data, 1);
            percent_gait_cycle = (1:n_samples)' / n_samples * 100;
            
            norm_table = table(percent_gait_cycle, joint_data(:,1), joint_data(:,2), joint_data(:,3), ...
                'VariableNames', {'PercentGaitCycle', 'X_deg', 'Y_deg', 'Z_deg'});
            
            filename = fullfile(output_folder, sprintf('%s_norm_angles.csv', joint_name));
            writetable(norm_table, filename);
        end
    end
    
    % Export normalized velocities (one file per joint)
    if isfield(gait_data, 'gait_steps') && isfield(gait_data.gait_steps, 'norm_velocities')
        norm_velocities = gait_data.gait_steps.norm_velocities;
        joints = fieldnames(norm_velocities);
        
        for j = 1:length(joints)
            joint_name = joints{j};
            joint_data = norm_velocities.(joint_name);
            
            % Ensure joint_data has at least 3 columns, pad with NaN if necessary
            if size(joint_data, 2) < 3
                warning('Joint "%s" normalized velocities data has only %d columns instead of expected 3. Padding with NaN.', joint_name, size(joint_data, 2));
                joint_data = [joint_data, NaN(size(joint_data, 1), 3 - size(joint_data, 2))];
            end
            
            n_samples = size(joint_data, 1);
            percent_gait_cycle = (1:n_samples)' / n_samples * 100;
            
            norm_vel_table = table(percent_gait_cycle, joint_data(:,1), joint_data(:,2), joint_data(:,3), ...
                'VariableNames', {'PercentGaitCycle', 'X_deg_per_s', 'Y_deg_per_s', 'Z_deg_per_s'});
            
            filename = fullfile(output_folder, sprintf('%s_norm_velocities.csv', joint_name));
            writetable(norm_vel_table, filename);
        end
    end
    
    % Export joint centers
    if isfield(gait_data, 'gait_kinematics') && isfield(gait_data.gait_kinematics, 'joint_centers')
        joint_centers = gait_data.gait_kinematics.joint_centers;
        joints = fieldnames(joint_centers);
        
        % Create combined table for all joint centers
        jc_data = [];
        joint_names = {};
        
        for j = 1:length(joints)
            joint_name = joints{j};
            joint_data = joint_centers.(joint_name);
            
            % Ensure joint_data has at least 3 columns, pad with NaN if necessary
            if size(joint_data, 2) < 3
                warning('Joint "%s" centers data has only %d columns instead of expected 3. Padding with NaN.', joint_name, size(joint_data, 2));
                joint_data = [joint_data, NaN(size(joint_data, 1), 3 - size(joint_data, 2))];
            end
            
            n_samples = size(joint_data, 1);
            jc_data = [jc_data; joint_data(:, 1:3)];  % Only take first 3 columns
            joint_names = [joint_names; repmat({joint_name}, n_samples, 1)];
        end
        
        if ~isempty(jc_data)
            jc_table = table(joint_names, jc_data(:,1), jc_data(:,2), jc_data(:,3), ...
                'VariableNames', {'Joint', 'X_coord', 'Y_coord', 'Z_coord'});
            
            filename = fullfile(output_folder, 'joint_centers.csv');
            writetable(jc_table, filename);
        end
    end
    
    % Export djc (joint center derivatives)
    if isfield(gait_data, 'gait_kinematics') && isfield(gait_data.gait_kinematics, 'djc')
        djc = gait_data.gait_kinematics.djc;
        joints = fieldnames(djc);
        
        % Create combined table for all djc
        djc_data = [];
        joint_names = {};
        
        for j = 1:length(joints)
            joint_name = joints{j};
            joint_data = transpose(djc.(joint_name));
            
            % Ensure joint_data has at least 3 columns, pad with NaN if necessary
            if size(joint_data, 2) < 3
                warning('Joint "%s" djc data has only %d columns instead of expected 3. Padding with NaN.', joint_name, size(joint_data, 2));
                joint_data = [joint_data, NaN(size(joint_data, 1), 3 - size(joint_data, 2))];
            end
            
            n_samples = size(joint_data, 1);
            djc_data = [djc_data; joint_data(:, 1:3)];  % Only take first 3 columns
            joint_names = [joint_names; repmat({joint_name}, n_samples, 1)];
        end
        
        if ~isempty(djc_data)
            djc_table = table(joint_names, djc_data(:,1), djc_data(:,2), djc_data(:,3), ...
                'VariableNames', {'Joint', 'X_velocity', 'Y_velocity', 'Z_velocity'});
            
            filename = fullfile(output_folder, 'djc.csv');
            writetable(djc_table, filename);
        end
    end
    
    % Export events
    if isfield(gait_data, 'gait_steps') && isfield(gait_data.gait_steps, 'event')
        event_data = gait_data.gait_steps.event;
        
        if ~isempty(event_data)
            event_table = table(event_data, 'VariableNames', {'EventIndex'});
            event_table.EventNumber = (1:length(event_data))';
            
            % Reorder columns
            event_table = event_table(:, {'EventNumber', 'EventIndex'});
            
            filename = fullfile(output_folder, 'event.csv');
            writetable(event_table, filename);
        end
    end
    
end

%% Helper function to collect discrete variables for combined file
function [discrete_vars_data, discrete_vars_headers, discrete_vars_initialized] = ...
    collect_discrete_variables(gait_data, id, speed_output, label, hz, discrete_vars_data, discrete_vars_headers, discrete_vars_initialized)
    
    if isfield(gait_data, 'gait_steps') && isfield(gait_data.gait_steps, 'discrete_variables')
        discrete_vars = gait_data.gait_steps.discrete_variables;

        % Define variable names for the 77 discrete variables
        var_names = {
            'Variable_1', 'Step_Width', 'Stride_Rate', 'Stride_Length', 'Swing_Time', ...
            'Stance_Time', 'Pelvis_Peak_Drop_Angle', 'Pelvis_Drop_Percent_Stance', ...
            'Pelvis_Drop_At_HS', 'Pelvis_Drop_Excursion', 'Ankle_DF_Peak_Angle', ...
            'Ankle_DF_Percent_Stance', 'Ankle_DF_At_HS', 'Ankle_DF_Excursion', ...
            'Ankle_Eve_Peak_Angle', 'Ankle_Eve_Percent_Stance', 'Ankle_Eve_At_HS', ...
            'Ankle_Eve_Excursion', 'Ankle_Rot_Peak_Angle', 'Ankle_Rot_Percent_Stance', ...
            'Ankle_Rot_At_HS', 'Ankle_Rot_Excursion', 'Knee_Flex_Peak_Angle', ...
            'Knee_Flex_Percent_Stance', 'Knee_Flex_At_HS', 'Knee_Flex_Excursion', ...
            'Knee_Add_Peak_Angle', 'Knee_Add_Percent_Stance', 'Knee_Add_At_HS', ...
            'Knee_Add_Excursion', 'Knee_Abd_Peak_Angle', 'Knee_Abd_Percent_Stance', ...
            'Knee_Abd_At_HS', 'Knee_Abd_Excursion', 'Knee_Rot_Peak_Angle', ...
            'Knee_Rot_Percent_Stance', 'Knee_Rot_At_HS', 'Knee_Rot_Excursion', ...
            'Hip_Ext_Peak_Angle', 'Hip_Ext_Percent_Stance', 'Hip_Ext_At_HS', ...
            'Hip_Ext_Excursion', 'Hip_Add_Peak_Angle', 'Hip_Add_Percent_Stance', ...
            'Hip_Add_At_HS', 'Hip_Add_Excursion', 'Hip_Rot_Peak_Angle', ...
            'Hip_Rot_Percent_Stance', 'Hip_Rot_At_HS', 'Hip_Rot_Excursion', ...
            'Foot_Prog_Angle', 'Foot_Ang_At_HS', 'Foot_Ang_At_TO', 'Med_Heel_Whip_Peak', ...
            'MHW_Percent_Swing', 'MHW_Exc_From_TO', 'Ankle_DF_Peak_Vel', ...
            'Ankle_DF_Vel_Percent_Stance', 'Ankle_Eve_Peak_Vel', 'Ankle_Eve_Vel_Percent_Stance', ...
            'Ankle_Rot_Peak_Vel', 'Ankle_Rot_Vel_Percent_Stance', 'Knee_Flex_Peak_Vel', ...
            'Knee_Flex_Vel_Percent_Stance', 'Knee_Abd_Peak_Vel', 'Knee_Abd_Vel_Percent_Stance', ...
            'Knee_Add_Peak_Vel', 'Knee_Add_Vel_Percent_Stance', 'Hip_Abd_Peak_Vel', ...
            'Hip_Abd_Vel_Percent_Stance', 'Knee_Rot_Peak_Vel', 'Hip_Rot_Peak_Vel', ...
            'Pronation_Onset', 'Supination_Timing', 'Hip_Add_Peak_Vel', 'Pelvic_Drop_Peak_Vel', ...
            'Vertical_Oscillation'
        };
        
        % Initialize headers if this is the first file
        if ~discrete_vars_initialized
            discrete_vars_headers = {'ID', 'Speed_Output', 'Label', 'Hz'};
            
            % Only add variables that have non-empty/non-zero values
            for v = 1:min(length(var_names), size(discrete_vars, 1))
                left_val = discrete_vars(v, 2);
                right_val = discrete_vars(v, 3);
                
                % Check if values are non-empty and not just placeholders (0 or NaN)
                if ~(isnan(left_val) && isnan(right_val)) && ~(left_val == 0 && right_val == 0)
                    var_name = var_names{v};
                    left_col_name = sprintf('%s_Left', var_name);
                    right_col_name = sprintf('%s_Right', var_name);
                    discrete_vars_headers{end+1} = left_col_name;
                    discrete_vars_headers{end+1} = right_col_name;
                end
            end
            discrete_vars_initialized = true;
        end
        
        % Create row data for this file
        row_data = {id, speed_output, label, hz};
        
        % Add discrete variable values (only for non-empty variables)
        for v = 1:min(length(var_names), size(discrete_vars, 1))
            left_val = discrete_vars(v, 2);
            right_val = discrete_vars(v, 3);
            
            % Only add if values are non-empty and not just placeholders
            if ~(isnan(left_val) && isnan(right_val)) && ~(left_val == 0 && right_val == 0)
                row_data{end+1} = left_val;   % Left side
                row_data{end+1} = right_val;  % Right side
            end
        end
        
                 % Add this row to the combined data
         discrete_vars_data = [discrete_vars_data; row_data];
     end
end

%% Helper function to export inputs data to CSV format
function export_inputs_to_csv(inputs_data, output_folder)
    % Export input data (joints, neutral, etc.) to CSV files
    
    % Export neutral joint markers
    if isfield(inputs_data, 'neutral')
        neutral = inputs_data.neutral;
        joints = fieldnames(neutral);
        
        % Create combined table for all neutral markers
        neutral_data = [];
        joint_names = {};
        
        for j = 1:length(joints)
            joint_name = joints{j};
            joint_data = neutral.(joint_name);
            
            % Ensure joint_data has at least 3 columns, pad with NaN if necessary
            if size(joint_data, 2) < 3
                warning('Neutral joint "%s" data has only %d columns instead of expected 3. Padding with NaN.', joint_name, size(joint_data, 2));
                joint_data = [joint_data, NaN(size(joint_data, 1), 3 - size(joint_data, 2))];
            end
            
            n_samples = size(joint_data, 1);
            neutral_data = [neutral_data; joint_data(:, 1:3)];  % Only take first 3 columns
            joint_names = [joint_names; repmat({joint_name}, n_samples, 1)];
        end
        
        if ~isempty(neutral_data)
            neutral_table = table(joint_names, neutral_data(:,1), neutral_data(:,2), neutral_data(:,3), ...
                'VariableNames', {'Joint', 'X_coord', 'Y_coord', 'Z_coord'});
            
            filename = fullfile(output_folder, 'neutral_joint_markers.csv');
            writetable(neutral_table, filename);
        end
    end
    
    % Export joint markers
    if isfield(inputs_data, 'joints')
        joints = inputs_data.joints;
        joint_names = fieldnames(joints);
        
        % Create combined table for all joint markers
        joints_data = [];
        joint_name_list = {};
        
        for j = 1:length(joint_names)
            joint_name = joint_names{j};
            joint_data = joints.(joint_name);
            
            % Ensure joint_data has at least 3 columns, pad with NaN if necessary
            if size(joint_data, 2) < 3
                warning('Joint marker "%s" data has only %d columns instead of expected 3. Padding with NaN.', joint_name, size(joint_data, 2));
                joint_data = [joint_data, NaN(size(joint_data, 1), 3 - size(joint_data, 2))];
            end
            
            n_samples = size(joint_data, 1);
            joints_data = [joints_data; joint_data(:, 1:3)];  % Only take first 3 columns
            joint_name_list = [joint_name_list; repmat({joint_name}, n_samples, 1)];
        end
        
        if ~isempty(joints_data)
            joints_table = table(joint_name_list, joints_data(:,1), joints_data(:,2), joints_data(:,3), ...
                'VariableNames', {'Joint', 'X_coord', 'Y_coord', 'Z_coord'});
            
            filename = fullfile(output_folder, 'joint_markers.csv');
            writetable(joints_table, filename);
        end
    end
    
end 