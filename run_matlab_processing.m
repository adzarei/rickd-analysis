%% Example script to run MATLAB processing and generate outputs for Python testing
%
% This script demonstrates how to run the updated processing script
% that saves outputs for comparison with Python implementation.

clear all; close all; clc;

fprintf('Starting MATLAB processing for Python comparison...\n');
fprintf('=================================================\n\n');

% Record start time
start_time = tic;

try
    % Add the code directory to path if needed
    code_dir = '/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/src/suplemental_material/Code';
    if ~isempty(code_dir) && exist(code_dir, 'dir')
        addpath(code_dir);
        fprintf('Added code directory to path: %s\n', code_dir);
    end
    
    % Run the processing script
    fprintf('Running processing_code_example_with_outputs...\n\n');
    run('processing_code_example_with_outputs.m');
    
    % Record end time
    elapsed_time = toc(start_time);
    
    fprintf('\n=================================================\n');
    fprintf('MATLAB processing completed successfully!\n');
    fprintf('Elapsed time: %.2f seconds\n', elapsed_time);
    fprintf('=================================================\n\n');
    
    % Provide next steps
    fprintf('Next steps:\n');
    fprintf('1. Check the matlab_outputs directory for generated .mat files\n');
    fprintf('2. Run the Python comparison script:\n');
    fprintf('   cd /Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis\n');
    fprintf('   python test_matlab_vs_python.py\n\n');
    
catch ME
    fprintf('\n❌ Error occurred during processing:\n');
    fprintf('Error message: %s\n', ME.message);
    fprintf('Error in: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    
    if length(ME.stack) > 1
        fprintf('Call stack:\n');
        for i = 1:length(ME.stack)
            fprintf('  %d. %s (line %d)\n', i, ME.stack(i).name, ME.stack(i).line);
        end
    end
end 