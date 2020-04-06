p = start_script('path');
add_subpath('bin/external/na_matlab_tools/external/yamlmatlab');

% Parameters
p.study_name_list = {'dys_contr_2', 'dys'};
params_yaml = [p.settings_dir, filesep, 'params.yaml'];
params = ReadYaml(params_yaml);

%% Start plot tool
for study_name = p.study_name_list
    subj_codes = params.studies.(study_name{:}).subjects.control';
    switch study_name{:}
        case 'dys'
            % Dyslexic study
            plot_ET_on_bmp(subj_codes, @load_ET_results_dys,...
                @load_stim_bmp_dys, p.work_path, 'X', study_name{:});
        case 'dys_contr_2'
            % Dyslexic control 2 study
            plot_ET_on_bmp(subj_codes, @load_ET_results_dys_contr_2,...
                @load_stim_bmp_dys_contr_2, p.work_path, 'X', study_name{:});
    end
end

%% Adaptive ET results loading functions
function ET_results = load_ET_results_dys_contr_2(subj_code, work_path)
et_data_path = [work_path, '/data/dys_contr_2_study/ET_features/ET_results_', subj_code, '.mat'];
data = load(et_data_path);
ET_results = data.ET_results;
end

function ET_results = load_ET_results_dys(subj_code, work_path)
et_data_path = [work_path, '/data/dys_study/ET_features/', subj_code, '_TextLinesWithLeftValidation_correctionType1_limitedToReading.mat'];
data = load(et_data_path);
ET_results = data.ETresults;
end

%% Stim bmp loading functions
function img = load_stim_bmp_dys_contr_2(subj_code, stim_num, work_path)
stim_data_path = [work_path, '/data/dys_contr_2_study/subject_data/', subj_code, '/', subj_code, '_senpres.mat'];
bmp_dir = [work_path, '/data/dys_contr_2_study/stimuli/leftaligned'];

% Load sen pres data
data = load(stim_data_path);
sen_pres_data = data.sen_pres_data;

% Get bmp file name based on ID
id = sen_pres_data(stim_num,1);
file_name_pattern = ['id', num2str(id), '_sen_'];
file_list = dir(bmp_dir);
file_idx = find(~cellfun(@isempty, strfind({file_list.name}, file_name_pattern)), 1);
file_name = file_list(file_idx).name;

% Get image
img = imread([bmp_dir, filesep, file_name]);

% Get condition
end

function img = load_stim_bmp_dys(subj_code, stim_num, work_path)
stim_data_path = [work_path, '/data/dys_study/subject_data/', subj_code, '/', subj_code, '_senpres.mat'];
bmp_dir = [work_path, '/data/dys_study/stimuli/leftaligned'];

% Load sen pres data
data = load(stim_data_path);
sen_pres_data = data.sen_pres_data;

% Get bmp file name based on ID
id = sen_pres_data(stim_num,1);
file_name_pattern = ['id', num2str(id), '_sen_'];
file_list = dir(bmp_dir);
file_idx = find(~cellfun(@isempty, strfind({file_list.name}, file_name_pattern)), 1);
file_name = file_list(file_idx).name;

% Get image
img = imread([bmp_dir, filesep, file_name]);
end

