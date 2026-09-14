clear; clc;
%% 1. 基础路径与参数设置
baseDir = '\';

% 核心时序参数
selected_ch_i = 1:59;    % 选用通道
fs = 1000;               % 原始采样率
pre_time = 0.;          % 基线时长（0.秒）
task_time = 1.0;         % 有效数据时长（1秒）
total_time = pre_time + task_time; % 总截取时长：1.秒

% 降采样参数
fs_new = 250; 

%% 2. 遍历所有block文件夹
blockDirs = dir(fullfile(baseDir, 'block_*'));
blockDirs = blockDirs([blockDirs.isdir]); % 仅保留文件夹

for i = 1:length(blockDirs)
    blockPath = fullfile(baseDir, blockDirs(i).name);
    fprintf('\n==================== 处理第 %d 个 block：%s ====================\n', i, blockPath);
    
    % 查找bdf文件
    files = dir(fullfile(blockPath, '*.bdf'));
    if isempty(files)
        fprintf('未找到bdf文件，跳过该block\n');
        continue;
    end
    fileNames = {files.name};
    disp('找到文件：'); disp(fileNames);
    
    % 导入Neuracle数据
    EEG = pop_importNeuracle(fileNames, blockPath);
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
    EEG = pop_eegfiltnew(EEG, 'hicutoff',45);
    data = double(EEG.data);
    event = EEG.event;
    channel = {EEG.chanlocs.labels};
    clear EEG;
    
    %% 3. 通道选择
    [is_selected, idx] = ismember(channel(selected_ch_i), channel);
    data_select = data(idx, :);
    
    % 获取事件
    video_idx = strcmp({event.type}, '255');
    event_latency = round([event(video_idx).latency]);
    n_trials = length(event_latency);
    if n_trials == 0
        fprintf('未找到有效事件，跳过\n');
        continue;
    end
    
    % 采样点计算
    task_sample = round(task_time * fs);
    
    % 预分配
    final_trial = zeros(n_trials, sum(is_selected), round(task_time * fs_new));
    
    %% 逐试次处理
    for t = 1:n_trials
        % 1. 截取 1s
        start_point = event_latency(t);
        end_point = start_point + task_sample - 1;
        
        if start_point < 1 || end_point > size(data_select,2)
            fprintf('试次 %d 越界\n',t);
            continue;
        end
        
        trial_raw = data_select(:, start_point:end_point);
        
        % 2. 降采样
        trial_down = resample(trial_raw', fs_new, fs)';
        final_trial(t,:,:) = trial_down;
    end
    
    %% 保存
    saveName = sprintf('%s_final.mat', blockDirs(i).name);
    savePath = fullfile(blockPath, saveName);
    save(savePath, 'final_trial');
    fprintf('保存成功：%s\n维度：[试次,通道,时间点] = %s\n', savePath, mat2str(size(final_trial)));
end
