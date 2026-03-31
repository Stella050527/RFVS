% =========================================================================
% 无人机射频(RF)原始IQ数据转换为STFT时频图的示例代码
% =========================================================================

% 1. 设置文件路径 (使用相对路径方便开源分享，请确保数据在对应目录下)
% 假设我们将该脚本放在项目根目录，数据存放在 RF_raw/val 等目录下
filename = './RF_raw/val/sumCorr_15';  % 修改为你要读取的示例文件路径
outDir = './output_images';            % 时频图输出目录

% 打开文件并检查
fid = fopen(filename, 'r');
if fid == -1
    error('无法打开文件，请检查文件路径是否正确: %s', filename);
end

% 读取文件头信息（按指定偏移量）
fseek(fid, 157, -1);       % 移动到频率信息位置（初始位置为文件开头）
freq = fread(fid, 1, 'float32') / 1e6;  % 读取频率信息（单位转换为MHz）

fseek(fid, 248, -1);       % 移动到数据起始位置

% 读取IQ数据（short类型，即int16）
data = fread(fid, inf, 'short');  % 读取所有数据
fclose(fid);  % 关闭文件

% 2. 数据处理：转换为复信号（IQ组合）
data = data(:);  % 确保为列向量
s = 1 * data(1:2:end) + 1j * data(2:2:end);  % 实部+虚部组合为复信号

% 3. STFT参数设置 (需根据实际采集设备的采样率调整Fs)
Fs = 153.6e6;            % 采样率（单位：Hz），请根据实际数据修改！！！
window = hamming(1024);  % 窗函数（可调整窗长，如256、512等）
noverlap = 512;          % 重叠点数（通常为窗长的1/2）
nfft = 1024;             % FFT点数

% 执行STFT
fprintf('正在计算STFT...\n');
[S, F, T] = stft(s, Fs, ...
    'Window', window, ...
    'OverlapLength', noverlap, ...
    'FFTLength', nfft);

% 4. 绘制并保存时频图
% 创建一个不可见的Figure，加快批量处理速度并防止弹窗打扰
fig = figure('Visible', 'off'); 

% 显示时频图（幅度转换为dB，频率单位转换为MHz）
imagesc(T, F/1e6, 20*log10(abs(S)));
axis xy;  % 频率轴从下到上递增（符合常规显示习惯）
colormap('jet');  % 使用jet颜色映射（增强对比度）

% ===== 只保存坐标轴内图像（不含标题/轴标签/刻度/色条/白边） =====
% 如果你想保留标题和坐标轴用于文章配图，请注释掉下面两行
axis off; 
set(gca, 'Position', [0 0 1 1]); 

% 检查输出文件夹是否存在，不存在则创建
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% 构造输出文件名（用原始文件名 + 中心频率，避免覆盖）
[~, baseName, ~] = fileparts(filename);
outFile = fullfile(outDir, sprintf("%s_%.3fMHz.png", baseName, freq));

% 导出图像
saveas(fig, outFile);
close(fig);

fprintf('处理成功！纯净时频图已保存至: %s\n', outFile);