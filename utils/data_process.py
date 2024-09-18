import os
import pandas as pd

# 文件夹路径
folder_path = 'Spectrum Data-20240918'  # 修改为实际路径

# 获取所有文件名并按照自然顺序排序
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".dat")])

# 用于存储所有文件的CH3列和Wavelength列
data_frames = []

# 遍历文件夹中的所有.dat文件
for i, filename in enumerate(file_list):
    # 构造完整的文件路径
    file_path = os.path.join(folder_path, filename)
    
    # 读取文件并跳过第一行的列名（因为没有合适的分隔符，我们手动命名列）
    col_names = ['Wavelength', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']
    data = pd.read_csv(file_path, sep='\s+', names=col_names, skiprows=2)

    # 提取 Wavelength 和 CH3 列，并重命名 CH3 列为独特的名字
    df = data[['Wavelength', 'CH1']].copy()
    df.columns = ['Wavelength', f'CH1_File_{i+1}']
    
    # 将每个文件的数据存入data_frames列表
    data_frames.append(df)

# 合并所有文件的数据，按列合并（基于Wavelength）
all_data = data_frames[0]  # 第一个文件的数据作为基础
for df in data_frames[1:]:
    all_data = pd.merge(all_data, df, on='Wavelength', how='outer')

# 将结果写入Excel文件
output_path = 'Data_Sets/output_noise_new.xlsx'  # 修改为实际输出路径
all_data.to_excel(output_path, index=False)

print(f"数据已成功写入 {output_path}")