import os
import numpy as np
import matplotlib.pyplot as plt




def append_array_to_txt(filename, array, delimiter=',', fmt='%.2f'):
    
    with open(filename, 'a') as f:
        np.savetxt(f, array, fmt=fmt, delimiter=delimiter)


def write_data_to_txt(file_list, txt_name, folder_path):
    for i, filename in enumerate(file_list):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        open_file = open(file_path, 'r')
        lines = open_file.readlines()
        data = np.zeros((2,2000))
        for i in range(2000):
            data[0,i] , data[1,i] = lines[i+2].split('\t')[2:4]
        append_array_to_txt(txt_name, data)


def ckeck_peaks(file_list, folder_path):
    '检查ch2为4个峰 ch3为3个峰'
    x = np.linspace(0, 2000, 2000)
    for i, filename in enumerate(file_list):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        open_file = open(file_path, 'r')
        lines = open_file.readlines()
        data = np.zeros((2,2000))
        for i in range(2000):
            data[0,i] , data[1,i] = lines[i+2].split('\t')[2:4]
        plt.plot(x, data[0,:], label='ch2', color='r')
        plt.plot(x, data[1,:], label='ch3', color='b')
        plt.legend()
        plt.show()
        break

# txt_name = 'data.txt'
# folder_path = '../Data_sets/Fiber_7points/bei30/Spectrum Data-20240930'  # 修改为实际路径
# file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".dat")])
        
array = np.empty((384, 3))
for i in range(0, 24):
    for j in range(0, 16):
        new_data = [0, i, j]
        array[i * 16 + j] = new_data
y = array
append_array_to_txt('lable.txt', y, delimiter=',', fmt='%.2f')
for _ in range(0,25):
    array = np.empty((384, 3))
    for i in range(0, 24):
        for j in range(0, 16):
            new_data = [ _ , i, j]
            array[i * 16 + j] = new_data
    y = array
    append_array_to_txt('lable.txt', y, delimiter=',', fmt='%.2f')

data = np.loadtxt('../Data_sets/data.txt', delimiter=',')
data = data.reshape(-1, 2, 2000)
ch2_x = data[768,0,:]
ch3_x = data[768,1,:]
y = np.linspace(0, 2000, 2000)
plt.plot(y, ch2_x, label='ch2', color='r')
plt.plot(y, ch3_x, label='ch3', color='b')
plt.legend()
plt.savefig('data.png')
def main():
    # print(len(file_list))
    # write_data_to_txt(file_list, txt_name, folder_path)
    # ckeck_peaks(file_list, folder_path)
    pass


if __name__ == '__main__':
    main()
