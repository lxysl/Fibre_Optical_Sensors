import numpy as np
import torch
from sklearn.neighbors import KernelDensity


def get_mixup_sample_rate(mixtype, kde_type, kde_bandwidth, y, use_kde=True, show_process=False):
    mix_idx = []
    assert isinstance(y, np.ndarray) or isinstance(y, torch.Tensor), "y must be a numpy array or a torch tensor"
    data = torch.tensor(y, dtype=torch.float32) if isinstance(y, np.ndarray) else y
    data = data.reshape(-1, 1)

    N = len(data)

    ######## use kde rate or uniform rate #######
    for i in range(N):
        if mixtype == 'kde' or use_kde: # kde
            data_i = data[i]
            data_i = data_i.reshape(-1, 1) # get 2D
            
            if show_process:
                if i % (N // 10) == 0:
                    print('Mixup sample prepare {:.2f}%'.format(i * 100.0 / N ))
                # if i == 0: print(f'data.shape = {data.shape}, std(data) = {torch.std(data)}')#, data_i = {data_i}' + f'data_i.shape = {data_i.shape}')
                
            ######### get kde sample rate ##########
            kd = KernelDensity(kernel=kde_type, bandwidth=kde_bandwidth).fit(data_i)  # should be 2D
            each_rate = np.exp(kd.score_samples(data))
            each_rate /= np.sum(each_rate)  # norm
        else:
            each_rate = np.ones(y.shape[0]) * 1.0 / y.shape[0]
        
        ####### visualization: observe relative rate distribution shot #######
        # if show_process and i == 0:
        #         print(f'bw = {kde_bandwidth}')
        #         print(f'each_rate[:10] = {each_rate[:10]}')
        #         stats_values(each_rate)
            
        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]
    if show_process:
        print(f'len(y) = {len(y)}, len(mix_idx) = {len(mix_idx)}, np.mean(self_rate) = {np.mean(self_rate)}, np.std(self_rate) = {np.std(self_rate)},  np.min(self_rate) = {np.min(self_rate)}, np.max(self_rate) = {np.max(self_rate)}')

    return mix_idx
