__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .PatchTST_backbone import PatchTST_backbone
from .PatchTST_layers import series_decomp


CONFIGS = {
    "enc_in": 2,
    "seq_len": 2000,
    "pred_len": 128,
    "e_layers": 4,
    "n_heads": 8,
    "d_model": 128,
    "d_ff": 256,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "head_dropout": 0.1,
    "mlp_dropout": 0.1,
    "mlp_dim": 128,
    "individual": False,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": 'end',
    "revin": True,
    "affine": False,
    "subtract_last": False,
    "decomposition": False,
    "kernel_size": 25,
}


class PatchTST(nn.Module):
    def __init__(self, num_classes_1, num_classes_2, configs, max_seq_len:Optional[int]=2048, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs['enc_in']
        context_window = configs['seq_len']
        target_window = configs['pred_len']
        
        n_layers = configs['e_layers']
        n_heads = configs['n_heads']
        d_model = configs['d_model']
        d_ff = configs['d_ff']
        dropout = configs['dropout']
        fc_dropout = configs['fc_dropout']
        head_dropout = configs['head_dropout']
        mlp_dropout = configs['mlp_dropout']
        mlp_dim = configs['mlp_dim']
        individual = configs['individual']
    
        patch_len = configs['patch_len']
        stride = configs['stride']
        padding_patch = configs['padding_patch']
        
        revin = configs['revin']
        affine = configs['affine']
        subtract_last = configs['subtract_last']
        
        decomposition = configs['decomposition']
        kernel_size = configs['kernel_size']
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        
        self.classification_head_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * target_window, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, num_classes_1)
        )

        self.classification_head_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * target_window, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, num_classes_2)
        )

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * target_window, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, 1)
        )
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        
        classification_output_1 = self.classification_head_1(x)
        classification_output_2 = self.classification_head_2(x)
        regression_output = self.regression_head(x).squeeze(-1)
        return classification_output_1, classification_output_2, regression_output
