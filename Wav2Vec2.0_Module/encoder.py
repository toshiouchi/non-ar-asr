# -*- coding: utf-8 -*-

#
# Transformer エンコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from attention import ResidualAttentionBlock
#from dilated_transformer import DilatedTransformerEncoderLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalConvEmbbeding(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, groups):
        super().__init__()
        self.conv = nn.Conv1d( in_dim, out_dim, kernel_size, stride = 1, padding = padding, groups = groups)

    def forward( self, x ):
        x = self.conv( x.transpose(1,2))
        x = F.gelu(x[:,:,:-1])
        return x.transpose(1,2)

class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim=80,
        conv_layers=3,
        conv_channels=512,
        conv_kernel_size=5,
        conv_dropout_rate = 0.1,
        enc_hidden_dim = 512,
        num_enc_layers = 6,
        enc_num_heads = 4,
        enc_kernel_size = [5,1],
        enc_filter_size = 2048,
        enc_input_maxlen = 3000,
        enc_dropout_rate = 0.1,
        enc_dil_seg = [256,512,1024],
        enc_dil_rate = [1,2,4],
        enc_seg_max = 1024
    ):
        super(Encoder, self).__init__()
        '''
        self.pos_emb = nn.Embedding(enc_input_maxlen, enc_hidden_dim)
        # 1 次元畳み込みの重ね合わせ：局所的な時間依存関係のモデル化
        convs = nn.ModuleList()
        for layer in range(conv_layers):
            in_channels = embed_dim if layer == 0 else conv_channels
            out_channels = enc_hidden_dim if layer == conv_layers - 1 else conv_channels
            print( " in_channels:{}".format( in_channels ))
            print( " out_channels:{}".format( out_channels ))
            convs += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2,
                    bias=False,  # この bias は不要です
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(conv_dropout_rate),
            ]
        self.convs = nn.Sequential(*convs)
        '''
        self.positional_embedding = PositionalConvEmbbeding( \
                 in_dim = enc_hidden_dim, out_dim = enc_hidden_dim, \
                 kernel_size = 128, padding = 128 // 2 , groups = 16 )
        
        # Attention Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(enc_hidden_dim, enc_num_heads, cross_attention = False, kernel_size = enc_kernel_size, filter_size = enc_filter_size ) for _ in range(num_enc_layers)]
        )
        #self.blocks: Iterable[DilatedTransformerEncoderLayer] = nn.ModuleList(
        #    [DilatedTransformerEncoderLayer(enc_hidden_dim, enc_num_heads, segment_lengths = enc_dil_seg, dilation_rates = enc_dil_rate, dim_feedforward = enc_filter_size   ) for _ in range(num_enc_layers)]
        #)

        #self.ln1 = nn.BatchNorm1d( conv_channels )
        self.ln1 = nn.LayerNorm( conv_channels )
        
        self.input_maxlen = enc_input_maxlen
        self.enc_seg_max = enc_seg_max
        
    def forward(self, x, in_lens ):

        out = x
        
        # positional embbeding
        x = out + self.positional_embedding( out )
        x = self.ln1( x )

        #x2 = torch.zeros( ( x.size(0), self.enc_seg_max, x.size(2) ) ).to(device)
        #x2[:,:x.size(1),:] = x
        
        # attention block
        for i, block in enumerate( self.blocks ):
            x = block(x, x, mask = None)
            #x2 = block(x2, x2, mask = None)
            #x2 = block(x2)
       
        return x  # (batch_size, input_seq_len, d_model)
