# -*- coding: utf-8 -*-

#
# Transformer デコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
from attention import ResidualAttentionBlock
#from dilated_transformer import DilatedTransformerDecoderLayer

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(
        self,
        dec_num_layers=6,
        dec_input_maxlen=300,
        decoder_hidden_dim=512,
        dec_num_heads = 4,
        dec_kernel_size = [5,1],
        dec_filter_size = 2048,
        dec_dropout_rate = 0.1,
        dec_dil_seg = [128,256,512],
        dec_dil_rate = [1,2,4],
        dec_seg_max = 512
    ):
        super().__init__()
        self.num_heads = dec_num_heads

        #  Attention  Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(decoder_hidden_dim, dec_num_heads, cross_attention=True, kernel_size = dec_kernel_size, filter_size = dec_filter_size ) for _ in range(dec_num_layers)]
            #[ResidualAttentionBlock(decoder_hidden_dim, dec_num_heads, cross_attention=False, kernel_size = dec_kernel_size, filter_size = dec_filter_size  ) for _ in range(dec_num_layers)]
        )
        #self.blocks: Iterable[DilatedTransformerDecoderLayer] = nn.ModuleList(
        #    [DilatedTransformerDecoderLayer(decoder_hidden_dim, dec_num_heads, segment_lengths = dec_dil_seg, dilation_rates = dec_dil_rate, dim_feedforward = dec_filter_size   ) for _ in range(dec_num_layers)]
        #)
        
        
        # position embedding
        self.pos_emb = nn.Embedding(dec_input_maxlen, decoder_hidden_dim)
        
        self.dec_input_maxlen = dec_input_maxlen
        self.dec_seg_max = dec_seg_max

    def forward(self, encoder_outs, decoder_targets=None):

        emb = decoder_targets
        # position embedding
        maxlen = emb.size()[1]
        positions = torch.arange(start=0, end=self.dec_input_maxlen, step=1, device=torch.device(device)).to(torch.long)
        positions = self.pos_emb(positions)[:maxlen,:]
        #print( "size of emb:", emb.size() )
        #print( "size of positions:", positions.size() )
        x = emb + positions

        #x2 = torch.zeros( ( x.size(0), self.dec_seg_max, x.size(2) ) ).to(device)
        #x2[:,:x.size(1),:] = x
        
        #encoder_outs2 = torch.zeros( ( encoder_outs.size(0), self.dec_seg_max, encoder_outs.size(2) ) ).to(device)
        #encoder_outs2[:,:encoder_outs.size(1),:] = encoder_outs
        
       
        #attention block
        
        #n_ctx = encoder_outs.size(1)
        #mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1).to(device)
        for i, block in enumerate( self.blocks ):
            x = block(x, encoder_outs, mask=None)
            #x2= block(x2, encoder_outs2, mask=None)
            #x2= block(x2, encoder_outs2)
        
        return x

