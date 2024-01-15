# -*- coding: utf-8 -*-

#
# モデル構造を定義します
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import gc
import numpy as np
import random
#import time

# 作成したEncoder, Decoderクラスをインポート
from encoder import Encoder
from decoder import Decoder
from feature_extractor import FeatureExtractor
from quantize import Quantize


# 作成した初期化関数をインポート
from initialize import lecun_initialization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyE2EModel(nn.Module):
    ''' Attention RNN によるEnd-to-Endモデルの定義
    dim_in:             入力次元数
    dim_out:            num_tokens　語彙数
    conv_layers:        エンコーダーの conv 層の数
    conv_channels:      エンコーダーの conv 層のチャンネル数
    conv_dropout_rate:  エンコーダーの covn 層の dropout_rate
    enc_num_layers:     エンコーダー層数
    enc_att_hidden_dim: エンコーダーのアテンションの隠れ層数
    enc_num_heads:      エンコーダーのhead数
    enc_input_maxlen:   エンコーダーの入力の時間数の最大フレーム値 3000
    enc_att_kernel_size:エンコーダートランスフォーマーのカーネルサイズ
    enc_att_filter_size:エンコーダートランスフォーマーのフィルター数
    enc_dropout_rage:   エンコーダーのドロップアウト
    dec_num_layers:     デコーダー層数
    dec_att_hidden_dim: デコーダーのアテンションの隠れ層数
    dec_num_heads:      デコーダーのhead数
    dec_input_maxlen:   デコーダーの入力の時間数の最大フレーム値 300
    dec_att_kernel_size:デコーダートランスフォーマーのカーネルサイズ
    dec_att_filter_size:デコーダートランスフォーマーのフィルター数
    dec_dropout_rage   :デコーダーのドロップアウト
    sos_id:             token の <sos> の番号
    '''
    def __init__(self, dim_in, dim_out,
                 fe_conv_layer, fe_conv_channel, fe_conv_kernel, fe_conv_stride, fe_conv_dropout_rate, fe_out_dim,
                 conv_layers, conv_channels, conv_kernel_size, conv_dropout_rate,
                 enc_num_layers, enc_att_hidden_dim, enc_num_heads, enc_input_maxlen,  enc_att_kernel_size, enc_att_filter_size, enc_dropout_rate,
                 enc_dil_seg, enc_dil_rate, enc_seg_max,
                 ds_rate,n_mask,n_consec, entryV, num_codebook, tau, temprature_multi,tau_min,
                 dec_num_layers, dec_att_hidden_dim, dec_num_heads, dec_target_maxlen, dec_att_kernel_size, dec_att_filter_size, dec_dropout_rate,
                 dec_dil_seg, dec_dil_rate, dec_seg_max,
                 sos_id,
                 ):
        super(MyE2EModel, self).__init__()

        self.fe = FeatureExtractor(
            fe_conv_layer=fe_conv_layer,
            fe_conv_channel=fe_conv_channel,
            fe_conv_kernel=fe_conv_kernel,
            fe_conv_stride=fe_conv_stride,
            fe_conv_dropout_rate = fe_conv_dropout_rate,
            fe_out_dim=fe_out_dim,
        )
        self.quantize = Quantize(
            hidden_dim = enc_att_hidden_dim,
            entryV = entryV,
            num_codebook = num_codebook,
            tau_min = tau_min,
        )
        #self.quantize_linear = nn.Linear( enc_att_hidden_dim, enc_att_hidden_dim )
        
        # エンコーダを作成
        self.encoder = Encoder(
            conv_layers=conv_layers,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
            conv_dropout_rate=conv_dropout_rate,
            embed_dim = dim_in,
            num_enc_layers = enc_num_layers,
            enc_hidden_dim = enc_att_hidden_dim,
            enc_num_heads = enc_num_heads,
            enc_input_maxlen = enc_input_maxlen,
            enc_kernel_size = enc_att_kernel_size,
            enc_filter_size = enc_att_filter_size,
            enc_dropout_rate = enc_dropout_rate,
            enc_dil_seg = enc_dil_seg,
            enc_dil_rate = enc_dil_rate,
            enc_seg_max = enc_seg_max,
        )
        
        # デコーダを作成
        self.decoder = Decoder(
            dec_num_layers = dec_num_layers,
            dec_input_maxlen = dec_target_maxlen,
            decoder_hidden_dim = dec_att_hidden_dim,
            dec_num_heads = dec_num_heads,
            dec_kernel_size = dec_att_kernel_size,
            dec_filter_size = dec_att_filter_size,
            dec_dropout_rate = dec_dropout_rate,
            dec_dil_seg = dec_dil_seg,
            dec_dil_rate = dec_dil_rate,
            dec_seg_max = dec_seg_max,
        )

        #　デコーダーのあとに、n * t * hidden を n * t * num_vocab にする線形層。
        #self.classifier = nn.Linear( dec_att_hidden_dim, dim_out, bias=False )
        self.classifier = nn.Linear( dec_att_hidden_dim, dec_att_hidden_dim )
        self.gelu = nn.GELU()
        #self.gelu = nn.GELU(approximate="tanh")
        self.ln = nn.LayerNorm( dec_att_hidden_dim )
        
        self.dec_target_maxlen = dec_target_maxlen
        self.sos_id = sos_id
        self.ds_rate = ds_rate
        self.n_mask = n_mask
        self.mask_time_prob = n_mask
        self.n_consec = n_consec
        self.num_mask_time_steps = n_consec

        # LeCunのパラメータ初期化を実行
        #lecun_initialization(self)

    def time_masking(self, hidden_states0: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            Masked hidden states (torch.Tensor with shape `(B, L, D)`),
            Time mask (torch.BoolTensor with `(B, L)`)
            )
        """

        hidden_states = hidden_states0.clone()

        batch_size, num_steps, hidden_size = hidden_states.size()

        # non mask: 0, mask: 1
        time_mask_indices = torch.zeros(
            batch_size, num_steps + self.num_mask_time_steps,
            device=hidden_states.device, dtype=torch.bool
        )

        for batch in range(batch_size):
            time_mask_idx_candidates = list(range(int(lengths[batch])))
            k = int(self.mask_time_prob * lengths[batch])
            start_time_idx_array = torch.tensor(
                random.sample(time_mask_idx_candidates, k=k), device=hidden_states.device
            )

            for i in range(self.num_mask_time_steps):
                time_mask_indices[batch, start_time_idx_array+i] = 1

        time_mask_indices = time_mask_indices[:, :-self.num_mask_time_steps]
        num_masks = sum(time_mask_indices.flatten())

        # Maks hidden states
        mask_values = torch.zeros(num_masks, hidden_size, device=hidden_states.device)
        hidden_states[time_mask_indices] = mask_values

        return hidden_states, time_mask_indices


    #@profile
    def forward(self,
                input_sequence,
                input_lengths,
                tau):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        '''
        input_sequence2, input_lengths2 = self.fe( input_sequence, input_lengths )
        
        #input_sequence3 = input_sequence2.clone()
        #input_sequence3 = input_sequence2
        #mask = torch.zeros( (input_sequence3.size(0), input_sequence3.size(1)), device=torch.device(device) ).to(torch.bool)

        #for n in range( input_sequence2.size(0) ):
        #    n_mask = torch.round( ( input_lengths2[n] * self.n_mask ) ).to( torch.int32 )
        #    mask_pos = torch.randint( 0, input_lengths2[n] - self.n_consec - 1, ( n_mask, ) )
        #    for m2 in mask_pos:
        #        for m3 in range( m2, m2 + self.n_consec ):
        #            #if m3 < input_sequence3.size(1):
        #            input_sequence3[n,m3] = 0 # 論文の trained mask feature vector を input_sequence3[n,m3] = [0,0,・・・,0] としている。
        #            mask[n,m3] = True
                    
        input_sequence3, mask = self.time_masking( input_sequence2, input_lengths2 )
        #print( "input_sequence3:", input_sequence3 )
        #print( "maks:", mask )
       
        qv, pgv_bar = self.quantize( input_sequence2, tau )
        # エンコーダに入力する
        enc_out = self.encoder(input_sequence3,input_lengths2)
        enc_lengths = input_lengths2
        
        #注意、quantize_vector もダウンサンプリングしなければならない
        dec_input, outputs_lens, mask, qv = self.downsample( enc_out, input_lengths2, mask, qv )
        
        # デコーダに入力する
        dec_out = self.decoder(enc_out, dec_input)
        dec_out = self.gelu( dec_out )
        dec_out = self.ln( dec_out )

        # n * T * hidden → n * T * num_vocab 
        outputs = self.classifier( dec_out )


        # デコーダ出力とエンコーダ出力系列長を出力する
        #return dec_out, outputs_lens, qv1, qv2, mask, qv
        #return outputs, outputs_lens, p1_bar, p2_bar, mask, qv
        return outputs, outputs_lens, pgv_bar, mask, qv

    def downsample(self, enc_out, input_lengths, mask, qv ):
        
        max_label_length = int( round( enc_out.size(1) * self.ds_rate ) )
        
        polated_lengths = torch.round( torch.ones( (enc_out.size(0)), device=torch.device(device) ) * enc_out.size(1) * self.ds_rate ).long()

        outputs_lens = torch.ceil( input_lengths * self.ds_rate ).long()
        

        x = enc_out
        out_lens = polated_lengths
        #x2 = qv1.to(device)
        #x4 = qv2.to(device)
        x6 = torch.unsqueeze( mask, dim = 2 ).to( torch.float32 )
        x8 = qv

        for i in range( x.size(0) ):
            x0 = torch.unsqueeze( x[i], dim = 0 )
            #x3 = torch.unsqueeze( x2[i], dim = 0 )
            #x5 = torch.unsqueeze( x4[i], dim = 0 )
            x7 = torch.unsqueeze( x6[i], dim = 0 )
            x9 = torch.unsqueeze( x8[i], dim = 0 )
            x0 = x0.permute( 0,2,1 )
            #x3 = x3.permute( 0,2,1 )
            #x5 = x5.permute( 0,2,1 )
            x7 = x7.permute( 0,2,1 )
            x9 = x9.permute( 0,2,1 )
            x_out = torch.nn.functional.interpolate(x0, size = (out_lens[i]), mode='nearest-exact')
            #print( "x_out device in downsample", x_out.get_device() )
            #x_out2 = torch.nn.functional.interpolate(x3, size = (out_lens[i]), mode='nearest-exact')
            #x_out4 = torch.nn.functional.interpolate(x5, size = (out_lens[i]), mode='nearest-exact')
            x_out6 = torch.nn.functional.interpolate(x7, size = (out_lens[i]), mode='nearest-exact')
            x_out8 = torch.nn.functional.interpolate(x9, size = (out_lens[i]), mode='nearest-exact')
            #print( "size of x0:{}".format( x0.size() ))
            #print( "size of x_out:{}".format( x_out.size() ))
            z = torch.zeros( (x_out.size(0), x_out.size(1), max_label_length), device=torch.device(device) )
            #z2 = torch.zeros( (x_out2.size(0), x_out2.size(1), max_label_length), device=torch.device(device) )
            #z4 = torch.zeros( (x_out4.size(0), x_out4.size(1), max_label_length), device=torch.device(device) )
            z6 = torch.zeros( (x_out6.size(0), x_out6.size(1), max_label_length), device=torch.device(device) )
            z8 = torch.zeros( (x_out8.size(0), x_out8.size(1), max_label_length), device=torch.device(device) )
            #print( " size of z2:{}".format(z2.size()))
            #print( " size of x_out2:{}".format(x_out2.size()))
            if z.size(2) > x_out.size(2):
            	z[:,:,:x_out.size(2)] = x_out[:,:,:]
            else:
                z[:,:,:] = x_out[:,:,:z.size(2)]
            #if z2.size(2) > x_out2.size(2):
            #    z2[:,:,:x_out2.size(2)] = x_out2[:,:,:]
            #else:
            #    z2[:,:,:] = x_out2[:,:,:z2.size(2)]
            #if z4.size(2) > x_out4.size(2):
            #    z4[:,:,:x_out4.size(2)] = x_out4[:,:,:]
            #else:
            #    z4[:,:,:] = x_out4[:,:,:z4.size(2)]
            if z6.size(2) > x_out6.size(2):
            	z6[:,:,:x_out6.size(2)] = x_out6[:,:,:]
            else:
                z6[:,:,:] = x_out6[:,:,:z6.size(2)]
            if z8.size(2) > x_out8.size(2):
            	z8[:,:,:x_out8.size(2)] = x_out8[:,:,:]
            else:
                z8[:,:,:] = x_out8[:,:,:z8.size(2)]
            #z[:,:,:max_label_length] = x_out[:,:,:]
            x_out = z.permute( 0, 2, 1 )
            #x_out2 = z2.permute( 0,2,1)
            #x_out4 = z4.permute( 0,2,1)
            x_out6 = z6.permute( 0,2,1)
            x_out8 = z8.permute( 0,2,1)
            if i == 0:
                y = x_out
                #print( "y device in downsample", y.get_device() )
                #y2 = x_out2
                #y4 = x_out4
                y6 = x_out6
                y8 = x_out8
            if i > 0:
                y = torch.cat( (y, x_out), dim = 0 )
                #y2 = torch.cat( (y2, x_out2), dim = 0 )
                #y4 = torch.cat( (y4, x_out4), dim = 0 )
                y6 = torch.cat( (y6, x_out6), dim = 0 )
                y8 = torch.cat( (y8, x_out8), dim = 0 )

        
        y6 = torch.squeeze( y6, dim = 2 ).to( torch.bool )

        #return y, outputs_lens, y2, y4, y6, y8
        return y, outputs_lens, y6, y8
