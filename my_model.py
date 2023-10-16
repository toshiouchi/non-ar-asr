# -*- coding: utf-8 -*-

#
# モデル構造を定義します
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn

# 作成したEncoder, Decoderクラスをインポート
from encoder import Encoder
from decoder import Decoder

# 作成した初期化関数をインポート
from initialize import lecun_initialization

import numpy as np

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
                 conv_layers, conv_channels, conv_kernel_size, conv_dropout_rate,
                 enc_num_layers, enc_att_hidden_dim, enc_num_heads, enc_input_maxlen,  enc_att_kernel_size, enc_att_filter_size, enc_dropout_rate,
                 ds_rate,
                 dec_num_layers, dec_att_hidden_dim, dec_num_heads, dec_target_maxlen, dec_att_kernel_size, dec_att_filter_size, dec_dropout_rate,
                 sos_id,
                 ):
        super(MyE2EModel, self).__init__()

        # エンコーダを作成
        self.encoder = Encoder(
            embed_dim = dim_in,
            conv_layers = conv_layers,
            conv_channels = conv_channels,
            conv_kernel_size = conv_kernel_size,
            conv_dropout_rate = conv_dropout_rate,
            num_enc_layers = enc_num_layers,
            enc_hidden_dim = enc_att_hidden_dim,
            enc_num_heads = enc_num_heads,
            enc_input_maxlen = enc_input_maxlen,
            enc_kernel_size = enc_att_kernel_size,
            enc_filter_size = enc_att_filter_size,
            enc_dropout_rate = enc_dropout_rate,
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
            dim_out = dim_out
        )

        #　デコーダーのあとに、n * t * hidden を n * t * num_vocab にする線形層。
        self.classifier = nn.Linear( dec_att_hidden_dim, dim_out, bias=False )
        
        self.dec_target_maxlen = dec_target_maxlen
        self.sos_id = sos_id
        self.ds_rate = ds_rate

        # LeCunのパラメータ初期化を実行
        #lecun_initialization(self)

    def forward(self,
                input_sequence,
                input_lengths):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        '''
        # エンコーダに入力する
        enc_out, attn_ws_enc = self.encoder(input_sequence,input_lengths)
        #print( "enc_out.size:{}".format( enc_out.size() ))
        enc_lengths = input_lengths
        
        dec_input, outputs_lens = self.downsample( enc_out, input_lengths )
        
        # デコーダに入力する
        dec_out, attn_ws_dec = self.decoder(enc_out, dec_input)

        # n * T * hidden → n * T * num_vocab 
        outputs = self.classifier( dec_out )

        # デコーダ出力とエンコーダ出力系列長を出力する
        return outputs, outputs_lens

    def save_att_matrix(self, utt, filename):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        # decoderのsave_att_matrixを実行
        self.decoder.save_att_matrix(utt, filename)
        
    # 推論モジュール    
    def inference(self,
                input_sequence,
                input_lengths):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        '''
        # エンコーダに入力する
        enc_out, attn_ws_enc = self.encoder(input_sequence, input_lengths)
        #print( "enc_out.size:{}".format( enc_out.size() ))
        enc_lengths = input_lengths
        
        #print( "labes:{}".format( labels[0] ))
        dec_input, outputs_lens = self.downsample( enc_out, input_lengths )
        
        # デコーダに入力する
        dec_out, attn_ws_dec = self.decoder(enc_out, dec_input)
        outputs = self.classifier( dec_out )
        #outputs = F.log_softmax( outpus, dim = 2 ) 
        #logits = torch.argmax( outputs, dim = 2 ).long()

        # デコーダ出力とエンコーダ出力系列長を出力する
        #print("dec_input:{}".format( dec_input ))
        #return logits
        return outputs
        
    def downsample(self, enc_out, input_lengths ):
        
        max_label_length = int( round( enc_out.size(1) * self.ds_rate ) )
        
        polated_lengths = torch.round( torch.ones( enc_out.size(0) ) * enc_out.size(1) * self.ds_rate ).long()
        #print("size 0f polated_lengths:{}".format( polated_lengths.size() ))
        #polated_lengths = torch.tensor( polated_lengths, requires_grad = False )

        outputs_lens = torch.ceil( input_lengths * self.ds_rate ).long()
        
        #print( "input_lengths:{}".format(input_lengths ))
        #print( "outputs_lens:{}".format( outputs_lens ))

        x = enc_out
        out_lens = polated_lengths

        for i in range( x.size(0) ):
            x0 = torch.unsqueeze( x[i], dim = 0 )
            x0 = x0.permute( 0,2,1)
            x_out = torch.nn.functional.interpolate(x0, size = (out_lens[i]), mode='nearest-exact')
            #print( "size of x0:{}".format( x0.size() ))
            #print( "size of x_out:{}".format( x_out.size() ))
            z = torch.zeros( x_out.size(0), x_out.size(1), max_label_length )
            #print( " size of z:{}".format(z[:,:,:x_out.size(2)].size()))
            #print( " size of x_out:{}".format(x_out[:,:,:].size()))
            if z.size(2) > x_out.size(2):
            	z[:,:,:x_out.size(2)] = x_out[:,:,:]
            else:
                z[:,:,:] = x_out[:,:,:z.size(2)]
            #z[:,:,:max_label_length] = x_out[:,:,:]
            x_out = z.permute( 0, 2, 1 )
            if i == 0:
                y = x_out
            if i > 0:
                y = torch.cat( (y, x_out), dim = 0 )
        
        return y, outputs_lens
