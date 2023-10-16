# -*- coding: utf-8 -*-

#
# Transformer デコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
from attention import ResidualAttentionBlock

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
        dim_out = 2680
    ):
        super().__init__()
        self.num_heads = dec_num_heads

        #  Attention  Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(decoder_hidden_dim, dec_num_heads, cross_attention=True, kernel_size = dec_kernel_size, filter_size = dec_filter_size  ) for _ in range(dec_num_layers)]
            #[ResidualAttentionBlock(decoder_hidden_dim, dec_num_heads, cross_attention=False, kernel_size = dec_kernel_size, filter_size = dec_filter_size  ) for _ in range(dec_num_layers)]
        )
        
        #self.emb = nn.Embedding(dim_out,decoder_hidden_dim)
        
        # position embedding
        self.pos_emb = nn.Embedding(dec_input_maxlen, decoder_hidden_dim)
        
        self.dec_input_maxlen = dec_input_maxlen
        self.att3 = None

    def forward(self, encoder_outs, decoder_targets=None):

        # decoder_targets のベクトル化（埋め込み）
        #prenet_out = self.emb( decoder_targets )
        #emb = self.emb( decoder_targets )
        emb = decoder_targets
        # position embedding
        #maxlen = prenet_out.size()[1]
        maxlen = emb.size()[1]
        positions = torch.arange(start=0, end=self.dec_input_maxlen, step=1).to(torch.long)
        positions = self.pos_emb(positions.to(device))[:maxlen,:]
        #print( "size of emb:{}".format( emb.size() ) )
        #print( "size of posiions:{}".format( positions.size() ) )
        #x = prenet_out.to( device ) + positions.to(device)
        x = emb.to(device) + positions.to(device )
        
        #attention block
        attention_weights = []
        attention_weights2 = []
        
        n_ctx = encoder_outs.size(1)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1).to(device)
        for i, block in enumerate( self.blocks ):
            x, attn1, attn2 = block(x, encoder_outs, mask=mask)
            #x, attn1, attn2 = block(x, x, mask=mask)
            #attention_weights["{}".format( 2*i + 1 )] = attn1
            #attention_weights["{}".format( 2*i + 2 )] = attn2
            attention_weights.append( attn1 )
            attention_weights.append( attn2 )
            attention_weights2.append( attn2)
        
        attention_weights2 = torch.stack( attention_weights2  )
        # attention の　hidden_dim から num_vocab へ
        #outs = self.feat_out(x)
        #outs = torch.permute(outs, (0, 2, 1))
        
        for j in range( 0, i ):
            #print( "shape of attention_weights:{}".format( attention_weights2.size() ))
            self.att3 = torch.sum( attention_weights2[j], dim = 1 )
        
        return x, attention_weights
                
    def save_att_matrix(self, utt, filename):
        '''
        Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        att_mat = self.att3[utt].cpu().detach().numpy()
        #print( "shape of att_mat:{}".format( att_mat.shape ))

        # プロットの描画領域を作成
        plt.figure(figsize=(5,5))
        # カラーマップのレンジを調整
        att_mat -= np.max(att_mat)
        vmax = np.abs(np.min(att_mat)) * 0.0
        vmin = - np.abs(np.min(att_mat)) * 0.7
        # プロット
        plt.imshow(att_mat, 
                   cmap = 'gray',
                   vmax = vmax,
                   vmin = vmin,
                   aspect = 'auto')
        # 横軸と縦軸のラベルを定義
        plt.xlabel('Decoder index')
        plt.ylabel('Decoder index')

        # プロットを保存する
        plt.savefig(filename)
        plt.close()

