# -*- coding: utf-8 -*-

#
# Transformer エンコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import numpy as np
#from my_dataset import SequenceDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        fe_conv_layer=7,
        fe_conv_channel=[512,512,512,512,512,512,512],
        fe_conv_kernel=[10,3,3,3,3,2,2],
        fe_conv_stride=[5,2,2,2,2,2,2],
        fe_conv_dropout_rate = 0.0,
        fe_out_dim = 768,
    ):
        super(FeatureExtractor, self).__init__()

        # 1 次元畳み込みの重ね合わせ：局所的な時間依存関係のモデル化
        convs = nn.ModuleList()
        
        #print( "0" )
        convs += [
            nn.Conv1d(
                1,
                fe_conv_channel[0],
                fe_conv_kernel[0],
                stride = fe_conv_stride[0],
                bias = False,
            ),
            nn.GroupNorm( 1, fe_conv_channel[0] ),
            #nn.BatchNorm1d( fe_conv_channel[0], eps=1e-5 ),
            #nn.GELU(approximate="tanh")
            nn.GELU()
            #nn.Tanh()
            #nn.ReLU()
        ]
        for layer in range( 1, fe_conv_layer - 2 ):
            #print( layer  )
            #print( " in_channels:{}".format( in_channels ))
            #print( " out_channels:{}".format( out_channels ))
            convs += [
                nn.Conv1d(
                    fe_conv_channel[layer-1],
                    fe_conv_channel[layer],
                    fe_conv_kernel[layer],
                    stride = fe_conv_stride[layer],
                    bias = False,
                ),
                nn.GroupNorm( 1, fe_conv_channel[layer] ),
                #nn.BatchNorm1d( fe_conv_channel[layer], eps=1e-5 ),
                #nn.GELU(approximate="tanh"),
                nn.GELU(),
                #nn.Tanh()
                #nn.ReLU()
            ]
        for layer in range( fe_conv_layer -2, fe_conv_layer):
            #print( layer )
            #print( " in_channels:{}".format( in_channels ))
            #print( " out_channels:{}".format( out_channels ))
            convs += [
                nn.Conv1d(
                    fe_conv_channel[layer-1],
                    fe_conv_channel[layer],
                    fe_conv_kernel[layer],
                    stride = fe_conv_stride[layer],
                    bias = False,
                ),
                nn.GroupNorm( 1, fe_conv_channel[layer] ),
                #nn.BatchNorm1d( fe_conv_channel[layer],eps=1e-5),
                #nn.GELU(approximate="tanh"),
                nn.GELU(),
                #nn.Tanh()
                #nn.ReLU()
            ]

        self.convs = nn.Sequential(*convs)
        self.ln = nn.LayerNorm( fe_out_dim,eps=1e-5,elementwise_affine=True )
        self.linear = nn.Linear( fe_conv_channel[layer], fe_out_dim, bias = True )
        self.dropout = nn.Dropout(p=fe_conv_dropout_rate)
        #self.ln2 = nn.LayerNorm( fe_out_dim,eps=1e-5,elementwise_affine=True )
        #self.gelu2 = nn.GELU(approximate="tanh")
        
        self.fe_conv_kernel = fe_conv_kernel
        self.fe_conv_stride = fe_conv_stride
        
    def forward(self, x, x_len ):

        # conv 層
        out = self.convs(x.transpose(1, 2)).transpose(1, 2)
        #print( "size of out:{}".format( out.size() ) )
        
        out_len = x_len
        for kernel, stride in zip( self.fe_conv_kernel, self.fe_conv_stride ):
            out_len = torch.round( ( out_len  - kernel ) / stride  + 1 ).long()

        y = self.ln( out )
        y = self.linear( y )
        out = self.dropout( y )
        #out = self.ln2( out )
        #out = self.gelu2( out )

        return out, out_len  # (batch_size, input_seq_len, d_model)
        
'''
def main():
    
    dataset = SequenceDataset("/home/uchiyats/non-ar-asr-wav/01compute_features/wav/train_large/feats.scp","/home/uchiyats/non-ar-asr-wav/09transformer_conv_non_ar_wav/exp_train_large/data/char/label_train_large")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=4)

    feats, label, feat_len, label_len, utt_id = next(iter(data_loader))

    print( feats )
    print( "size of feats:{}".format( feats.size() ))
    
    fe = FeatureExtractor()
    
    #feats1 = torch.unsqueeze( feats, dim = 2 )
    
    y, y_len = fe( feats, feat_len )
    

    print( y )
    print( feat_len )
    print( y_len )
    print( "size of y:{}".format( y.size() ) )


if __name__ == "__main__":
    main()
'''
