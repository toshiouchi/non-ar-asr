# -*- coding: utf-8 -*-

#
# RNN Attention Encoder-Decoderモデルを学習します．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
#import pytorch_warmup as warmup


# 作成したDatasetクラスをインポート
from my_dataset_pre import SequenceDataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

# 認識エラー率を計算するモジュールをインポート
import levenshtein

# モデルの定義をインポート
from my_model import MyE2EModel

# json形式の入出力を行うモジュールをインポート
import json

# os, sys モジュールをインポート
import os
import sys
import gc
import psutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def ctc_simple_decode(int_vector, token_list):
    ''' 以下の手順で，フレーム単位のCTC出力をトークン列に変換する
        1. 同じ文字が連続して出現する場合は削除
        2. blank を削除
    int_vector: フレーム単位のCTC出力(整数値列)
    token_list: トークンリスト
    output:     トークン列
    '''
    # 出力文字列
    output = []
    # 一つ前フレームの文字番号
    prev_token = -1
    # フレーム毎の出力文字系列を前から順番にチェックしていく
    for n in int_vector:
        #print( " n:{}".format( n ))
        #print( " prev_token:{}".format( prev_token ))
        if n != prev_token:
            # 1. 前フレームと同じトークンではない
            if n != 0:
                # 2. かつ，blank(番号=0)ではない
                # --> token_listから対応する文字を抽出し，
                #     出力文字列に加える
                output.append(token_list[n])
                if token_list[n] == '<eos>':
                    break
            # 前フレームのトークンを更新
            prev_token = n
    return output

def ld_loss(pgv_bar):
                
    ld = torch.mean( torch.mean( ( pgv_bar * torch.log( pgv_bar + 1e-10 )), dim = 0 ), dim = 0 )
    #print( "ld:", ld )

    return ld

def cosine_similarity(a, b):
    a_normalized = torch.nn.functional.normalize(a, p=2.0, dim=1, eps=1e-12, out=None)
    b_normalized = torch.nn.functional.normalize(b, p=2.0, dim=1, eps=1e-12, out=None)
    cosine_sim = a_normalized * b_normalized
    #print( "size of cosine_sim:", cosine_sim.size() )
    return torch.sum(cosine_sim, dim=1)


def cosine_similarity_matmul(a,b):
    a_normalized = torch.nn.functional.normalize(a, p=2.0, dim=1, eps=1e-12, out=None)
    b_normalized = torch.nn.functional.normalize(b, p=2.0, dim=1, eps=1e-12, out=None)
    b_normalized = torch.permute( b_normalized,( 1,0 ) )

    cosine_similarities = torch.matmul( a_normalized, b_normalized )
    return cosine_similarities


def lm_loss( outputs, quantized_vector, mask_n_len ):

    kappar = 0.1
    #K = 100
    pos_sim_avg_sum = 0
    neg_sim_sum_avg_avg_sum = 0
    lm_n_sum = 0
    for n in range( quantized_vector.size(0) ):
       
        #for t in range( mask_n_len[n]):
        #    while True:
        #        neg_similarity_mask0 = torch.randint( 0, mask_n_len[n], ( K + 1, ), device=torch.device(device))
        #        if t in neg_similarity_mask0:
        #            for t2 in range( mask_n_len[n]):
        #                if t2 in neg_similarity_mask0:
        #                    neg_similarity_mask[t][t2] = 1.0
        #            break

        #print( "neg_similarity_mask:", neg_similarity_mask )

        pos_sim = torch.zeros( ( mask_n_len[n] ), device=torch.device(device) )
        pos_sim[:] = F.cosine_similarity( outputs[n,:mask_n_len[n]], quantized_vector[n,:mask_n_len[n]], dim = 1, eps=1e-6)
        #pos_sim = cosine_similarity(outputs[n,:mask_n_len[n]], quantized_vector[n,:mask_n_len[n]])
        pos_sim_avg = torch.mean( pos_sim, dim = 0 )
        pos_sim_avg_sum += pos_sim_avg.item()

        neg_sim = cosine_similarity_matmul( outputs[n,:mask_n_len[n],:], quantized_vector[n,:mask_n_len[n],:] )
        
        a = torch.ones( ( mask_n_len[n] ), device=torch.device(device) )
        neg_sim_mask = torch.diag( a )

        only_neg_sim = neg_sim * ( 1 - neg_sim_mask )
        neg_sim_avg_avg = torch.mean( (torch.sum( only_neg_sim, dim = 1 ) / (mask_n_len[n] - 1 )), dim = 0 )
        neg_sim_sum_avg_avg_sum += neg_sim_avg_avg.item()

        neg_similarity = torch.exp( neg_sim / kappar )
        #neg_similarity = torch.exp( only_neg_sim / kappar )

        numerator = torch.exp( pos_sim / kappar )
        denominator = torch.sum( neg_similarity, dim = 1 )
        lm = torch.zeros( ( mask_n_len[n] ), device=torch.device(device) )
        #lm[:] = (-1.0) * pos_sim[:] / kappar + torch.log( neg_similarity_sum[:] + 1e-9 )
        lm[:] = - torch.log( numerator / ( denominator + 1e-9 ) )
        lm_n_sum += torch.mean( lm, dim = 0 )

    lm_avg = lm_n_sum / outputs.size(0)
    
    return lm_avg, pos_sim_avg_sum, neg_sim_sum_avg_avg_sum


def masking( outputs, quantized_vector, mask ):

    mask_n_len = torch.sum( mask.to( torch.int32 ), dim = 1 )
    outputs2 = torch.zeros( ( outputs.size(0), torch.max( mask_n_len ), outputs.size(2) ), device=torch.device(device) )
    quantized_vector2 = torch.zeros( ( quantized_vector.size(0), torch.max( mask_n_len ), outputs.size(2) ), device=torch.device(device) )
    for n in range( mask.size(0) ):
       t2 = 0
       for t in range( mask.size(1) ):
           if mask[n,t] == True:
               outputs2[n,t2,:] = outputs[n,t,:]
               quantized_vector2[n,t2,:] = quantized_vector[n,t,:]
               t2 += 1

    return outputs2, quantized_vector2, mask_n_len

#@profile
def main():

    #torch.autograd.set_detect_anomaly(True)
    
    #
    # 設定ここから
    #

    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit = 'char'

    # 学習データの特徴量(feats.scp)が存在するディレクトリ
    #feat_dir_train = '../01compute_features/fbank/train_large'
    feat_dir_train = '../01compute_features/wav/train_large'
    # 開発データの特徴量(Feats.scp)が存在するディレクトリ
    #feat_dir_dev = '../01compute_features/fbank/dev'
    feat_dir_dev = '../01compute_features/wav/dev'


    # 実験ディレクトリ
    # train_set_name = 'train_small' or 'train_large'
    train_set_name = os.path.basename(feat_dir_train) 
    exp_dir = './exp_' + os.path.basename(feat_dir_train) 

    # 学習/開発データの特徴量リストファイル
    feat_scp_train = os.path.join(feat_dir_train, 'feats0.scp')
    feat_scp_dev = os.path.join(feat_dir_dev, 'feats0.scp')
    #feat_scp_train = os.path.join(feat_dir_train, 'feats.scp')
    #feat_scp_dev = os.path.join(feat_dir_dev, 'feats.scp')


    # 学習/開発データのラベルファイル
    label_train = os.path.join(exp_dir, 'data', unit,
                               'label_'+train_set_name+'0')
    label_dev = os.path.join(exp_dir, 'data', unit,
                             'label_dev0')
    #label_train = os.path.join(exp_dir, 'data', unit,
    #                           'label_'+train_set_name)
    #label_dev = os.path.join(exp_dir, 'data', unit,
    #                         'label_dev')
    
    # 訓練データから計算された特徴量の平均/標準偏差ファイル
    mean_std_file = os.path.join(feat_dir_train, 'mean_std.txt')

    # トークンリスト
    token_list_path = os.path.join(exp_dir, 'data', unit,
                                   'token_list')
    # 学習結果を出力するディレクトリ
    output_dir = os.path.join(exp_dir, unit+'_model_wav2vec2.0_060')

    # ミニバッチに含める発話数
    #batch_size = 10
    batch_size = 8
    #batch_size = 2

    # 最大エポック数
    #max_num_epoch = 60
    max_num_epoch = 100
    #max_num_epoch = 20
    #max_num_epoch = 1
    
    # feature_extractor の conv 層の設定
    fe_conv_layer = 7
    fe_conv_channel = [512,512,512,512,512,512,512]
    fe_conv_kernel = [10,3,3,3,3,2,2]
    fe_conv_stride = [5,2,2,2,2,2,2]
    fe_conv_dropout = 0.1
    fe_out_dim = 512

    # Encoder の Attention に入力するための conv 層の設定
    conv_layers = 3
    conv_channels = 512
    conv_kernel_size = 5
    conv_dropout_rate = 0.1


    # Encoderの設定
    # レイヤー数
    enc_num_layers = 6
    # encoder の head の数
    enc_num_heads = 8
    # Encoder の Attention block の次元数
    enc_att_hidden_dim = 512
    # encoder 入力の時間の最大数
    enc_input_maxlen = 3000
    # Encoder の Attention Bolock の kernel_size
    enc_att_kernel_size = [5,1]
    # Encoder の Attention Block の filter_size
    enc_att_filter_size = 2048
    # Encoder の dropout
    enc_dropout = 0.1
    # Encoder の dilated transformer の segment_length
    enc_dil_seg = [256,512,1024]
    # Encoder の dilated transformer の dilation rate
    enc_dil_rate = [1,2,4]
    # Encoder の dilated seg max
    enc_seg_max = 1024

    
    #ダウンサンプリングの割合
    ds_rate = 0.25
    
    # マスクの数と連続フレーム数
    n_mask = 0.065
    n_consec = 10
    
    # コードブックG のエントリの数V
    entryV = 320
    #コードブックの数
    num_codebook = 2

    #Gumbel Softmax の温度
    tau = 2.0
    temprature_multi = 0.999995
    # GumbleSoftmax の最小温度
    tau_min = 0.5

    # Decoderの設定
    # attnesion blockのレイヤー数
    dec_num_layers = 6
    # decoder の head の数
    dec_num_heads = 8
    # Decoder の Attention block の次元数
    dec_att_hidden_dim = 512
    # decoder 入力( decoder targets, encoder_outs ではない）の時間の最大数
    dec_target_maxlen = 1000
    # Deccoder の Attention Bolock の kernel_size
    dec_att_kernel_size = [5,1]
    # Decoder の Attention Block の filter_size
    dec_att_filter_size = 2048
    # Decoder の dropout
    dec_dropout = 0.1
    # Decoder の dilated transformer の segment_length
    dec_dil_seg = [256,512,1024]
    # Deccoder の dilated transformer の dilation rate
    dec_dil_rate = [1,2,4]
    # DEcoder の segment の設定値 decl_dil_seg[3]
    dec_seg_max = 1024

    # 初期学習率
    #initial_learning_rate = 1.0
    #initial_learning_rate = 5e-4
    #initial_learning_rate = 1e-4
    initial_learning_rate = 1e-5

    # Clipping Gradientの閾値
    clip_grad_threshold = 5.0
    #clip_grad_threshold = 1.0

    # 学習率の減衰やEarly stoppingの
    # 判定を開始するエポック数
    # (= 最低限このエポックまではどれだけ
    # validation結果が悪くても学習を続ける)
    lr_decay_start_epoch = 7

    # 学習率を減衰する割合
    # (減衰後学習率 <- 現在の学習率*lr_decay_factor)
    # 1.0以上なら，減衰させない
    lr_decay_factor = 0.5

    # Early stoppingの閾値
    # 最低損失値を更新しない場合が
    # 何エポック続けば学習を打ち切るか
    early_stop_threshold = 3

    # 学習過程で，認識エラー率を計算するか否か
    # 認識エラー率の計算は時間がかかるので注意
    # (ここではvalidationフェーズのみTrue(計算する)にしている)
    evaluate_error = {'train': True, 'validation': True}

    #
    # 設定ここまで
    #
    
    # Attention重み行列情報の保存先
    out_att_dir = os.path.join(output_dir, 'att_matrix')
    
    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_att_dir, exist_ok=True)

    # 設定を辞書形式にする
    config = {'fe_conv_layer' : fe_conv_layer,
              'fe_conv_channel' : fe_conv_channel,
              'fe_conv_kernel' : fe_conv_kernel,
              'fe_conv_stride' : fe_conv_stride,
              'fe_conv_dropout_rate' : fe_conv_dropout,
              'fe_out_dim' : fe_out_dim,
              'conv_layers' : conv_layers,
              'conv_channels' : conv_channels,
              'conv_kernel_size' : conv_kernel_size,
              'conv_dropout_rate' : conv_dropout_rate,
              'enc_num_layers': enc_num_layers,
              'enc_num_heads': enc_num_heads,
              'enc_input_maxlen' : enc_input_maxlen,
              'enc_att_hidden_dim': enc_att_hidden_dim,
              'enc_att_kernel_size': enc_att_kernel_size,
              'enc_att_filter_size': enc_att_filter_size,
              'enc_dil_seg': enc_dil_seg,
              'enc_dil_rate': enc_dil_rate,
              'enc_seg_max' : enc_seg_max,
              'downsampling_rate': ds_rate,
              'n_mask': n_mask,
              'n_consec': n_consec,
              'entryV': entryV,
              'num_codebook': num_codebook,
              'tau': tau,
              'temprature_multi': temprature_multi,
              'tau_min': tau_min,
              'enc_dropout_rate': enc_dropout,
              'dec_num_layers': dec_num_layers,
              'dec_num_heads': dec_num_heads,
              'dec_target_maxlen': dec_target_maxlen,
              'dec_att_hidden_dim': dec_att_hidden_dim,
              'dec_att_kernel_size': dec_att_kernel_size,
              'dec_att_filter_size': dec_att_filter_size,
              'dec_dropout_rate': dec_dropout,
              'dec_dil_seg': dec_dil_seg,
              'dec_dil_rate': dec_dil_rate,
              'dec_seg_max': dec_seg_max,
              'batch_size': batch_size,
              'max_num_epoch': max_num_epoch,
              'clip_grad_threshold': clip_grad_threshold,
              'initial_learning_rate': initial_learning_rate,
              'lr_decay_start_epoch': lr_decay_start_epoch, 
              'lr_decay_factor': lr_decay_factor,
              'early_stop_threshold': early_stop_threshold
             }

    # 設定をJSON形式で保存する
    conf_file = os.path.join(output_dir, 'config.json')
    with open(conf_file, mode='w', encoding='utf-8' ) as f:
        json.dump(config, f, indent=4)

    f.close()
    
    ## 特徴量の平均/標準偏差ファイルを読み込む
    #with open(mean_std_file, mode='r') as f:
    #    # 全行読み込み
    #    lines = f.readlines()
    #    #print( "lines:", lines )
    #    # 1行目(0始まり)が平均値ベクトル(mean)，
    #    # 3行目が標準偏差ベクトル(std)
    #    mean_line = lines[1]
    #    std_line = lines[3]
    #    # スペース区切りのリストに変換
    #    feat_mean = mean_line.split()
    #    #print( "feat_mean:",feat_mean )
    #    feat_std = std_line.split()
    #    # numpy arrayに変換
    #    feat_mean = np.array(feat_mean, 
    #                            dtype=np.float32)
    #    #print( "feat_mean:",feat_mean )
    #    feat_std = np.array(feat_std, 
    #                           dtype=np.float32)
    #    #print( "feat_std:", feat_std )    
    #
    #f.close()
    
    # 次元数の情報を得る
    #feat_dim = np.size(feat_mean)
    feat_dim = 1

    # トークンリストをdictionary型で読み込む
    # このとき，0番目は blank と定義する
    # (ただし，このプログラムではblankは使われない)
    token_list = {0: '<blank>'}
    with open(token_list_path, mode='r', encoding='utf-8' ) as f:
        # 1行ずつ読み込む
        for line in f: 
            # 読み込んだ行をスペースで区切り，
            # リスト型の変数にする
            parts = line.split()
            # 0番目の要素がトークン，1番目の要素がID
            token_list[int(parts[1])] = parts[0]
    f.close()

    # <eos>トークンをユニットリストの末尾に追加
    eos_id = len(token_list)
    token_list[eos_id] = '<eos>'
    # 本プログラムでは、<sos>と<eos>を
    # 同じトークンとして扱う
    #sos_id = eos_id
    sos_id = len(token_list)
    token_list[sos_id] = '<sos>'

    # トークン数(blankを含む)
    num_tokens = len(token_list)
    
    # ニューラルネットワークモデルを作成する
    # 入力の次元数は特徴量の次元数，
    # 出力の次元数はトークン数となる
    model = MyE2EModel(dim_in=feat_dim,
                       dim_out=num_tokens,
                       fe_conv_layer=fe_conv_layer,
                       fe_conv_channel=fe_conv_channel,
                       fe_conv_kernel=fe_conv_kernel,
                       fe_conv_stride=fe_conv_stride,
                       fe_conv_dropout_rate=fe_conv_dropout,
                       fe_out_dim=fe_out_dim,
                       conv_layers=conv_layers,
                       conv_channels=conv_channels,
                       conv_kernel_size=conv_kernel_size,
                       conv_dropout_rate=conv_dropout_rate,
                       enc_num_layers = enc_num_layers,
                       enc_att_hidden_dim=enc_att_hidden_dim,
                       enc_num_heads = enc_num_heads,
                       enc_input_maxlen = enc_input_maxlen, 
                       enc_att_kernel_size=enc_att_kernel_size,
                       enc_att_filter_size=enc_att_filter_size,
                       enc_dropout_rate = enc_dropout,
                       enc_dil_seg = enc_dil_seg,
                       enc_dil_rate = enc_dil_rate,
                       enc_seg_max = enc_seg_max,
                       ds_rate = ds_rate,
                       n_mask = n_mask,
                       n_consec = n_consec,
                       entryV = entryV,
                       num_codebook = num_codebook,
                       tau = tau,
                       temprature_multi = temprature_multi,
                       tau_min = tau_min,
                       dec_num_layers = dec_num_layers,
                       dec_att_hidden_dim=dec_att_hidden_dim,
                       dec_num_heads = dec_num_heads, 
                       dec_target_maxlen = dec_target_maxlen,
                       dec_att_kernel_size = dec_att_kernel_size,
                       dec_att_filter_size = dec_att_filter_size,
                       dec_dropout_rate = dec_dropout,
                       dec_dil_seg = dec_dil_seg,
                       dec_dil_rate = dec_dil_rate,
                       dec_seg_max = dec_seg_max,
                       sos_id=sos_id, 
                       )
    print(model)

    #for name, param in model.named_parameters():
    #    if name == 'quantize.linear1.weight':
    #        param.requires_grad = False
    #    elif name == 'quantize.linear1.bias':
    #        param.requires_grad = False
    #    elif name == 'quantize.linear2.weight':
    #        param.requires_grad = False
    #    elif name == 'quantize.linear2.bias':
    #        param.requires_grad = False
    #    elif name == 'quantize.linear5.weight':
    #        param.requires_grad = False
    #    elif name == 'quantize.linear5.bias':
    #        param.requires_grad = False

    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        #print( name, param.data )
    #        print( name )



    # オプティマイザを定義
    #optimizer = optim.Adadelta(model.parameters(),
    #                           lr=initial_learning_rate,
    #                           rho=0.95,
    #                           eps=1e-8,
    #                           weight_decay=0.0)
    optimizer = optim.AdamW(model.parameters(),
                               lr=initial_learning_rate,
                               eps = 1e-6,
                               weight_decay = 0.1
                               )

    # 訓練/開発データのデータセットを作成する
    train_dataset = SequenceDataset(feat_scp_train,
                                    label_train,
    #train_dataset = SequenceDataset(feat_scp_dev,
    #                                label_dev,
                                    #feat_mean,
                                    #feat_std,
                                    )

    # 開発データのデータセットを作成する
    dev_dataset = SequenceDataset(feat_scp_dev,
                                  label_dev,
                                  #feat_mean,
                                  #feat_std,
                                  )

    # 訓練データのDataLoaderを呼び出す
    # 訓練データはシャッフルして用いる
    #  (num_workerは大きい程処理が速くなりますが，
    #   PCに負担が出ます．PCのスペックに応じて
    #   設定してください)
    train_loader = DataLoader(train_dataset,
    #train_loader = DataLoader(dev_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    # 開発データのDataLoaderを呼び出す
    # 開発データはデータはシャッフルしない
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    del train_dataset
    del dev_dataset
    gc.collect()

    # クロスエントロピー損失を用いる．ゼロ埋めしているラベルを
    # 損失計算に考慮しないようにするため，ignore_index=0を設定
    #criterion = nn.CrossEntropyLoss(ignore_index=0)
    # CTC損失関数を呼び出す．
    # blankは0番目と定義する．
    #criterion = nn.CTCLoss(blank=0, reduction='mean')    

    # CUDAが使える場合はモデルパラメータをGPUに，
    # そうでなければCPUに配置する
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = model.to(device, non_blocking=True)

    #ダウンサンプリングの割合
    ds_rate = torch.tensor( ds_rate, device=torch.device(device) )
    
    # マスクの数と連続フレーム数
    n_mask = torch.tensor(n_mask, device=torch.device(device))
    n_consec = torch.tensor(n_consec, device=torch.device(device))
    
    # コードブックG のエントリの数V
    entryV = 320
    
    #Gumbel Softmax の温度
    tau = torch.tensor(tau, device=torch.device(device))
    temprature_multi = torch.tensor(temprature_multi, device=torch.device(device))
    tau_min = torch.tensor(tau_min, device=torch.device(device))

    # モデルをトレーニングモードに設定する
    model.train()

    # 訓練データの処理と開発データの処理を
    # for でシンプルに記述するために，辞書データ化しておく
    dataset_loader = {'train': train_loader,
    #dataset_loader = {'train': dev_loader,
                      'validation': dev_loader}

    # 各エポックにおける損失値と誤り率の履歴
    loss_history = {'train': [],
                    'validation': []}
    error_history = {'train': [],
                     'validation': []}

    # 本プログラムでは，validation時の損失値が
    # 最も低かったモデルを保存する．
    # そのため，最も低い損失値，
    # そのときのモデルとエポック数を記憶しておく
    best_loss = -1
    best_model = None
    best_epoch = 0
    # Early stoppingフラグ．Trueになると学習を打ち切る
    early_stop_flag = False
    # Early stopping判定用(損失値の最低値が
    # 更新されないエポックが何回続いているか)のカウンタ
    counter_for_early_stop = 0

    # ログファイルの準備
    log_file = open(os.path.join(output_dir,
                                 'log.txt'),
                                 mode='w', encoding='utf-8' )
    log_file.write('epoch\ttrain loss\t'\
                   'train err\tvalid loss\tvalid err')

    #num_steps =  len( dataset_loader['train'] ) * max_num_epoch
    ##lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    #lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.0001, total_iters=100)
    #warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=1000)
    #iters = len(  dataset_loader['train'] ) 
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor = 0.01, total_iters=100)

    del token_list
    gc.collect()

    # エポックの数だけループ
    for epoch in range(max_num_epoch):
        # early stopフラグが立っている場合は，
        # 学習を打ち切る
        if early_stop_flag:
            print('    Early stopping.'\
                  ' (early_stop_threshold = %d)' \
                  % (early_stop_threshold))
            log_file.write('\n    Early stopping.'\
                           ' (early_stop_threshold = %d)' \
                           % (early_stop_threshold))
            break

        # エポック数を表示
        print('epoch %d/%d:' % (epoch+1, max_num_epoch))
        log_file.write('\n%d\t' % (epoch+1))

        # trainフェーズとvalidationフェーズを交互に実施する
        for phase in ['train', 'validation']:
        #for phase in ['train']:
            # このエポックにおける累積損失値と発話数
            total_loss = 0
            total_utt = 0
            # このエポックにおける累積認識誤り文字数と総文字数
            total_error = 0
            total_token_length = 0

            # 各フェーズのDataLoaderから1ミニバッチ
            # ずつ取り出して処理する．
            # これを全ミニバッチ処理が終わるまで繰り返す．
            # ミニバッチに含まれるデータは，
            # 音声特徴量，ラベル，フレーム数，
            # ラベル長，発話ID
            n_batch = 0
            total_pos_sim = 0
            total_neg_sim = 0
            total_lm = 0
            total_ld = 0
            for (features, feat_lens, utt_ids) \
                    in dataset_loader[phase]:
                n_batch += 1
                    
                # 現時点でラベルのテンソルサイズは
                # [発話数 x 全データの最大ラベル長]
                # これを[発話数 x バッチ内の最大ラベル長]
                # に切る。(decoder部の冗長な処理を少なくするため。)
                features = features[:,:torch.max(feat_lens)]
                #labels = labels[:,:torch.max(label_lens)]

                # CUDAが使える場合はデータをGPUに，
                # そうでなければCPUに配置する
                features, feat_lens = features.to(device, non_blocking=True), feat_lens.to(device, non_blocking=True)

                # 勾配をリセット
                #optimizer.zero_grad()

                # モデルの出力を計算(フォワード処理)
                if phase == 'train':
                    outputs, outputs_lens, pgv_bar, mask, quantized_vector = model(features, feat_lens, tau )
                else:
                    with torch.no_grad():
                        outputs, outputs_lens, pgv_bar, mask, quantized_vector = model(features, feat_lens, tau )
                
                outputs, quantized_vector, mask_n_len = masking( outputs, quantized_vector, mask )
                
                #outputs = F.log_softmax( outputs, dim=2 )

                # クロスエントロピー損失関数の入力は
                # [(バッチ）ｘ（クラス）ｘ（時間）」なので、transpose する。
                # target は、hot_vector にしないで、「（バッチ）×（時間）」のままで良い。
                #loss = criterion( outputs.transpose(1,2).to(device), dec_target.to(device) )
                # 損失値を計算する．このとき，CTCLossへの入力は
                # [フレーム数 x バッチサイズ x クラス数] 
                # である必要があるため，テンソルの0軸と1軸を
                # 転置(transpose(0,1))した上で入力する
                #T = outputs.size(1)
                #outputs_lens[outputs_lens > T] = T
                #out_lens = outputs_lens
                #print( "out_lens:{}".format( out_lens ) )
                #start_lm = time.time()
                

                # Constrastive Loss and nagtive and positive_similarity
                #print("0 memory_usage:", get_memory_usage())
                lm_avg, pos_sim_avg_sum, neg_sim_sum_avg_avg_sum = lm_loss( outputs, quantized_vector, mask_n_len )
                #print("1 memory_usage:", get_memory_usage())
                total_pos_sim += pos_sim_avg_sum
                total_neg_sim += neg_sim_sum_avg_avg_sum
                total_lm += lm_avg.item()

                # diversity loss
                ld = ld_loss( pgv_bar )
                total_ld += ld.item()

                #alpha = 0.0
                #alpha = 0.1
                #alpha = 1.0
                #alpha = 10.0
                #alpha = 20.0
                alpha = 100.0
                #loss = lm_avg
                loss = lm_avg + alpha * ld

                #print( "loss", loss)
                #loss = criterion(outputs.transpose(0, 1),labels,out_lens,label_lens)
                #loss *= np.mean(label_lens.numpy())

                # 訓練フェーズの場合は，誤差逆伝搬を実行し，
                # モデルパラメータを更新する
                if phase == 'train':
                    # 勾配を計算する
                    #loss.backward(retain_graph=True)
                    loss.backward()
                    # Cliping Gradient により勾配が
                    # 閾値以下になるよう調整する
                    torch.nn.utils.clip_grad_norm_(\
                                              model.parameters(),
                                              clip_grad_threshold)
                    # オプティマイザにより，パラメータを更新する
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    tau = tau * temprature_multi
                    #with warmup_scheduler.dampening():
                    #    lr_scheduler.step( (epoch + 1 ) + n_batch / iters )

                '''
                # 認識エラーの算出をTrueにしている場合は，算出する
                if evaluate_error[phase]:
                    # バッチ内の1発話ごとに誤りを計算
                    for n in range(outputs.size(0)):
                        # 各ステップのデコーダ出力を得る
                        _, hyp_per_frame = torch.max(outputs[n], 1)
                        # numpy.array型に変換
                        hyp_per_frame = hyp_per_frame.cpu().numpy()
                        # 認識結果の文字列を取得
                        hypothesis = \
                           ctc_simple_decode(hyp_per_frame,
                                      token_list)
                        # 正解の文字列を取得
                        reference = []
                        for m in labels[n][:label_lens[n]].cpu().numpy().astype( np.int32 ):
                            if token_list[m] != "<sos>" and token_list[m] != "<eos>":
                                reference.append(token_list[m])
                        #print( *reference, sep="" )
                        # 認識誤りを計算
                        (error, substitute, 
                         delete, insert, ref_length) = \
                            levenshtein.calculate_error(hypothesis,
                                                        reference)
                        # 誤り文字数を累積する
                        total_error += error
                        # 文字の総数を累積する
                        total_token_length += ref_length
                        
                        if n < 4 and n_batch == len( dataset_loader[phase] ):
                            print( "%12s, reference :%s" % (phase,''.join(reference) ) )
                            print( "%12s, hypothesis:%s" % (phase,''.join(hypothesis) ) )
                '''

                # 損失値を累積する
                total_loss += loss.item()
                # 処理した発話数をカウントする
                total_utt += outputs.size(0)

                #b2 = psutil.virtual_memory()
                #print( "n_batch:{}, virturl_memory:{} ".format(n_batch, b2.used ) )
                #mem = torch.cuda.memory_allocated(device)
                #print( "n_batch:{}, memory:{} ".format(n_batch, mem ) )

                if n_batch % 5 == 0:
                    print( "n_batch:{:>2d},phase:{:>9s},lm:{:>.3e},ld:{:>.3e},loss:{:>.3e},pos_avg:{:>.3e},neg_avg:{:>.3e},lr:{:>.3e}".format( n_batch, phase, \
                        total_lm / n_batch, total_ld /  n_batch, total_loss /  n_batch, total_pos_sim / total_utt, total_neg_sim / total_utt, optimizer.param_groups[0]['lr'] ) )


                #print("2 memory_usage:", get_memory_usage())
            
            torch.cuda.empty_cache()

            #
            # このフェーズにおいて，1エポック終了
            # 損失値，認識エラー率，モデルの保存等を行う
            # 

            # 損失値の累積値を，処理した発話数で割る
            #epoch_loss = total_loss / total_utt
            epoch_loss = total_loss / n_batch
            # 画面とログファイルに出力する
            print("n_batch:{}".format( n_batch ))
            print('    %s loss: %f' \
                  % (phase, epoch_loss))
            log_file.write('%.6f\t' % (epoch_loss))
            # 履歴に加える
            loss_history[phase].append(epoch_loss)
            '''
            # 認識エラー率を計算する
            if evaluate_error[phase]:
                # 総誤りトークン数を，
                # 総トークン数で割ってエラー率に換算
                epoch_error = 100.0 * total_error \
                            / total_token_length
                # 画面とログファイルに出力する
                print('    %s token error rate: %f %%' \
                    % (phase, epoch_error))
                log_file.write('%.6f\t' % (epoch_error))
                # 履歴に加える
                error_history[phase].append(epoch_error)
            else:
                # エラー率を計算していない場合
                log_file.write('     ---     \t')
            '''
            #
            # validationフェーズ特有の処理を train で行う。
            #
            #if phase == 'validation':
            if phase == 'train':
                if epoch == 0 or best_loss > epoch_loss:
                    # 損失値が最低値を更新した場合は，
                    # その時のモデルを保存する
                    best_loss = epoch_loss
                    torch.save({'model_state_dict': model.state_dict(), 
                               'optimizer_state_dict': optimizer.state_dict(),},
                               output_dir+'/best_model.pt')
                    best_epoch = epoch
                    # Early stopping判定用の
                    # カウンタをリセットする
                    counter_for_early_stop = 0
                else:
                    # 最低値を更新しておらず，
                    if epoch+1 >= lr_decay_start_epoch:
                        # かつlr_decay_start_epoch以上の
                        # エポックに達している場合
                        if counter_for_early_stop+1 \
                               >= early_stop_threshold:
                            # 更新していないエポックが，
                            # 閾値回数以上続いている場合，
                            # Early stopping フラグを立てる
                            early_stop_flag = True
                        else:
                            #Early stopping条件に
                            #達していない場合は
                            #学習率を減衰させて学習続行
                            if lr_decay_factor < 1.0:
                                for i, param_group \
                                      in enumerate(\
                                      optimizer.param_groups):
                                    if i == 0:
                                        lr = param_group['lr']
                                        dlr = lr_decay_factor \
                                            * lr
                                        print('    (Decay '\
                                          'learning rate:'\
                                          ' %f -> %f)' \
                                          % (lr, dlr))
                                        log_file.write(\
                                          '(Decay learning'\
                                          ' rate: %f -> %f)'\
                                           % (lr, dlr))
                                    param_group['lr'] = dlr
                            # Early stopping判定用の
                            # カウンタを増やす
                            counter_for_early_stop += 1

        scheduler.step()

    #
    # 全エポック終了
    # 学習済みモデルの保存とログの書き込みを行う
    #
    print('---------------Summary'\
          '------------------')
    log_file.write('\n---------------Summary'\
                   '------------------\n')
   
    # 最終エポックのモデルを保存する
    torch.save({'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), },
               os.path.join(output_dir,'final_model.pt'))
    print('Final epoch model -> %s/final_model.pt' \
          % (output_dir))
    log_file.write('Final epoch model ->'\
                   ' %s/final_model.pt\n' \
                   % (output_dir))

    # 最終エポックの情報
    for phase in ['train', 'validation']:
        # 最終エポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][-1]))
        log_file.write('    %s loss: %f\n' \
                       % (phase, loss_history[phase][-1]))
        '''

        # 最終エポックのエラー率を出力    
        if evaluate_error[phase]:
            print('    %s token error rate: %f %%' \
                % (phase, error_history[phase][-1]))
            log_file.write('    %s token error rate: %f %%\n' \
                % (phase, error_history[phase][-1]))
        else:
            print('    %s token error rate: (not evaluated)' \
                % (phase))
            log_file.write('    %s token error rate: '\
                '(not evaluated)\n' % (phase))
        '''

    # ベストエポックの情報
    # (validationの損失が最小だったエポック)
    print('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt' \
          % (best_epoch+1, output_dir))
    log_file.write('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt\n' \
          % (best_epoch+1, output_dir))
    for phase in ['train', 'validation']:
        # ベストエポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][best_epoch]))
        log_file.write('    %s loss: %f\n' \
              % (phase, loss_history[phase][best_epoch]))
        '''
        # ベストエポックのエラー率を出力
        if evaluate_error[phase]:
            print('    %s token error rate: %f %%' \
                  % (phase, error_history[phase][best_epoch]))
            log_file.write('    %s token error rate: %f %%\n' \
                  % (phase, error_history[phase][best_epoch]))
        else:
            print('    %s token error rate: '\
                  '(not evaluated)' % (phase))
            log_file.write('    %s token error rate: '\
                  '(not evaluated)\n' % (phase))
        '''

    # 損失値の履歴(Learning Curve)グラフにして保存する
    fig1 = plt.figure()
    for phase in ['train', 'validation']:
        plt.plot(loss_history[phase],
                 label=phase+' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig1.legend()
    fig1.savefig(output_dir+'/loss.png')

    '''
    # 認識誤り率の履歴グラフにして保存する
    fig2 = plt.figure()
    for phase in ['train', 'validation']:
        if evaluate_error[phase]:
            plt.plot(error_history[phase],
                     label=phase+' error')
    plt.xlabel('Epoch')
    plt.ylabel('Error [%]')
    fig2.legend()
    fig2.savefig(output_dir+'/error.png')
    '''
    # ログファイルを閉じる
    log_file.close()

#
# メイン関数
#
if __name__ == "__main__":

    main()

