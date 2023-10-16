# -*- coding: utf-8 -*-

#
# RNN Attention Encoder-Decoderモデルを学習します．
#
#MultiheadAttention の layer_norm と dropout 使わないようにに戻した。

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

# 作成したDatasetクラスをインポート
from my_dataset import SequenceDataset

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

# os, sys, shutilモジュールをインポート
import os
import sys
import shutil

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


#
# メイン関数
#
if __name__ == "__main__":
    
    #
    # 設定ここから
    #

    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit = 'char'

    # 評価データの特徴量(Feats.scp)が存在するディレクトリ
    feat_dir_test = '../01compute_features/fbank/test'

    # 実験ディレクトリ
    # train_set_name = 'train_small' or 'train_large'
    #exp_dir = './exp_' + os.path.basename(feat_dir_test) 
    exp1_dir = './exp_train_large'

    # 学習/開発データの特徴量リストファイル
    feat_scp_test = os.path.join(feat_dir_test, 'feats.scp')

    # 学習/開発データのラベルファイル
    label_test = os.path.join(exp1_dir, 'data', unit, 'label_test')

    # 学習済みモデルが格納されているディレクトリ
    #model_dir = os.path.join(exp_dir, unit+'_model_conv_non_ar_001')
    model1_dir = os.path.join(exp1_dir, unit+'_model_conv_non_ar_001')

    
    # 訓練データから計算された特徴量の平均/標準偏差ファイル
    mean_std_file = os.path.join(model1_dir, 'mean_std.txt')

    # トークンリスト
    token_list_path = os.path.join(exp1_dir, 'data', unit,
                                   'token_list')
    # 学習結果を出力するディレクトリ
    output_dir = os.path.join(exp1_dir, unit+'_model_conv_non_ar_001')

    # 学習済みのモデルファイル
    model_file = os.path.join(output_dir, 'final_model.pt')

    # デコード結果を出力するディレクトリ
    output_dir2 = os.path.join(output_dir, 'decode_test')

    # デコード結果および正解文の出力ファイル
    hypothesis_file = os.path.join(output_dir2, 'hypothesis.txt')
    reference_file = os.path.join(output_dir2, 'reference.txt')

    # ミニバッチに含める発話数
    #batch_size = 10
    batch_size = 8

    #
    # 設定ここまで
    #

    # 学習時に出力した設定ファイル
    config_file = os.path.join(model1_dir, 'config.json')
    
    # 設定ファイルを読み込む
    with open(config_file, mode='r') as f:
        config = json.load(f)

    # 読み込んだ設定を反映する

    conv_layers = config['conv_layers']
    conv_channels = config['conv_channels']
    conv_kernel_size = config['conv_kernel_size']
    conv_dropout = config['conv_dropout_rate']
    enc_num_layers = config['enc_num_layers']
    enc_num_heads = config['enc_num_heads']
    enc_input_maxlen = config['enc_input_maxlen']
    enc_att_hidden_dim = config['enc_att_hidden_dim']
    enc_att_kernel_size = config['enc_att_kernel_size']
    enc_att_filter_size = config['enc_att_filter_size']
    ds_rate = config['downsampling_rate']
    enc_dropout = config['enc_dropout_rate']
    dec_num_layers = config['dec_num_layers']
    dec_num_heads = config['dec_num_heads']
    dec_target_maxlen = config['dec_target_maxlen']
    dec_att_hidden_dim = config['dec_att_hidden_dim']
    dec_att_kernel_size = config['dec_att_kernel_size']
    dec_att_filter_size = config['dec_att_filter_size']
    dec_dropout = config['dec_dropout_rate']
    batch_size = config['batch_size']
    max_num_epoch = config['max_num_epoch']
    clip_grad_threshold = config['clip_grad_threshold']
    initial_learning_rate = config['initial_learning_rate']
    lr_decay_start_epoch = config['lr_decay_start_epoch']
    lr_decay_factor = config['lr_decay_factor']
    early_stop_threshold = config['early_stop_threshold']

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)


    # 特徴量の平均/標準偏差ファイルを読み込む
    with open(mean_std_file, mode='r', encoding='utf-8' ) as f:
        # 全行読み込み
        lines = f.readlines()
        # 1行目(0始まり)が平均値ベクトル(mean)，
        # 3行目が標準偏差ベクトル(std)
        mean_line = lines[1]
        std_line = lines[3]
        # スペース区切りのリストに変換
        feat_mean = mean_line.split()
        feat_std = std_line.split()
        # numpy arrayに変換
        feat_mean = np.array(feat_mean, 
                                dtype=np.float32)
        feat_std = np.array(feat_std, 
                               dtype=np.float32)

    # 次元数の情報を得る
    feat_dim = np.size(feat_mean)

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
                       conv_layers=conv_layers,
                       conv_channels=conv_channels,
                       conv_kernel_size=conv_kernel_size,
                       conv_dropout_rate=conv_dropout,
                       enc_num_layers = enc_num_layers,
                       enc_att_hidden_dim=enc_att_hidden_dim,
                       enc_num_heads = enc_num_heads,
                       enc_input_maxlen = enc_input_maxlen, 
                       enc_att_kernel_size=enc_att_kernel_size,
                       enc_att_filter_size=enc_att_filter_size,
                       enc_dropout_rate = enc_dropout,
                       ds_rate = ds_rate,
                       dec_num_layers = dec_num_layers,
                       dec_att_hidden_dim=dec_att_hidden_dim,
                       dec_num_heads = dec_num_heads, 
                       dec_target_maxlen = dec_target_maxlen,
                       dec_att_kernel_size = dec_att_kernel_size,
                       dec_att_filter_size = dec_att_filter_size,
                       dec_dropout_rate = dec_dropout,
                       sos_id=sos_id, 
                       )
    print(model)

    # オプティマイザを定義
    optimizer = optim.Adadelta(model.parameters(),
                               lr=initial_learning_rate,
                               rho=0.95,
                               eps=1e-8,
                               weight_decay=0.0)
                               
    # モデルのパラメータを読み込む
    model.load_state_dict(torch.load(model_file))

    # 訓練/開発データのデータセットを作成する
    test_dataset = SequenceDataset(feat_scp_test,
                                    label_test,
                                    feat_mean,
                                    feat_std)


    # 訓練データのDataLoaderを呼び出す
    # 訓練データはシャッフルして用いる
    #  (num_workerは大きい程処理が速くなりますが，
    #   PCに負担が出ます．PCのスペックに応じて
    #   設定してください)
    test_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4)

    # クロスエントロピー損失を用いる．ゼロ埋めしているラベルを
    # 損失計算に考慮しないようにするため，ignore_index=0を設定
    #criterion = nn.CrossEntropyLoss(ignore_index=0)
    # CTC損失関数を呼び出す．
    # blankは0番目と定義する．
    criterion = nn.CTCLoss(blank=0, reduction='mean')    
    

    # CUDAが使える場合はモデルパラメータをGPUに，
    # そうでなければCPUに配置する
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # 学習途中で、学習の進み度合を確認するため、推論する。
    # モデルを評価モードに設定する
    model.eval()
    
    # デコード結果および正解ラベルをファイルに書き込みながら
    # 以下の処理を行う
    with open(hypothesis_file, mode='w') as hyp_file, \
         open(reference_file, mode='w') as ref_file:
        for (features2, labels2, feat_lens2,
                   label_lens2, utt_ids2) in test_loader:
        
            #
            # ラベルの末尾に<eos>を付与する最初に<sos>
            #
            # ゼロ埋めにより全体の長さを1増やす
            labels2 = torch.cat( ( torch.zeros(labels2.size()[0],1,dtype=torch.long), labels2), dim = 1 )
            labels2 = torch.cat((labels2,
                                torch.zeros(labels2.size()[0],
                                1,
                                dtype=torch.long)), dim=1)
            # 末尾に<eos>追加、最初に<sos> を追加
            for m, length in enumerate(label_lens2):
                labels2[m][0] = sos_id
                labels2[m][length+1] = eos_id
            label_lens2 += 2

            # モデルの出力を計算(フォワード処理)
            with torch.no_grad():
                outputs3 = model.inference(features2.to(device), feat_lens2.to(device) )
                outputs3 = F.log_softmax( outputs3, dim = 2 )

            # バッチ内の1発話ごとに以下の処理を行う
            for n in range(outputs3.size(0)):
                
                # 正解の文字列を取得
                reference2 = []
                for m in labels2[n][:label_lens2[n]].cpu().numpy():
                    reference2.append(token_list[m])
            
                print( "train    reference:   ", *reference2, sep="" )                
                
                idx3 = n

                # 各ステップのデコーダ出力を得る
                _, hyp_per_frame3 = torch.max(outputs3[idx3], 1)
                    
                # numpy.array型に変換
                hyp_per_frame3 = hyp_per_frame3.detach().cpu().numpy()
                # 認識結果の文字列を取得
                hypothesis = \
                    ctc_simple_decode(hyp_per_frame3,
                                      token_list)
                                             
                #print( "train    hypothesis:  ", *hypothesis2, sep="" )
                print( "train    hypothesis:  ", *hypothesis, sep="" )
                
                # 結果を書き込む
                # (' '.join() は，リスト形式のデータを
                # スペース区切りで文字列に変換している)
                hyp_file.write('%s %s\n' \
                    % (utt_ids2[idx3], ' '.join(hypothesis)))
                ref_file.write('%s %s\n' \
                    % (utt_ids2[idx3], ' '.join(reference2)))
             
