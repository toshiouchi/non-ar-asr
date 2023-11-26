# non-ar-asr
Machine laerning ASR python program, non autoregressive

Machine learning result

A machine learning neural ASR program learned With JSUT 1.1 5,000 data and Mozzila common voice Japanese 11 with 33,000 data. For 1,000 test data, TER was 17%.

Feature of program

Transformer with convolutional position wise feed forward network is used. 

TransformerEncoder has 12 layers, 4 heads, hidden dim 512. Also TrnasformerDecoder has 12 layers, 4 heads, hidden dim 512.

Position wise feed forward network in both TransformerEncoder and TransformerDecoder has two convolutional layers which have filters (512, 2048) , (2048, 512)), kernel sizes are 5 and 1, strides are both 1, layer norm and dropout with rate 0.1.
,
Input of encoder is 80 bin mel spectrogram, it is input in three convolational layers, filter_size of layers are (80,256), (256,256), (256,512), all kernel sizes are 5, all strides are 1. After convolution there are BatchNorm1d, ReLU and dropout with rate 0.1. And positional embedding values are generated. Then, the sum of comvolatio values and positional embedding values are input in TransformerEncoder with self attention module.

Especially encoder outputs of encoder reduce by 0.25 times with time axis using downsampling module.

TransformerDecoder is used as cross attention module. Source input is encoder ouput and target input is downsampled encoder output with positional embedding. 

Decoder(TransformerDecoder) output is input in a linear projection layer.

CTCLoss is used by loss calculation. 

CTCLoss is used, so ctc_simple_decode(int_vector, token_list) function in source code is used in order to decode outputs of model.inference.

Explanation of detail machine learning in Japanese
https://qiita.com/toshiouchi/items/033dd91bd10e1181297e
