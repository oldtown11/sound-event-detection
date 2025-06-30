import warnings

import torch.nn as nn
import torch
from .RNN import BidirectionalGRU
from .CNN import CNN
from .conformer.conformer_encoder import ConformerEncoder, ConformerPromptedEncoder
from .beats.BEATs import BEATsModel
class CRNN(nn.Module):
    def __init__(
        self,
        n_in_channel=1,
        nclass=10,
        attention=True,
        activation="glu",
        dropout=0.5,
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        cnn_integration=False,
        freeze_bn=False,
        embedding_size=527,
        embedding_type="global",
        aggregation_type="global",
        n_classtoken = 1,
        n_prompts = 10,
        **kwargs,
    ):
        """
            Initialization of CRNN model
        
        Args:
            n_in_channel: int, number of input channel
            n_class: int, number of classes
            attention: bool, adding attention layer or not
            activation: str, activation function
            dropout: float, dropout
            train_cnn: bool, training cnn layers
            rnn_type: str, rnn type
            n_RNN_cell: int, RNN nodes
            n_layer_RNN: int, number of RNN layers
            dropout_recurrent: float, recurrent layers dropout
            cnn_integration: bool, integration of cnn
            freeze_bn: 
            **kwargs: keywords arguments for CNN.
        """
        super(CRNN, self).__init__()

        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn
        self.embedding_type = embedding_type
        self.aggregation_type = aggregation_type
        n_in_cnn = n_in_channel
        if cnn_integration:
            n_in_cnn = 1

        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell*2, nclass)
        self.sigmoid = nn.Sigmoid()
        self.dense_softmax = nn.Linear(n_RNN_cell*2 , nclass)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_768 = nn.Linear(768, nclass)
        # embeddings
        self.cat_tf = torch.nn.Linear(nb_in+embedding_size, nb_in)

        # # todo e2e
        # # 定义beatsmodel
        self.beats = BEATsModel(cfg_path="data/pretrained_models/BEATs_iter3_plus_AS2M.pt")
        # # 让beats model本身的参数不训练，prompt训练
        for k, v in self.beats.named_parameters():
            if k.find('prompt_token') != -1:
                v.requires_grad = True
                print(k,'   :   ',v.requires_grad)
            elif k.find('frame_prompt') != -1:
                v.requires_grad = True
                print(k,'   :   ',v.requires_grad)
            else :
                v.requires_grad = False
        
    def forward(self, x, audio, pad_mask=None, embeddings=None, tuning_tag=True):
        # x [160, 128, 626] (batch_size, n_freq, n_frames)
        # embeddings.shape = [160, 768, 496] # [bs, ? , frames]
        # todo e2e
        if audio==None:
            # print("audio is None, without beats prompt")
            embeddings = embeddings
        else:
            embeddings, cls_token = self.beats(audio)
            embeddings = embeddings.transpose(1, 2)
        # todo pado:
        x = x.transpose(1, 2).unsqueeze(1) # [160, 1, 626, 128] (batch_size, n_channels, n_frames, n_freq)

        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # (batch_size, n_channels, n_frames, n_freq) [160, 1, 626, 128]
        # conv features
        x = self.cnn(x) # [160, 128, 156, 1]
        bs, chan, frames, freq = x.size() 
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan] [160, 156, 128]
        # embeddings.shape = [160, 768, 496] # [bs, ? , frames]
        reshape_emb = torch.nn.functional.adaptive_avg_pool1d(embeddings, x.shape[1]).transpose(1, 2) #reshape_emb: [160, 156, 768] 
        # x : [160, 156, 128]   reshape_emb : [160, 156, 768] 
        x = self.cat_tf(torch.cat((x, reshape_emb), -1)) # cat_tf : [160, 156, 896] - > [160, 156, 128] 
      
        # 108 156 128 #  128  20个强
        
        # x = torch.cat([self.cls_token.expand(bs, -1, -1), x], dim=1) # [bs, frames, chan] 108 157 128
        x = self.rnn(x) 
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)

        sof = self.dense_softmax(x)  # [bs, frames, nclass]
        if not pad_mask is None:
            sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)  # mask attention
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        cls_token = cls_token.mean(dim=1)
        cls_token = self.linear_768(cls_token)
        cls_token = torch.sigmoid(cls_token)
        weak = cls_token * 0.5 + weak * 0.5

        return strong.transpose(1, 2), weak

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CRNN, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
