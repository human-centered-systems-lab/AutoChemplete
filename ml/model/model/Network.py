import numpy as np
import torch
from torch import nn
import torchvision
from torch.nn import Sequential
import torch.utils.checkpoint as checkpoint
#import torchvision.models as models
#variation of CNN and RNN with attention

class Encoder(nn.Module):
    """
    Encoder network
    """
    def __init__(self, encoded_image_size=14, embed_dim=512, model_type='wide_res',
                 tf_encoder=0, checkpointing_cnn=0):
        """
        :param encoded_image_size: size of preprocessed image data
        :param model_type: select encoder model type from 'wide resnet', 'resnet', and 'resnext'
        """
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        self.projector = None #we add this here to match the size, because output of resnet is 2048, efficientnetB2 is  1048
        if model_type == 'wide_res':
            resnet = torchvision.models.wide_resnet101_2(pretrained=True)  # pretrained ImageNet wide_ResNet-101_2
        elif model_type == 'res':
            resnet = torchvision.models.resnet152(pretrained=True)  # pretrained ImageNet wide_ResNet152
        elif model_type == 'resnext':
            resnet = torchvision.models.resnext101_32x8d(pretrained=True)  # pretrained ImageNet wide_ResNet-101_2
        elif model_type == 'efficientnetB0':
            resnet = torchvision.models.efficientnet_b0(pretrained=True)  # pretrained ImageNet efficientnet_b0
            #self.projector = torch.nn.Linear(1280, 2048)

            if tf_encoder > 0:
                # if we have transformer encoders then we need to keep embed dim
                self.projector = torch.nn.Linear(1280, embed_dim)
            else:
                # 4 * embed dim because tqhis is the requirement for the init lstm function in the decoder
                self.projector = torch.nn.Linear(1280, 4 * embed_dim)
        elif model_type == 'efficientnetB2':
            resnet = torchvision.models.efficientnet_b2(pretrained=True)

            if tf_encoder > 0:
                # if we have transformer encoders then we need to keep embed dim
                self.projector = torch.nn.Linear(1408, embed_dim)
            else:
                # 4 * embed dim because this is the requirement for the init lstm function in the decoder
                self.projector = torch.nn.Linear(1408, 4 * embed_dim)
                # we add this here to match the size, because output of resnet is 2048, efficientnetB2 is 1048
        elif model_type == 'efficientnetB3':
            resnet = torchvision.models.efficientnet_b3(pretrained=True)

            if tf_encoder > 0:
                # if we have transformer encoders then we need to keep embed dim
                self.projector = torch.nn.Linear(1536, embed_dim)
            else:
                # 4 * embed dim because this is the requirement for the init lstm function in the decoder
                self.projector = torch.nn.Linear(1536, 4 * embed_dim)

        elif model_type == 'efficientnetB7':
            resnet = torchvision.models.efficientnet_b7(pretrained=True)
            if tf_encoder > 0:
                # if we have transformer encoders then we need to keep embed dim
                self.projector = torch.nn.Linear(2560, embed_dim)
            else:
            # 4 * embed dim because this is the requirement for the init lstm function in the decoder
               self.projector = torch.nn.Linear(2560, 4 * embed_dim)
            #s   elf.projector = torch.nn.Linear(2560, 2048)
            #image: 600*600
        else:
            print("Invalid network type:", model_type)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.checkpointing_cnn = checkpointing_cnn

        self.tf_encoder = tf_encoder
        if tf_encoder > 0:
            from .modules.TransformerLayers import TransformerEncoderLayer
            self.transformer_layers = nn.ModuleList()
            for i in range(tf_encoder):
                # default just keep heads = embed_dim // 64
                n_heads = embed_dim // 64
                tf_encoder_layer = TransformerEncoderLayer(embed_dim, n_heads)
                self.transformer_layers.append(tf_encoder_layer)
                self.tf_out_projector = torch.nn.Linear(embed_dim, 4 * embed_dim)

        self.fine_tune()


    def forward(self, images):

        # for the pretrained efficient net,  if we don't need to update the weights
        # then we can tell pytorch to save memory by releasing the intermediate layers of the
        # efficient net
        #images = images.contiguous()
        # this can help to save memory when using the Transformer

        if not self._fine_tune:
                with torch.no_grad():
                    out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
                    out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
                    out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        else:
            if self.checkpointing_cnn > 0 and self.training:
                images = images.detach()

                images.requires_grad_() # fix the error: None of the inputs have requires_grad=True. Gradients will be Non
                if len(self.resnet) > 1:
                    segment = min(self.checkpointing_cnn, len(self.resnet))
                    out = checkpoint.checkpoint_sequential(self.resnet, segment, images)
                else:
                    net = self.resnet[0]
                    segment = min(self.checkpointing_cnn, len(net))
                    out = checkpoint.checkpoint_sequential(net, segment, images)
            else:
                out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
            out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
            out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        if self.tf_encoder > 0:
            b, w, h = out.size(0), out.size(1), out.size(2)

            # we "flattened the 2d input to 1d"
            out = out.view(b, w*h, -1).transpose(0, 1).contiguous()

            # here we re-project the size of cnn output to 512 (for our transformer_
            if self.projector is not None:
                out = self.projector(out)

            # out here is [T x B x H] so remember to transpose again later
            for layer in self.transformer_layers:
                out = layer(out)

            # here we project the size from 512 to 2048 for the lstm layer
            out = out.transpose(0, 1)
            out = self.tf_out_projector(out)

            # re-view the output so we don't have to change the decoder
            out = out.view(b, w, h, -1)

        else:
            # later if you have transformer layers: start from here
            # for example out = self.transformer_layer(out) ....
            if self.projector is not None:
                out = self.projector(out)
        # to do
        return out

    def fine_tune(self, fine_tune=False):
        self._fine_tune = fine_tune
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        # for c in list(self.resnet.children())[5:]:
        #    for p in c.parameters():
        #        p.requires_grad = fine_tune


        # sgd:
        # weight = weight - learning_rate * gradient
        # no_grad(): pytorch don't compute gradient
        # if gradeint exist, but weight.requires_grad == False: no update
        # if we want to update: we need both conditions


class Attention(nn.Module):
    """
    Attention network for calculate attention value
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: input size of encoder network
        :param decoder_dim: input size of decoder network
        :param attention_dim: input size of attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


# class PredictiveDecoder(nn.Module):
#     """
#     Decoder network with attention network used for decode smile sequence from image
#     """
#     def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, device, encoder_dim=2048, dropout=1.):
#         """
#         :param attention_dim: input size of attention network
#         :param embed_dim: input size of embedding network
#         :param decoder_dim: input size of decoder network
#         :param vocab_size: total number of characters used in training
#         :param encoder_dim: input size of encoder network
#         :param dropout: dropout rate
#         """
#         super(PredictiveDecoder, self).__init__()
#
#         self.encoder_dim = encoder_dim
#         self.attention_dim = attention_dim
#         self.embed_dim = embed_dim
#         self.decoder_dim = decoder_dim
#         self.vocab_size = vocab_size
#         self.dropout = dropout
#         self.device = device
#
#         # attention network
#         self.attention = Attention(encoder_dim, decoder_dim, attention_dim).to(non_blocking=True)
#         # embedding layer
#         self.embedding = nn.Embedding(vocab_size, embed_dim).to(non_blocking=True)
#         self.dropout = nn.Dropout(p=self.dropout).to(non_blocking=True)
#         # decoding LSTMCell
#         self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True).to(non_blocking=True)
#         # linear layer to find initial hidden state of LSTMCell
#         self.init_h = nn.Linear(encoder_dim, decoder_dim).to(non_blocking=True)
#         # linear layer to find initial cell state of LSTMCell
#         self.init_c = nn.Linear(encoder_dim, decoder_dim).to(non_blocking=True)
#         # linear layer to create a sigmoid-activated gate
#         self.f_beta = nn.Linear(decoder_dim, encoder_dim).to(non_blocking=True)
#         self.sigmoid = nn.Sigmoid()
#         # linear layer to find scores over vocabulary
#         self.fc = nn.Linear(decoder_dim, vocab_size).to(non_blocking=True)
#
#     def init_hidden_state(self, encoder_out):
#         mean_encoder_out = encoder_out.mean(dim=1)
#         h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
#         c = self.init_c(mean_encoder_out)
#         return h, c
#
#     def forward(self, encoder_out, decode_lengths=70):
#         batch_size = encoder_out.size(0)
#         encoder_dim = encoder_out.size(-1)
#         vocab_size = self.vocab_size
#
#         encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
#         num_pixels = encoder_out.size(1)
#
#         # embed start tocken for LSTM input
#         start_tockens = torch.ones(batch_size, dtype=torch.long).to(self.device) * 68
#         embeddings = self.embedding(start_tockens)
#
#         # initialize hidden state and cell state of LSTM cell
#         h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
#
#         predictions = torch.zeros(batch_size, decode_lengths, vocab_size).to(self.device)
#
#         # predict sequence
#         for t in range(decode_lengths):
#             attention_weighted_encoding, alpha = self.attention(encoder_out, h)
#
#             gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
#             attention_weighted_encoding = gate * attention_weighted_encoding
#
#             h, c = self.decode_step(
#                 torch.cat([embeddings, attention_weighted_encoding], dim=1),
#                 (h, c))  # (batch_size_t, decoder_dim)
#
#             preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
#
#             predictions[:, t, :] = preds
#             if np.argmax(preds.detach().cpu().numpy()) == 69:
#                 break
#             embeddings = self.embedding(torch.argmax(preds, -1))
#
#         return predictions


class DecoderWithAttention(nn.Module):
    """
    Decoder network with attention network used for training
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, device, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: input size of attention network
        :param embed_dim: input size of embedding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of characters used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions=None, caption_lengths=None,
                mode='teacher_forcing', decode_lengths=70):

        if mode == 'teacher_forcing':
            return self.forward_teacher_forcing(encoder_out, encoded_captions, caption_lengths)
        elif mode == 'generation':
            return self.predict(encoder_out, decode_lengths=decode_lengths)

    def forward_teacher_forcing(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: output of encoder network
        :param encoded_captions: transformed sequence from character to integer
        :param caption_lengths: length of transformed sequence
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size


        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        #why sort?

        # embedding transformed sequence for vector
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # set decode length by caption length - 1 because of omitting start token
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # predict sequence
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def predict(self, encoder_out, decode_lengths=70):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        #num_pixels = encoder_out.size(1)

        # embed start tocken for LSTM input
        start_tockens = torch.ones(batch_size, dtype=torch.long).to(self.device) * 68
        #start_tockens： [68,68,68...]
        #shape:(B)

        embeddings = self.embedding(start_tockens)
        #shape: (B*Embedding_size)

        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        predictions = torch.zeros(batch_size, decode_lengths, vocab_size).to(self.device)

        # predict sequence
        for t in range(decode_lengths):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)

            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

            predictions[:, t, :] = preds
            if np.argmax(preds.detach().cpu().numpy()) == 69:
                break
            embeddings = self.embedding(torch.argmax(preds, -1))

        return predictions


import math
class PositionalEncoding(nn.Module):
    '''PE(pos,2i) =sin(pos/100002i/dmodel)
       PE(pos,2i+1) =cos(pos/100002i/dmodel)
    '''
    def __init__(self, model_size, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_size, 2).float() * (-math.log(10000.0) / model_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)

        # so pe should have size [max_len, 1, model_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [ batch_size, seq_len, model_size]
        '''
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


# def get_attn_subsequence_mask(seq):
#     '''
#     seq: [batch_size, tgt_len]
#     '''
#     attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
#     subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
#     subsequence_mask = torch.from_numpy(subsequence_mask).byte()
#     return subsequence_mask # [batch_size, tgt_len, tgt_len]


class TransformerDecoder(nn.Module):
    """
    Decoder network with attention network used for training
    """

    def __init__(self, embed_dim, decoder_dim, vocab_size, device,
                 encoder_dim=2048, dropout=0.5, n_layers=1, max_len=101):
        #max_len: we can set very high number
        """
        :param embed_dim: input size of embedding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of characters used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        """
        super(TransformerDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device
        #self.heads = n_heads
        assert embed_dim == decoder_dim, "For the Transformer, embed dim needs to be the same with decoder_rim"

        # self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.positional_encoder = PositionalEncoding(embed_dim, dropout=dropout, max_len=max_len)
        #max_length: the length of sequence during decoder
        self.layers = nn.ModuleList()
        from .modules.TransformerLayers import TransformerDecoderLayer
        for _ in range(n_layers):
            # its generally accepted that the head size is 64
            # so the number of heads is just decoder dim dividing by 64
            n_heads = decoder_dim // 64
            #print(self.dropout)
            self.layers.append(TransformerDecoderLayer(self.decoder_dim, n_heads, self.dropout.p))


        # self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        # self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        # self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, encoded_captions=None, caption_lengths=None,
                mode='teacher_forcing', decode_lengths=70):

        if mode == 'teacher_forcing':
            return self.forward_teacher_forcing(encoder_out, encoded_captions, caption_lengths)
        elif mode == 'generation':
            return self.predict(encoder_out, decode_lengths=decode_lengths)

    def forward_teacher_forcing(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: output of encoder network
        :param encoded_captions: transformed sequence from character to integer
        :param caption_lengths: length of transformed sequence
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        #vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # map the dimension of the encoder_out from the CNN dimensions to decoder dimension
        # we need to transpose because our transformer style is written in [T x B x H] (time-first) style
        # so the decoder input, and encoder output need to have this layout before going into the Transformer
        encoder_out = self.init_c(encoder_out).transpose(0, 1).contiguous()

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # encoder_out = encoder_out[sort_ind]
        # encoded_captions = encoded_captions[sort_ind]

        # embedding transformed sequence for vector
        # the decoder input is the N-1 first element in the encoded captions
        # for example the sequence is <bos> C C H H N H N N C H <eos>
        # then input is <bos> C C H H N H N N C H
        # the label is C C H H N H N N C H <eos>
        decoder_input = encoded_captions[:, :-1]
        embeddings = self.embedding(decoder_input)  # (batch_size, max_caption_length, embed_dim)
        # after embedding, we add position encoding
        embeddings = self.positional_encoder(embeddings)

        # initialize hidden state and cell state of LSTM cell
        # h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # set decode length by caption length - 1 because of omitting start token
        decode_lengths = (caption_lengths - 1).tolist()

        # predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # first, transpose x to have [T x B x H] (same layout with encoder out]
        x = embeddings.transpose(0, 1).contiguous()
        seq_len = x.size(0)
        self_attn_mask = torch.triu(x.new_ones(seq_len, seq_len), diagonal=1).bool()

        # run the Transformer decoder
        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_out, self_attn_mask, impl='fast')

        predictions = self.fc(x).transpose(0, 1).contiguous()

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def predict(self, encoder_out, decode_lengths=70):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        encoder_out = self.init_c(encoder_out).transpose(0, 1).contiguous()

        # embed start tocken for input
        start_tokens = torch.ones(batch_size, 1, dtype=torch.long).to(self.device) * 68
        embeddings = self.embedding(start_tokens)
        #b*1*emb_size
        next_token = embeddings

        #initialize previous output
        predictions = torch.zeros(batch_size, decode_lengths, vocab_size).to(self.device)

        for t in range(decode_lengths):

            # hint 1: the dimension of encoded_caption_t is [batch_size, T, model_size]
            # T is increasing over time
            # hint 2: when T changes, you need to recompute self_attn_mask
            if t == 0:
                encoded_caption_t = next_token
            else:
                encoded_caption_t = torch.cat([encoded_caption_t, next_token], dim=1)
            #print("encoded_caption_t.shape:", encoded_caption_t.shape) #b, T,model_size
            # hint 3: the input to the Transformer has dimension [T x batch_size x model_size]
            # -> you need a transpose somewhere
            self_attn_mask = torch.triu(encoded_caption_t.new_ones(encoded_caption_t.size(1),
                                                                   encoded_caption_t.size(1)), diagonal=1).bool()
            #print("self_attn_mask.shape", self_attn_mask.shape)

            embeddings = self.positional_encoder(encoded_caption_t)
            embeddings = embeddings.transpose(0, 1).contiguous()
            # hint 5: is encoded caption_t embeddings or tokens?

            for i, layer in enumerate(self.layers):
                embeddings = layer(embeddings, encoder_out, self_attn_mask, impl='fast')
            preds = self.fc(embeddings[-1])
            #dec_outputs: T B Model_size
            #fc(dec_outputs): T* B *vacabulary_size
            #project: B*T*v
            # prob = projected.max(dim=-1, keepdim=False)[1]
            # it will find the maximum values but only in the last dimension
            # -> there are BxT max values because your tensor is BxTxV
            # this function doesn't return the actualy max values, they return the index of the max values
            # (where the values are, not what they are)
            # next_word = prob.data[:, -1].unsqueeze(1) # T*1 this line needs another command to ensure that line 624 is correct\
            #data[:, -1]: T,
            #next_token = self.embedding(next_word)
           # print("next_token", next_token.shape)
            predictions[:, t, :] = preds
            if np.argmax(preds.detach().cpu().numpy()) == 69:
                break
            next_token = self.embedding(torch.argmax(preds, -1)).unsqueeze(1)
            # hint 4: squeeze means that the dimension in bracket is 1, so it will remove the dimension
            # for example [1, batch_size, vocab_size] -> [batch_size, vcoab_size]

        return predictions


            # if t == 0:
            #     for i, layer in enumerate(self.layers):
            #     #???? how to call the transfomerDecoderLayer from self.layers.
            #         #print("layer", layer)
            #         #x = layer(x, encoder_out, self_attn_mask, impl='fast')
            #         layer_output = layer(x, encoder_out, self_attn_mask, impl='fast')
            #         x = layer_output
            # else:
            #
            #     for i, layer in enumerate(self.layers):
            #         previous_output = layer(x, encoder_out, self_attn_mask, impl='fast')
            #         h = previous_output
            #         #print("layer", layer)
            #         #x = layer(x, encoder_out, self_attn_mask, impl='fast')
            #         layer_output = layer(torch.cat([layer_output, previous_output], dim=1), encoder_out, self_attn_mask, impl='fast')
            #         x = previous_output
            # t += 1

            # get the prediction from x/ layer_output
            # recompute embeddings/ position encodings blah blah
            # IMPORTANT: for LSTM, the input to each cell has T=1
            # for Transformer, the inpt to each cell has T >=1 so that we don't have to keep the states
            # for the previous time steps

            # In other word: in every time step: we RECOMPUTE the hidden states of the previous time steps
            # and the length of the input (x, embeddings/encoded caption) increases over time




