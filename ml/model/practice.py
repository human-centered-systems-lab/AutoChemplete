
'''
Tutorial: Transformer
https://wmathor.com/index.php/archives/1455/
'''

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])

    #if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len): function?

    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数 ::步长为2取元素

    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    return positional_encoding

positional_encoding = get_positional_encoding(max_seq_len=100, embed_dim=16)
#plt.figure(figsize=(10,10))
#sns.heatmap(positional_encoding)
#plt.title("Sinusoidal Function")
#plt.xlabel("hidden dimension")




# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5 # enc_input max sequence length
tgt_len = 6 # dec_input(=dec_output) max sequence length

def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
        # print("enc_input", enc_input)
        # enc_input [[1, 2, 3, 4, 0]]
        # enc_input [[1, 2, 3, 5, 0]]

        enc_inputs.extend(enc_input)
        # print("enc_inputs.extend(enc_input)", enc_inputs)
        # enc_inputs.extend(enc_input) [[1, 2, 3, 4, 0]]
        # enc_inputs.extend(enc_input) [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]

        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
# print("enc_inputs", enc_inputs)
# enc_inputs tensor([[1, 2, 3, 4, 0],
#                    [1, 2, 3, 5, 0]])

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        print("enc_inputs[idx]", enc_inputs[idx])
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        #print("pe", pe.shape) #pe torch.Size([6, 5]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #print("position:", position.shape) #position: torch.Size([6, 1])
        #div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_sin = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #？？？？
        div_term_cos = torch.exp(torch.arange(1, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term_sin)
        pe[:, 1::2] = torch.cos(position * div_term_cos)
        #print("pe", pe.unsqueeze(0).shape) #torch.Size([1, 6, 5])
        pe = pe.unsqueeze(0).transpose(0, 1) #torch.Size([6, 1, 5])
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#model = PositionalEncoding(5, max_len=6)
#print(model)
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    #print("batch_size, len_q:", batch_size, len_q)
    batch_size, len_k = seq_k.size()
    #print("batch_size, len_k:", batch_size, len_k)
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    #print(pad_attn_mask)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

# seq_q = torch.randn(10, 5)
# seq_k = torch.randn(10, 6)

#get_attn_pad_mask(seq_q, seq_k)
#batch_size, len_q: 10 5
#batch_size, len_k: 10 6

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    #print("subsequence_mask" ,subsequence_mask)

    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    #why we need this line????
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

seq = torch.randn(10, 5)
get_attn_subsequence_mask(seq)


# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()
#
#     def forward(self, Q, K, V, attn_mask):
#         '''
#         Q: [batch_size, n_heads, len_q, d_k]
#         K: [batch_size, n_heads, len_k, d_k]
#         V: [batch_size, n_heads, len_v(=len_k), d_v]
#         attn_mask: [batch_size, n_heads, seq_len, seq_len]
#
#         d_k is dimension of k
#         '''
#         print("K.transpose(-1, -2).shape", K.transpose(-1, -2).shape)
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
#         scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
#
#         attn = nn.Softmax(dim=-1)(scores)
#         context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
#         return context, attn
#
# Q = torch.randn(10, 8, 6, 7)
# K = torch.randn(10, 8, 8, 7)
# V = torch.randn(10, 8, 8, 9)
# attn_mask = torch.randn(10, 8, 5, 5)
# model = ScaledDotProductAttention()
# model.forward(Q, K, V, attn_mask)
#K.transpose(-1, -2).shape: torch.Size([10, 8, 7, 8])

encoder_out = torch.randn(12, 7, 6)
print("encoder_out", encoder_out)
batch_size = 12
encoder_dim = 5

encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
print("encoder_out after view", encoder_out)
num_pixels = encoder_out.size(1)
print("num_pixels", num_pixels)

caption_lengths = 10
caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
print("caption_lengths, sort_ind", caption_lengths, sort_ind)
encoder_out = encoder_out[sort_ind]
print("encoder_out[sort_ind]", encoder_out)

encoded_captions = torch.range(10)
print("encoded_captions", encoded_captions)
encoded_captions = encoded_captions[sort_ind]
print("encoded_captions[sort_ind]", encoded_captions)
#why sort?\

# set decode length by caption length - 1 because of omitting start token
decode_lengths = (caption_lengths - 1).tolist()


for t in range(max(decode_lengths)):
    batch_size_t = sum([l > t for l in decode_lengths])
    print(batch_size_t)



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

        # predict sequence
        # for t in range(max(decode_lengths)):
        #     batch_size_t = sum([l > t for l in decode_lengths])
        #
        #     attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
        #
        #     gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
        #     attention_weighted_encoding = gate * attention_weighted_encoding
        #
        #     h, c = self.decode_step(
        #         torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
        #         (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
        #
        #     preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
        #     predictions[:batch_size_t, t, :] = preds
        #
        #     alphas[:batch_size_t, t, :] = alpha

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
        #num_pixels = encoder_out.size(1)

        encoder_out = self.init_c(encoder_out).transpose(0, 1).contiguous()

        # embed start tocken for input
        start_tokens = torch.ones(batch_size, dtype=torch.long).to(self.device) * 68
        embeddings = self.embedding(start_tokens)
        embeddings = self.positional_encoder(embeddings)
        #print("embedding_size", embeddings.shape),size: ([1, 101, 512]) (B*T*model_size)

        predictions = torch.zeros(batch_size, decode_lengths, vocab_size).to(self.device)


        # encoded_caption_t = start_tokens
        # embeddings = self.embedding(encoded_caption_t)
        # embeddings = self.positional_encoder(embeddings)
        h = torch.zeros(batch_size, decode_lengths, vocab_size).to(self.device)

        for t in range(decode_lengths):

            # suggestion:
            # for every timestep t: your encoded_caption_t is encoded_caption_t + previous_output
            # recompute embeddings
            # recompute embedings w/ position encoding

            # x = embeddings.tranpose(0, 1)
            # first, transpose x to have [T x B x H] (same layout with encoder out]

            #encoded_caption_t
            x = embeddings.transpose(0, 1).contiguous()
            #print("x", x.shape)
            seq_len = x.size(0)
            self_attn_mask = torch.triu(x.new_ones(seq_len, seq_len), diagonal=1).bool()

            #previous_output for the start token


            for i, layer in enumerate(self.layers):
                layer_output = layer(torch.cat([x, h], dim=1), encoder_out, self_attn_mask, impl='fast')

                x = layer_output

                previous_output = x
                h = torch.cat([h, previous_output], dim=1)


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











            predictions = self.fc(x).transpose(0, 1).contiguous()

            if np.argmax(predictions.detach().cpu().numpy()) == 69:
                break
            embeddings = self.embedding(torch.argmax(predictions, -1))

        return predictions



'''
#torch.unsqueeze()
import torch

x = torch.Tensor([1, 2, 3, 4])  # torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。

print('-' * 50)
print(x)  # tensor([1., 2., 3., 4.])
print(x.size())  # torch.Size([4])
print(x.dim())  # 1
print(x.numpy())  # [1. 2. 3. 4.]

print('-' * 50)
print(torch.unsqueeze(x, 0))  # tensor([[1., 2., 3., 4.]])
print(torch.unsqueeze(x, 0).size())  # torch.Size([1, 4])
print(torch.unsqueeze(x, 0).dim())  # 2
print(torch.unsqueeze(x, 0).numpy())  # [[1. 2. 3. 4.]]

print('-' * 50)
print(torch.unsqueeze(x, 1))
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
print(torch.unsqueeze(x, 1).size())  # torch.Size([4, 1])
print(torch.unsqueeze(x, 1).dim())  # 2

print('-' * 50)
print(torch.unsqueeze(x, -1))
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
print(torch.unsqueeze(x, -1).size())  # torch.Size([4, 1])
print(torch.unsqueeze(x, -1).dim())  # 2

print('-' * 50)
print(torch.unsqueeze(x, -2))  # tensor([[1., 2., 3., 4.]])
print(torch.unsqueeze(x, -2).size())  # torch.Size([1, 4])
print(torch.unsqueeze(x, -2).dim())  # 2

# 边界测试
# 说明：A dim value within the range [-input.dim() - 1, input.dim() + 1) （左闭右开）can be used.
# print('-' * 50)
# print(torch.unsqueeze(x, -3))
# IndexError: Dimension out of range (expected to be in range of [-2, 1], but got -3)

# print('-' * 50)
# print(torch.unsqueeze(x, 2))
# IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)

# 为何取值范围要如此设计呢？
# 原因：方便操作
# 0(-2)-行扩展
# 1(-1)-列扩展
# 正向：我们在0，1位置上扩展
# 逆向：我们在-2，-1位置上扩展
# 维度扩展：1维->2维，2维->3维，...，n维->n+1维
# 维度降低：n维->n-1维，n-1维->n-2维，...，2维->1维

# 以 1维->2维 为例，

# 从【正向】的角度思考：

# torch.Size([4])
# 最初的 tensor([1., 2., 3., 4.]) 是 1维，我们想让它扩展成 2维，那么，可以有两种扩展方式：

# 一种是：扩展成 1行4列 ，即 tensor([[1., 2., 3., 4.]])
# 针对第一种，扩展成 [1, 4]的形式，那么，在 dim=0 的位置上添加 1

# 另一种是：扩展成 4行1列，即
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
# 针对第二种，扩展成 [4, 1]的形式，那么，在dim=1的位置上添加 1

# 从【逆向】的角度思考：
# 原则：一般情况下， "-1" 是代表的是【最后一个元素】
# 在上述的原则下，
# 扩展成[1, 4]的形式，就变成了，在 dim=-2 的的位置上添加 1
# 扩展成[4, 1]的形式，就变成了，在 dim=-1 的的位置上添加 1
'''

'''
import pandas as pd

from src.config import input_data_dir, base_file_name, sample_submission_dir, sample_submission_labels_dir, reversed_token_map_dir
from rdkit import Chem, DataStructs
from Levenshtein import distance as levenshtein_distance

def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                d[(i-1,j)] + 1, # deletion
                d[(i,j-1)] + 1, # insertion
                d[(i-1,j-1)] + cost, # substitution
            )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]


predict_file = pd.read_csv('good_test.csv')
labels_file = pd.read_csv('/org/temp/anon/data/results_training/E2+2transformer+LSTM_1M_group1/good_test_labels.csv')
count = 0
sum_Tan = 0
sum_Lev = 0
sum_dam = 0
for _, row in predict_file.head(10000).iterrows(): #Iterate over DataFrame rows as (index, Series) pairs.
    idx = row['file_name']
    smiles_pred = row['SMILES']
    smiles_label = labels_file.loc[_]['SMILES']
    label_idx = labels_file.loc[_]['file_name']
    print("_", _)
    print('pred_idx', idx)
    print('smiles_pred:', smiles_pred)
    print("smiles_label:", smiles_label)
    print('label_idx:', label_idx)
    try:
        ref_pred = Chem.MolFromSmiles(smiles_pred)
        fp_pred = Chem.RDKFingerprint(ref_pred)
    except:
        print('Invalid SMILES:', smiles_pred)

    ref_label = Chem.MolFromSmiles(smiles_label)
    fp_label = Chem.RDKFingerprint(ref_label)

    Tan = DataStructs.TanimotoSimilarity(fp_pred,fp_label)
    print("Tanimoto Smililarity:", Tan)
    sum_Tan = sum_Tan + Tan
    count += 1

    #calculate levenshtein distance
    leven = levenshtein_distance(smiles_pred, smiles_label)
    leven = 1 - leven/ max(len(smiles_pred), len(smiles_label))
    print("levenshtein distance:", leven)
    sum_Lev = sum_Lev + leven

    #calculate Damerau-levenshtein distance
    dam_lev = damerau_levenshtein_distance(smiles_pred, smiles_label)
    dam_lev = 1 -dam_lev / max(len(smiles_pred), len(smiles_label))
    print("Damerau-levenshtein distance:", dam_lev)
    sum_dam = sum_dam + dam_lev


average = sum_Tan/count
average_lev = sum_Lev/count
average_dam = sum_dam/count
print("Tanimoto average:", average)
print("levenshtein average:", average_lev)
print("Damerau-levenshtein distance avergae:", average_dam)
'''
