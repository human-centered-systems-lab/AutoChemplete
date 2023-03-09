import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from time import time


# LSTM = long short term memory recurrent neural nets
# there are many implementations of LSTMs, why?
# - people want to try different variations, for example multiplicative LSTM, or convolutional LSTM
# - the "default" LSTMs in pytorch or tensorflow is VERY SLOW, and other people can make it run faster
# now everyone uses nn.LSTM because:
# - people are not interested in LSTM anymore
# - the default LSTM has been implemented well (as fast as the previous fast versions)
# - it takes years for the developers to finalize an implementation of LSTM to satisfy the users


def scale_dot_product_attention(q, k, v, scale=1.0, heads=8,
                                time_mask=None, pad_mask=None):
    # the inputs of dot product attention is
    # matrix Q: size [B x Tq x H]: translates to T time steps, each timestep has a batch of B vectors of size H
    # or you can also see it as: B sequences, each sequence has T time steps, each time step is a vector size H
    # matrix K: size [B x Tk x H]
    # matrix V: size [B x Tk x H]

    # matmul Q and K: it means that we compute the "content interaction" between Q (queries) and K (keys)
    # for every batch B, Q is a sequence of T query vectors and K and is sequence of T key vectors
    # the matmul will return a matrix in which: M_ij will compute the relationship between Q_i and K_j

    # For masking:
    # For Transformer attention we have to types of mask:
    # pad_mask size [B x T] indicates which element in the sequence is a pad
    # this mask has [B] because each sequence can have a different length -> different padding

    # time_mask size [T x T] indicates that we only look at the left or right hand size of the sequence
    # this mask doesn't have [B] but [T x T] because each element in the sequence has a different padding pattern
    # but this is shared between all sequences in the batch

    # in some poor implementations, people try to combine two masks as one -> [B x T x T]
    # they need to "repeat" the dimension [B] for the time mask
    # in some case, people use "float" mask, it means that the mask value is not Boolean [0, 1] but [0, -100000]
    # so they will "add" the mask to attention scores instead of using masked_fill_
    # the advantage is probably faster (not sure)
    # but it costs a lot more memory (16 - 32 times) more than a boolean

    # Q is transposed from [Tq x B x H] -> [B x Tk x H]
    # K is transposed from [Tq x B x H] -> [B x Tk x H] -> [B x H x Tk]
    matmul_qk = torch.bmm(q.transpose(0, 1), k.transpose(0, 1).transpose(1, 2))
    # the dimension of matmul_qk is [B x T x T] (no more H)
    # this also means that if we have an image (T = W * H) then this matrix will require ( W * H * W * H) storage

    attn_score = matmul_qk * scale

    # mask option: means that sometimes we don't want to look at some positions
    # those positions can be padding or we can dealing with local attention

    if time_mask is not None:
        # attention score size is [BxH, len_q, len_k]
        # it means that for every element in the sentence (len_q), we store a score for every pixel in the input (len_k)
        # and we need to store B times (batch size) but also we need to store additionally H times (head size)
        # because each attention is divided into H heads.
        # print("Using time mask ...")
        # print(time_mask)

        time_mask = time_mask.unsqueeze(0)
        #print("time_mask.shape",time_mask)
        attn_score = attn_score.masked_fill_(time_mask.bool(), -10000)
        #

    if pad_mask is not None:
        bsz = attn_score.size(0) // heads
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        attn_score = attn_score.view(bsz, heads, attn_score.size(1), attn_score.size(2)).masked_fill_(pad_mask, -999999)
        # the reason is that the pad_mask dimension is [bsz x len_k]
        # so by viewing, we can use the "broadcasting" feature of PyTorch to
        # mask [bsz x len_k] -> repeated into [bsz, heads, len_q, len_k]
        # but luckily in this specific problem, we don't ever need pad_mask
        # because in the encoder all images have the same dimension -> so we never need pad_mask

    # [B x Tq x Tk]
    # exp(-99999999) is approx 0
    attn_weights = F.softmax(attn_score, dim=2)

    # final matmul: we multiply the attn_weights with the values
    # (for the weighted sum of values)

    # [B x Tq x Tk] x [B x Tk x H] -> [B x Tq x H]
    output = torch.bmm(attn_weights, v.transpose(0, 1))
    # the output dimension suggest that for every query (in Tq), we have a vector sized H, which is the weighted sum
    # of the values -> this is what we need to for attention
    output = output

    return output


# implements MultiHeadAttention
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, model_size, n_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.model_size = model_size
        self.n_heads = n_heads
        # we divide the model size into different heads
        # 512 / 8 = 64.0  = float
        # 512 // 8 = 64 = integer
        self.head_dim = model_size // n_heads

        self.q_linear = nn.Linear(model_size, model_size)
        self.k_linear = nn.Linear(model_size, model_size)
        self.v_linear = nn.Linear(model_size, model_size)

        self.linear_out = nn.Linear(model_size, model_size)

    # the good thing about images is that all images in the batch have exactly the same size
    # -> we don't need a mask
    # if we batch different sequences and we have to add pads so they have the same size
    # -> we need a mask so that we don't pay attention to the padded positions
    def forward(self, x, impl='fast', pad_mask=None, time_mask=None):
        # x size: [T x B x model_size]
        # T is the number of pixels (W x H) after the efficient net (for example 16 x 16 = 256)
        # B is the batch size
        # model size is our choice
        t, b = x.size(0), x.size(1)
        scale = 1 / math.sqrt(self.head_dim)
        # this is self attention so x is considered as both queries, keys and values
        q = self.q_linear(x)  # [T x B x H]
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = q.view(t, b, self.n_heads, self.head_dim)
        k = k.view(t, b, self.n_heads, self.head_dim)
        v = v.view(t, b, self.n_heads, self.head_dim)

        # this is the slow way
        if impl == 'slow':
            outputs = list()
            for head in range(self.n_heads):
                q_head = q[:, :, head, :]
                k_head = k[:, :, head, :]
                v_head = v[:, :, head, :]

                output_head = scale_dot_product_attention(q_head, k_head, v_head, scale=scale,
                                                          heads=self.n_heads, pad_mask=pad_mask, time_mask=time_mask)
                outputs.append(output_head)

            output = torch.cat(outputs, dim=-1).view(b, t, self.model_size)  # [T x B*head x head_dim]
            output = output.transpose(0, 1).contiguous()
            output = self.linear_out(output)

            return output

        elif impl == 'fast':

            q = q.view(t, b * self.n_heads, self.head_dim)
            k = k.view(t, b * self.n_heads, self.head_dim)
            v = v.view(t, b * self.n_heads, self.head_dim)

            output = scale_dot_product_attention(q, k, v, scale=scale, heads=self.n_heads,
                                                 pad_mask=pad_mask, time_mask=time_mask)
            output = output.transpose(0, 1).contiguous().view(t, b, self.model_size)
            output = self.linear_out(output)

            return output


class MultiHeadCrossAttention(MultiHeadSelfAttention):

    def forward(self, x, encoder_output, impl='fast', pad_mask=None):
        # x size: [T x B x model_size]
        # T is the number of pixels (W x H) after the efficient net (for example 16 x 16 = 256)
        # B is the batch size
        # model size is our choice
        t, b = x.size(0), x.size(1)
        len_k = encoder_output.size(0)
        scale = 1 / math.sqrt(self.head_dim)
        # this is self attention so x is considered as both queries, keys and values
        q = self.q_linear(x)  # [T x B x H]
        k = self.k_linear(encoder_output)
        v = self.v_linear(encoder_output)

        q = q.view(t, b, self.n_heads, self.head_dim)
        k = k.view(len_k, b, self.n_heads, self.head_dim)
        v = v.view(len_k, b, self.n_heads, self.head_dim)

        # this is the slow way
        if impl == 'slow':
            outputs = list()
            for head in range(self.n_heads):
                q_head = q[:, :, head, :]
                k_head = k[:, :, head, :]
                v_head = v[:, :, head, :]

                output_head = scale_dot_product_attention(q_head, k_head, v_head, scale=scale, heads=self.n_heads,
                                                          pad_mask=pad_mask, time_mask=None)
                outputs.append(output_head)

            output = torch.cat(outputs, dim=-1).view(b, t, self.model_size)  # [T x B*head x head_dim]
            output = output.transpose(0, 1).contiguous()
            output = self.linear_out(output)

            return output

        elif impl == 'fast':

            q = q.view(t, b * self.n_heads, self.head_dim)
            k = k.view(len_k, b * self.n_heads, self.head_dim)
            v = v.view(len_k, b * self.n_heads, self.head_dim)

            output = scale_dot_product_attention(q, k, v, scale=scale, heads=self.n_heads,
                                                 pad_mask=pad_mask, time_mask=None)
            output = output.transpose(0, 1).contiguous().view(t, b, self.model_size)
            output = self.linear_out(output)

            return output


# implements Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):

    def __init__(self, model_size=512, n_heads=8, dropout=0.1):
        # have to call super init first to initialize the PyTorch module
        # (it means that PyTorch will understand that this is a network part and register the parmaters)
        super(TransformerEncoderLayer, self).__init__()

        # assign attributes to the class instannce (self)
        # it allows the instance (self) to re-access these numbers later if necessary
        self.model_size = model_size
        self.n_heads = n_heads

        # a transformer layer has 3 main components: multihead-self-attention, 2x layer norm and feed-forward neural net

        # the normalization before or after self-attention

        # layer normalization means that the final dimension of the input (x) will be normalized
        # ( the values are substracted by the mean and then divided by sqrt(var(x))
        # in order to make the layer a bit more robust, the normalized(x) is then multiplied with weights and plus bias
        # (the initialized values of weights is 1 and bias is 0) (it means that the layer norm tries
        # to center the input x around 0 and 1

        # batch norm is very similar but it takes the average over the batch dimension
        # (not the channel dimension as in layer norm)
        self.attn_norm = nn.LayerNorm(model_size, eps=1e-05, elementwise_affine=True)

        # layer normalization before or after the feed forward network
        # by default the layer is created in CPU, and then we will copy it to GPU later
        # (via model = model.cuda() or model = model.device("gpu0"))
        # but pytorch allows us to create a layer directly on GPU if we ever need
        self.ffn_norm = nn.LayerNorm(model_size, eps=1e-05, elementwise_affine=True)

        # each layer has two sets of parameters:
        self.fc1 = nn.Linear(self.model_size, self.model_size * 4)
        # the intermediate layer is larger the "model size" -> in some paper, this layer is called memory
        # so larger memory is better
        self.fc2 = nn.Linear(self.model_size * 4, self.model_size)

        # multihead attention
        self.self_attn = MultiHeadSelfAttention(self.model_size, self.n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, impl='fast'):
        # x size should be [T x B x H]

        # first block
        residual = x
        x = self.self_attn(x, impl=impl)
        x = self.dropout(x)  # apply dropout
        x = x + residual  # residual connection
        x = self.attn_norm(x)  # layer norm

        # second block
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)  # apply dropout
        x = x + residual
        x = self.ffn_norm(x)

        return x


# implements Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_size=512, n_heads=8, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # in PyTorch, there are modules (nn.Module) that have paramemters and some don't have parameters
        # for example: nn.Linear, nn.LayerNorm, nn.MultiHeadAttention, or anything that contains these
        # they have to be created during __init__ because:
        # 1. the weights (or parameters) need to be registered by the PyTorch graph manager
        # (it will understand that these are learnable parameters and can be changed)
        # 2. the weights need to be transferred to the device automatically by calling .cuda() (for example)
        # for some other functions such nn.Sigmoid(), nn.GeLU(), nn.ReLU()
        # you can create them during forward pass

        # assign attributes to the class instance (self)
        # it allows the instance (self) to re-access these numbers later if necessary
        self.model_size = model_size
        self.n_heads = n_heads

        # a transformer decoder layer has 3 main components:
        # Masked multihead-self-attentio, multihead-cross-attention, and feed-forward neural net
        # before each component we have layer norm and after each component we have dropout + residual

        # self-attention between decoder states
        self.self_attn_norm = nn.LayerNorm(model_size, eps=1e-05, elementwise_affine=True)
        self.self_attn = MultiHeadSelfAttention(self.model_size, self.n_heads)

        # attn between decoder states and encoder states
        self.cross_attn_norm = nn.LayerNorm(model_size, eps=1e-05, elementwise_affine=True)
        self.cross_attn = MultiHeadCrossAttention(self.model_size, self.n_heads)

        # layer normalization before or after the feed forward network
        # by default the layer is created in CPU, and then we will copy it to GPU later
        # (via model = model.cuda() or model = model.device("gpu0"))
        # but pytorch allows us to create a layer directly on GPU if we ever need
        self.ffn_norm = nn.LayerNorm(model_size, eps=1e-05, elementwise_affine=True)

        # each layer has two sets of parameters:
        self.fc1 = nn.Linear(self.model_size, self.model_size * 4)
        # the intermediate layer is larger the "model size" -> in some paper, this layer is called memory
        # so larger memory is better
        self.fc2 = nn.Linear(self.model_size * 4, self.model_size)

        self.dropout = nn.Dropout(dropout)

    # def forward(self, dec_inputs, enc_outputs,
    #             dec_self_attn_mask, dec_enc_attn_mask, impl='fast'):
    def forward(self, decoder_input, encoder_output, self_attn_mask, impl='fast'):
        """
        dec_input: [tgt_len, batch_size, model_size]
        enc_outputs: [tgt_len, batch_size, model_size]
        self_attn_mask: [tgt_len, tgt_len]
        """

        # here you can write
        # mod = nn.ReLU()
        # x = mod(x)

        # buf if you write:
        # mod = nn.Linear(self.size, self.size)
        # x = mod(x)
        # there are two problems:
        # x and mod can be in different devices (-> error), maybe we need to write
        # x = mod.to(x.device)(x) ( -> slow because we need to copy mod to device every time)
        # more importantly, the parameters in mod are not registered for optim -> never be updated

        x = decoder_input
        residual = x
        x = self.self_attn(x, impl=impl, time_mask=self_attn_mask)
        x = self.dropout(x)
        x = x + residual
        x = self.self_attn_norm(x)

        residual = x
        x = self.cross_attn(x, encoder_output, impl=impl)
        x = self.dropout(x)
        x = x + residual
        x = self.cross_attn_norm(x)

        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)  # apply dropout
        x = x + residual
        x = self.ffn_norm(x)

        return x


if __name__ == '__main__':
    # multihead_attn = MultiHeadSelfAttention(512, 64)

    seq_len = 8
    batch_size = 64
    seq_len_k = 16 * 16

    test_input = torch.randn(seq_len, batch_size, 512)
    test_encoder_out = torch.randn(seq_len_k, batch_size, 512)

    # multihead_attn = multihead_attn.cuda()
    test_input = test_input.cuda()
    test_encoder_out = test_encoder_out.cuda()

    dropout = 0.0
    # with dropout > 0: the outputs of each layer are randomly set to 0
    # and this process is randomly different run by run
    # -> so the same network with dropout will have different  results each run
    transformer_decoder_layer = TransformerDecoderLayer(512, 8, dropout)

    # this command will copy all modules to cuda, including the layer norms (they were initialized in CPU)
    transformer_decoder_layer = transformer_decoder_layer.cuda()
    self_attn_mask = torch.triu(test_input.new_ones(seq_len, seq_len), diagonal=1).byte()

    output = transformer_decoder_layer(test_input, test_encoder_out, self_attn_mask, impl='fast')
    output_slow = transformer_decoder_layer(test_input, test_encoder_out, self_attn_mask, impl='slow')

    print(output - output_slow)

    # output_fast = multihead_attn(test_input, impl='fast')
    # output_slow = multihead_attn(test_input, impl='slow')
    #
    # print(output_fast - output_slow)
    # (output_fast + output_slow).sum().backward()
    #
    # num_iters = 30
    # torch.cuda.profiler.start()
    # torch.cuda.synchronize()
    # start_time = time()
    # for _ in range(num_iters):
    #     output_fast = multihead_attn(test_input, impl='fast')
    #     output_fast.sum().backward()
    #     multihead_attn.zero_grad()
    #
    # torch.cuda.synchronize()
    # stop_time = time()
    # print(F"\nFast Self-Attn time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
    #
    # torch.cuda.profiler.start()
    # torch.cuda.synchronize()
    # start_time = time()
    # for _ in range(num_iters):
    #     output_slow = multihead_attn(test_input, impl='slow')
    #     output_slow.sum().backward()
    #     multihead_attn.zero_grad()
    #
    # torch.cuda.synchronize()
    # stop_time = time()
    # print(F"\nSlow Self-Attn time {(stop_time - start_time) * 1000. / num_iters:.4f} ms")
