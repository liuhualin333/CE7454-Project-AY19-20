# -*- coding: utf-8 -*-

# !pip install torch numpy

# tested with python 3.6

# %matplotlib inline
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import _pickle
import os
import argparse
from utils_JPLM import Batch,\
                       EncoderDecoder,\
                       Encoder,\
                       Decoder,\
                       BahdanauAttention,\
                       Generator
#
# class Batch:
#     def __init__(self, src, trg, pad_index=4789):
#
#         src, src_lengths = src
#
#         self.src = src
#         self.src_lengths = src_lengths
#         self.src_mask = (src != pad_index).unsqueeze(-2)
#         self.nseqs = src.size(0)
#
#         self.trg = None
#         self.trg_y = None
#         self.trg_mask = None
#         self.trg_lengths = None
#         self.ntokens = None
#
#         if trg is not None:
#             trg, trg_lengths = trg
#             self.trg = trg[:, :-1]
#             self.trg_lengths = trg_lengths
#             self.trg_y = trg[:, 1:]
#             self.trg_mask = (self.trg_y != pad_index)
#             self.ntokens = (self.trg_y != pad_index).data.sum().item()
#
#         if torch.cuda.is_available():
#             self.src = self.src.cuda()
#             self.src_mask = self.src_mask.cuda()
#
#             if trg is not None:
#                 self.trg = self.trg.cuda()
#                 self.trg_y = self.trg_y.cuda()
#                 self.trg_mask = self.trg_mask.cuda()
#

class LossPred:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm, input_x, pad_index):
        x = self.generator(x)

        valid = y.contiguous().view(-1) != pad_index
        valid_preds = x.contiguous().view(-1, x.size(-1))[valid]
        valid_trues = y.contiguous().view(-1)[valid]

        loss = self.criterion(valid_preds, valid_trues)

        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        only_mask = input_x.view(-1)[valid].view(-1) == 0
        preds = torch.argmax(valid_preds, dim=-1).view(-1)[only_mask].cpu().numpy().tolist()
        trues = valid_trues.view(-1)[only_mask].cpu().numpy().tolist()

        full_mask = y.contiguous() != pad_index
        full_input = input_x.contiguous()[full_mask].cpu().numpy().tolist()
        full_preds = torch.argmax(x.contiguous(), dim=-1)[full_mask].cpu().numpy().tolist()
        full_trues = y.contiguous()[full_mask].cpu().numpy().tolist()

        end_preds = torch.argmax(valid_preds, dim=-1).cpu().numpy().tolist()
        end_trues = valid_trues.cpu().numpy().tolist()

        return loss.data.item() * norm, preds, trues, full_input, full_preds, full_trues, end_preds, end_trues

#
# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.trg_embed = trg_embed
#         self.generator = generator
#
#     def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
#         """Take in and process masked src and target sequences."""
#         encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
#         return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
#
#     def encode(self, src, src_mask, src_lengths):
#         return self.encoder(self.src_embed(src), src_mask, src_lengths)
#
#     def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
#                decoder_hidden=None):
#         return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
#                             src_mask, trg_mask, hidden=decoder_hidden)
#
#
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.):
#         super(Encoder, self).__init__()
#         self.num_layers = num_layers
#         self.rnn = nn.GRU(input_size, hidden_size, num_layers,
#                           batch_first=True, bidirectional=True, dropout=dropout)
#
#     def forward(self, x, mask, lengths):
#         packed = pack_padded_sequence(x, lengths, batch_first=True)
#         output, final = self.rnn(packed)
#         output, _ = pad_packed_sequence(output, batch_first=True)
#
#         # we need to manually concatenate the final states for both directions
#         fwd_final = final[0:final.size(0):2]
#         bwd_final = final[1:final.size(0):2]
#         final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]
#
#         return output, final
#
#
# class Decoder(nn.Module):
#     def __init__(self, emb_size, hidden_size, attention, num_layers=3, dropout=0.5,
#                  bridge=True):
#         super(Decoder, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.attention = attention
#         self.dropout = dropout
#
#         self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
#                           batch_first=True, dropout=dropout)
#
#         # to initialize from the final encoder state
#         self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
#
#         self.dropout_layer = nn.Dropout(p=dropout)
#         self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
#                                           hidden_size, bias=False)
#
#     def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
#         """Perform a single decoder step (1 word)"""
#
#         # compute context vector using attention mechanism
#         query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
#         context, attn_probs = self.attention(
#             query=query, proj_key=proj_key,
#             value=encoder_hidden, mask=src_mask)
#
#         # update rnn hidden state
#         rnn_input = torch.cat([prev_embed, context], dim=2)
#         output, hidden = self.rnn(rnn_input, hidden)
#
#         pre_output = torch.cat([prev_embed, output, context], dim=2)
#         pre_output = self.dropout_layer(pre_output)
#         pre_output = self.pre_output_layer(pre_output)
#
#         return output, hidden, pre_output
#
#     def forward(self, trg_embed, encoder_hidden, encoder_final,
#                 src_mask, trg_mask, hidden=None, max_len=None):
#         """Unroll the decoder one step at a time."""
#
#         # the maximum number of steps to unroll the RNN
#         if max_len is None:
#             max_len = trg_mask.size(-1)
#
#         # initialize decoder hidden state
#         if hidden is None:
#             hidden = self.init_hidden(encoder_final)
#
#         # pre-compute projected encoder hidden states
#         # (the "keys" for the attention mechanism)
#         # this is only done for efficiency
#         proj_key = self.attention.key_layer(encoder_hidden)
#
#         # here we store all intermediate hidden states and pre-output vectors
#         decoder_states = []
#         pre_output_vectors = []
#
#         # unroll the decoder RNN for max_len steps
#         for i in range(max_len):
#             prev_embed = trg_embed[:, i].unsqueeze(1)
#             output, hidden, pre_output = self.forward_step(
#                 prev_embed, encoder_hidden, src_mask, proj_key, hidden)
#             decoder_states.append(output)
#             pre_output_vectors.append(pre_output)
#
#         decoder_states = torch.cat(decoder_states, dim=1)
#         pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
#         return decoder_states, hidden, pre_output_vectors  # [B, N, D]
#
#     def init_hidden(self, encoder_final):
#         """Returns the initial decoder state,
#         conditioned on the final encoder state."""
#
#         if encoder_final is None:
#             return None  # start with zeros
#
#         return torch.tanh(self.bridge(encoder_final))
#
#
# class BahdanauAttention(nn.Module):
#
#     def __init__(self, hidden_size, key_size=None, query_size=None):
#         super(BahdanauAttention, self).__init__()
#
#         # We assume a bi-directional encoder so key_size is 2*hidden_size
#         key_size = 2 * hidden_size if key_size is None else key_size
#         query_size = hidden_size if query_size is None else query_size
#
#         self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
#         self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
#         self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
#
#         # to store attention scores
#         self.alphas = None
#
#     def forward(self, query=None, proj_key=None, value=None, mask=None):
#         assert mask is not None, "mask is required"
#
#         # We first project the query (the decoder state).
#         # The projected keys (the encoder states) were already pre-computated.
#         query = self.query_layer(query)
#
#         # Calculate scores.
#         scores = self.energy_layer(torch.tanh(query + proj_key))
#         scores = scores.squeeze(2).unsqueeze(1)
#
#         # Mask out invalid positions.
#         # The mask marks valid positions so we invert it using `mask & 0`.
#         scores.data.masked_fill_(mask == 0, -float('inf'))
#
#         # Turn scores to probabilities.
#         alphas = F.softmax(scores, dim=-1)
#         self.alphas = alphas
#
#         # The context vector is the weighted sum of the values.
#         context = torch.bmm(alphas, value)
#
#         # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
#         return context, alphas
#
#
# class Generator(nn.Module):
#     def __init__(self, hidden_size, vocab_size):
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
#
#     def forward(self, x):
#         return F.log_softmax(self.proj(x), dim=-1)
#

def run_epoch(data_iter, model, loss_compute, positive_classes, mask_index, pad_index):
    total_loss = 0
    print_tokens = 0

    y_preds = []
    y_trues = []
    full_preds = []
    full_trues = []
    e_preds = []
    e_trues = []
    xs = []

    for i, batch in tqdm(enumerate(data_iter, 1)):
        if batch.nseqs != args.batch_size: continue

        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss, preds, trues, full_input, full_pred, full_true, end_preds, end_trues = loss_compute(pre_output,
                                                                                                  batch.trg_y,
                                                                                                  batch.nseqs,
                                                                                                  batch.src,
                                                                                                  pad_index)
        total_loss += loss
        print_tokens += batch.ntokens

        y_trues.extend(trues)
        y_preds.extend(preds)
        e_trues.extend(end_trues)
        e_preds.extend(end_preds)
        full_trues.append(full_true)
        full_preds.append(full_pred)
        xs.append(full_input)

    y_ensemble = []
    y_ensembles = []
    full_preds_flat = []
    for y_in, y_out, y_true in zip(xs, full_preds, full_trues):
        ye = [w if w!=mask_index else y_out[i] for i,w in enumerate(y_in)]
        y_ensemble.extend(ye)
        y_ensembles.append(ye)
        full_preds_flat.extend([w for w in y_true])

    metrics = {
        'LM accuracy': accuracy_score(e_trues, e_preds),
        'LM f1': f1_score(e_trues, e_preds, labels=positive_classes, average='micro'),
        'LM precision': precision_score(e_trues, e_preds, labels=positive_classes, average='micro'),
        'LM recall': recall_score(e_trues, e_preds, labels=positive_classes, average='micro'),
        '(masked) accuracy': accuracy_score(y_trues, y_preds),
        '(masked) f1': f1_score(y_trues, y_preds, labels=positive_classes, average='micro'),
        '(masked) precision': precision_score(y_trues, y_preds, labels=positive_classes, average='micro'),
        '(masked) recall': recall_score(y_trues, y_preds, labels=positive_classes, average='micro'),
        '(ensemble end-to-end) accuracy': accuracy_score(full_preds_flat, y_ensemble),
        '(ensemble end-to-end) f1': f1_score(full_preds_flat, y_ensemble, labels=positive_classes, average='micro'),
        '(ensemble end-to-end) precision': precision_score(full_preds_flat, y_ensemble, labels=positive_classes, average='micro'),
        '(ensemble end-to-end) recall': recall_score(full_preds_flat, y_ensemble, labels=positive_classes, average='micro')
    }

    return metrics, xs, full_preds, full_trues, y_ensembles


def data_load(settype, batch_size=5, pad_index=4789, sos_index=4788, max_seq_length=190, pp=1):
    _, _, input_seqs, output_seqs = data[settype]

    for i in range(0, len(input_seqs) * pp, batch_size):
        this_input_seqs = input_seqs[i:i + batch_size]
        this_output_seqs = [[sos_index] + seq for seq in output_seqs[i:i + batch_size]]
        to_pad = max_seq_length + 1 - len(this_output_seqs[0])

        try:
            outputs = torch.LongTensor(this_output_seqs)
        except:  # a fix for real eval set
            max_current_seq = 0
            for s in this_output_seqs:
                if len(s) > max_current_seq: max_current_seq = len(s)
            for i, s in enumerate(this_output_seqs):
                this_input_seqs[i] += [pad_index] * (max_current_seq - len(s))
                this_output_seqs[i] += [pad_index] * (max_current_seq - len(s))
                to_pad = max_seq_length + 1 - max_current_seq
            outputs = torch.LongTensor(this_output_seqs)
        outputs = torch.nn.functional.pad(outputs, (0, to_pad), "constant", pad_index)
        outputs = outputs.cuda() if torch.cuda.is_available() else outputs
        trg = outputs

        inputs = torch.LongTensor(this_input_seqs)
        inputs = torch.nn.functional.pad(inputs, (0, to_pad), "constant", pad_index)
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        src = inputs

        src_lengths = [inputs.shape[1]] * batch_size
        trg_lengths = [outputs.shape[1]] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)


if __name__ == '__main__':
    # torch.nn.Module.dump_patches = True

    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_pkl", default='train_corpus.pkl', type=str)
    parser.add_argument("--eval_pkl", default='test_corpus.pkl', type=str)
    parser.add_argument("--output_dir", default='eval_output', type=str)
    parser.add_argument("--dict_pkl", default='kaggle_u2c_and_c2u.pkl', type=str)
    parser.add_argument("--model_checkpoint_pkl", required=True, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--partially", default=1, type=int, help="Use only part of eval set data, to speed up debugging. E.g. 2 is using only half, 3 is using one third")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print('\nusing', args.eval_pkl)

    data = {}
    # data['train'] = _pickle.load(open(args.train_pkl, 'rb'))
    data['eval'] = _pickle.load(open(args.eval_pkl, 'rb'))

    vocab_list, vocab_dict, _, _ = data['eval']

    u2c, c2u = _pickle.load(open(args.dict_pkl, 'rb'))

    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    vocab_size = len(vocab_list) + 2
    positive_classes = list(range(1, len(vocab_list)))
    mask_index = vocab_dict['[MASK]']
    pad_index = vocab_size - 1


        # for args.model_checkpoint_pkl in os.listdir(args.output_dir):
        #     if args.model_checkpoint_pkl[-4:] != '.pkl': continue
        #     pkl_path = os.path.join(args.output_dir, args.model_checkpoint_pkl)
        #     print('\n', pkl_path)
        #     model = torch.load(pkl_path)

    model = torch.load(args.model_checkpoint_pkl)
    eval_data = list(data_load('eval', batch_size=args.batch_size))[::args.partially]

    if torch.cuda.is_available():
        print('using cuda')
        model.cuda()

    model.eval()

    with torch.no_grad():
        metrics, xs, y_preds, y_trues, y_ensembles = run_epoch(eval_data, model,
                                                               LossPred(model.generator, criterion, None),
                                                               positive_classes,
                                                               mask_index,
                                                               pad_index)
        input_x = ["".join([u2c[vocab_list[i]] if i != mask_index else 'M' for i in seq]) for seq in xs]
        predict = ["".join([u2c[vocab_list[i]] if i != mask_index else 'M' for i in seq]) for seq in y_preds]
        groundt = ["".join([u2c[vocab_list[i]] if i != mask_index else 'M' for i in seq]) for seq in y_trues]
        ensemble = ["".join([u2c[vocab_list[i]] if i != mask_index else 'M' for i in seq]) for seq in y_ensembles]
        with open(args.model_checkpoint_pkl+args.eval_pkl+'_eval_results_correct.txt', 'a+') as f:
            for x,p,g,e in zip(input_x, predict, groundt, ensemble):
                if 'M' in x:
                    has_true = False
                    for xt,et,gt in zip(x,e,g):
                        if xt=='M' and et==gt:
                            has_true = True
                    if has_true:
                        f.write('\n'.join([x.replace('M', '[MASK]'),
                                           p,
                                           e.replace('M', '[MASK]'),
                                           g.replace('M', '[MASK]'),
                                           '\n']))
        with open(args.model_checkpoint_pkl+args.eval_pkl+'_eval_results_wrong.txt', 'a+') as f:
            for x,p,g,e in zip(input_x, predict, groundt, ensemble):
                if e!=g:
                    f.write('\n'.join([x.replace('M', '[MASK]'),
                                       p,
                                       e.replace('M', '[MASK]'),
                                       g.replace('M', '[MASK]'),
                                       '\n']))
        for key, value in metrics.items():
            print("Evaluation {}: {}".format(key, value))
        print('results saved to:\n\t',
              args.model_checkpoint_pkl+args.eval_pkl+'_eval_results_correct.txt\n',
              args.model_checkpoint_pkl+args.eval_pkl+'_eval_results_wrong.txt')