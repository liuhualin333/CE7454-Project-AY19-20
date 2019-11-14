# -*- coding: utf-8 -*-

# !pip install torch numpy

# tested with python 3.6

# %matplotlib inline
import argparse
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import _pickle
import logging
import os
import sys
from utils_JPLM import Batch,\
                       EncoderDecoder,\
                       Encoder,\
                       Decoder,\
                       BahdanauAttention,\
                       Generator

logger = logging.getLogger(__name__)


class LossPred:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm, src, mask_index, pad_index, finetune):
        x = self.generator(x)

        if finetune:
            # valid = src[src != pad_index].view(-1) == mask_index
            valid = src.view(-1) == mask_index
            valid_preds = x.contiguous().view(-1, x.size(-1))[valid]
            valid_trues = y.contiguous().view(-1)[valid]
        else:
            valid = y.contiguous().view(-1) != pad_index
            valid_preds = x.contiguous().view(-1, x.size(-1))[valid]
            valid_trues = y.contiguous().view(-1)[valid]

        loss = self.criterion(valid_preds, valid_trues)

        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

	valid = y.contiguous().view(-1) != pad_index
        valid_preds = x.contiguous().view(-1, x.size(-1))[valid]
        valid_trues = y.contiguous().view(-1)[valid]

        preds = torch.argmax(valid_preds, dim=-1).cpu().numpy().tolist()
        trues = valid_trues.cpu().numpy().tolist()

        return loss.data.item() * norm, preds, trues


def eval_metrics(y_trues, y_preds, positive_classes, average, logger_prefix=None):
    metrics = {
        'accuracy': accuracy_score(y_trues, y_preds),
        'f1': f1_score(y_trues, y_preds, labels=positive_classes, average=average),
        'precision': precision_score(y_trues, y_preds, labels=positive_classes, average=average),
        'recall': recall_score(y_trues, y_preds, labels=positive_classes, average=average)
    }

    for key, value in metrics.items():
        logger.info(logger_prefix + " metrics {}: {}".format(key, value))

    return metrics


def run_epoch(args, data_iter, model, loss_compute, logger_prefix):

    start = time.time()

    total_loss = 0

    y_preds = []
    y_trues = []

    for i, batch in tqdm(enumerate(data_iter, 1)):
        if args.use_cuda:
            batch.src.cuda()
            batch.trg.cuda()
        if model.training and batch.nseqs != args.batch_size: continue

        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss, preds, trues = loss_compute(pre_output,
                                          batch.trg_y,
                                          batch.nseqs,
                                          batch.src,
                                          mask_index=args.mask_index,
                                          pad_index=args.pad_index,
                                          finetune=args.finetune_mask)
        total_loss += loss

        y_trues.extend(trues)
        y_preds.extend(preds)

        if model.training:
            if i % args.logging_steps == 0:
                logger.info("Epoch Step: %d Loss: %f" % (i, loss / batch.nseqs))
                start = time.time()

            if args.eval_during_training and i % args.eval_steps == 0:
                ms = eval_metrics(y_trues,
                                  y_preds,
                                  args.positive_classes,
                                  args.metrics_average,
                                  logger_prefix)

            if i % args.checkpoint_steps == 0:
                torch.save(model, 'model_saved_train_cache_{}_data{}_{}.pkl'.format(start, len(data_iter), i))

    metrics = eval_metrics(y_trues, y_preds, args.positive_classes, args.metrics_average, logger_prefix)

    return metrics


def data_load(args, data, batch_size=None):
    if not batch_size:
        batch_size = args.batch_size
    _, _, raw_input_seqs, raw_output_seqs = data
    if args.finetune_mask:
        input_seqs, output_seqs = [], []
        for seqi, seq in enumerate(raw_input_seqs):
            if args.mask_index in seq:
                input_seqs.append(seq)
                output_seqs.append(raw_output_seqs[seqi])
    else:
        input_seqs, output_seqs = raw_input_seqs, raw_output_seqs

    if args.partially>1 and args.use_cuda:
        logger.info("Limited GPU space. Only using 1/{} of the data".format(args.partially))
    for i in range(0, len(input_seqs), batch_size*args.partially):
        this_input_seqs = input_seqs[i:i + batch_size]
        this_output_seqs = [[args.sos_index] + seq for seq in output_seqs[i:i + batch_size]]
        to_pad = args.max_seq_length + 1 - len(this_output_seqs[0])

        try:
            outputs = torch.LongTensor(this_output_seqs)
        except:  # a fix for real eval set
            max_current_seq = 0
            for s in this_output_seqs:
                if len(s) > max_current_seq: max_current_seq = len(s)
            for i, s in enumerate(this_output_seqs):
                this_input_seqs[i] += [args.pad_index] * (max_current_seq - len(s))
                this_output_seqs[i] += [args.pad_index] * (max_current_seq - len(s))
                to_pad = args.max_seq_length + 1 - max_current_seq
            outputs = torch.LongTensor(this_output_seqs)
        outputs = torch.nn.functional.pad(outputs, (0, to_pad), "constant", args.pad_index)
        trg = outputs

        inputs = torch.LongTensor(this_input_seqs)
        inputs = torch.nn.functional.pad(inputs, (0, to_pad), "constant", args.pad_index)
        src = inputs

        src_lengths = [inputs.shape[1]] * batch_size
        trg_lengths = [outputs.shape[1]] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=args.pad_index)


def train_JPLM(args, data):
    train_data = list(data_load(args, data['train']))
    eval_data = list(data_load(args, data['eval'], 1))

    if args.load_model_from_pkl:
        model = torch.load(args.load_model_from_pkl)
        logger.info('model loaded from ' + args.load_model_from_pkl)
    else:
        if args.attention_class == 'Bahdanau':
            attention = BahdanauAttention(args.hidden_size)
        # else:
        #     attention = LuongAttention(args.hidden_size)
        model = EncoderDecoder(
            Encoder(args.emb_size,
                    args.hidden_size,
                    num_layers=args.num_layers,
                    dropout=args.dropout),
            Decoder(args.emb_size,
                    args.hidden_size,
                    attention,
                    num_layers=args.num_layers,
                    dropout=args.dropout),
            nn.Embedding(args.vocab_size, args.emb_size),
            nn.Embedding(args.vocab_size, args.emb_size),
            Generator(args.hidden_size, args.vocab_size))

    if args.use_cuda:
        model.cuda()

    logger.info('total train sample size: {} \t total eval sample size: {}'.format(len(train_data)*args.batch_size,
                                                                                   len(eval_data)))

    args.positive_classes = list(range(1, args.vocab_size))

    for epoch in range(args.num_epoch):
        logger.info("Epoch %d" % epoch)

        if args.train_pkl.split('/')[-1]!='train_corpus':
            args.rotating = False
        else:
            args.rotating = True
        for idx_rotate in range(round(len(train_data)/args.num_rotate)) if args.rotating else [1]:
            # train
            model.train()
            data = train_data[idx_rotate::args.num_rotate] if args.rotating else train_data
            _ = run_epoch(args, data, model,
                                LossPred(model.generator,
                                         criterion=nn.NLLLoss(reduction="sum", ignore_index=0),
                                         opt=torch.optim.Adam(model.parameters(), lr=args.lr)),
                                logger_prefix='Train')

            # evaluate
            model.eval()
            with torch.no_grad():
                metrics = run_epoch(args, eval_data, model,
                                          LossPred(model.generator, nn.NLLLoss(), None),
                                          logger_prefix='Evaluation')
                if metrics['f1'] > args.f1_save_threshold:
                    args.f1_save_threshold = metrics['f1']
                    output_file = os.path.join(args.output_dir, 'model_saved_{}.pkl'.format(time.time()))
                    torch.save(model, output_file)
                    logger.info('model saved to %s' % output_file)


if __name__ == '__main__':
    # torch.nn.Module.dump_patches = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pkl", default='train_corpus', type=str,
                        help="dataset for training; no need to add .pkl at the end")
    parser.add_argument("--eval_pkl", default='test_corpus', type=str,
                        help="dataset for evaluation; no need to add .pkl at the end")
    parser.add_argument("--load_model_from_pkl", default=None, type=str,
                        help="where you place .pkl file; please add .pkl at the end")
    parser.add_argument("--partially", default=1, type=int, help="for fast debugging, only use part of the data")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--emb_size", default=64, type=int)
    parser.add_argument("--num_epoch", default=15, type=int)
    parser.add_argument("--num_rotate", default=1, type=int)
    parser.add_argument("--attention_class", default='Bahdanau', type=str)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=190, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--metrics_average", default='micro', type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--lr", default=0.0003, type=float,
                        help="learning rate, default 0.0003")
    parser.add_argument("--f1_save_threshold", default=0.5, type=float)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--checkpoint_steps", default=600, type=int)
    parser.add_argument("--eval_during_training", default=True, type=bool)
    parser.add_argument("--eval_steps", default=300, type=int)
    parser.add_argument("--use_cpu", action='store_true')
    parser.add_argument("--finetune_mask", default=False, type=bool,
                        help="default false; when true, only masked results are considered for loss computation. (for finetune)")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    args.use_cuda = torch.cuda.is_available() if not args.use_cpu else False
    logger.info('device: {}'.format(torch.device("cuda" if args.use_cuda else "cpu")))

    data = {
        'train': _pickle.load(open(args.train_pkl+'.pkl', 'rb')),
        'eval' : _pickle.load(open(args.eval_pkl+'.pkl', 'rb'))
    }

    vocab_list, vocab_dict, _, _ = data['eval']
    args.vocab_size = len(vocab_list) + 2
    args.mask_index = vocab_dict['[MASK]']
    args.sos_index = len(vocab_list)
    args.pad_index = args.sos_index + 1
    args.positive_classes = list(range(1, args.vocab_size))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # train
    train_JPLM(args, data)

    print('DONE!')
