import os
import sys
import argparse
import time
import random

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from sru import *
import checklist_data_gen
import dataloader
import modules
from sklearn.metrics import classification_report, accuracy_score


class Model(nn.Module):

    def __init__(self,
                 embedding,
                 hidden_size=150,
                 depth=1,
                 dropout=0.3,
                 cnn=False,
                 nclasses=2):
        super(Model, self).__init__()
        self.cnn = cnn
        self.drop = nn.Dropout(dropout)
        self.emb_layer = modules.EmbeddingLayer(
            embs=dataloader.load_embedding(embedding))
        self.word2id = self.emb_layer.word2id

        self.id2word = {self.word2id[wd]: wd for wd in self.word2id}

        if cnn:
            print('using cnn model .....')
            self.encoder = modules.CNN_Text(self.emb_layer.n_d,
                                            widths=[3, 4, 5],
                                            filters=hidden_size)
            d_out = 3 * hidden_size
        else:
            print('using LSTM model .....')
            self.encoder = nn.LSTM(
                self.emb_layer.n_d,
                hidden_size // 2,
                depth,
                dropout=dropout,
                # batch_first=True,
                bidirectional=True)
            d_out = hidden_size
        # else:
        #     self.encoder = SRU(
        #         emb_layer.n_d,
        #         args.d,
        #         args.depth,
        #         dropout = args.dropout,
        #     )
        #     d_out = args.d
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        if self.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        emb = self.drop(emb)

        if self.cnn:
            output = self.encoder(emb)
        else:
            output, hidden = self.encoder(emb)
            # output = output[-1]
            output = torch.max(output, dim=0)[0].squeeze()

        output = self.drop(output)
        return self.out(output)

    def text_pred(self, text, batch_size=32):
        batches_x = dataloader.create_batches_x(
            text,
            batch_size,  ##TODO
            self.word2id)
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                if self.cnn:
                    x = x.t()
                emb = self.emb_layer(x)

                if self.cnn:
                    output = self.encoder(emb)
                else:
                    output, hidden = self.encoder(emb)
                    # output = output[-1]
                    output = torch.max(output, dim=0)[0]

                outs.append(F.softmax(self.out(output), dim=-1))

        return torch.cat(outs, dim=0)


def eval_model(niter,
               model,
               input_x,
               input_y,
               reports=None,
               raw_test_x=None,
               raw_test_y=None):
    model.eval()
    # N = len(valid_x)
    # criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0.
    all_pred_ids = []
    all_true_ids = []
    batch_idx = 0
    # total_loss = 0.0
    with torch.no_grad():
        for x, y in zip(input_x, input_y):
            x, y = Variable(x, volatile=True), Variable(y)
            output = model(x)
            # loss = criterion(output, y)
            # total_loss += loss.item()*x.size(1)
            pred = output.data.max(1)[1]
            if reports == 'all':
                all_pred_ids += list(pred.cpu().data.numpy())
                all_true_ids += list(y.cpu().data.numpy())

            if raw_test_x != None:
                r_data = raw_test_x
                r_y = raw_test_y
                r_pred = pred.cpu().data.numpy()
                for idx, x_item in enumerate(r_pred):
                    print(batch_idx * 32 + idx, 'context:',
                          r_data[batch_idx * 32 + idx], 'y:',
                          r_y[batch_idx * 32 + idx], 'pred:', r_pred[idx])

            correct += pred.eq(y.data).cpu().sum()
            cnt += y.numel()
            batch_idx += 1
    if reports == 'all':
        print('eval acc:',
              accuracy_score(y_true=all_true_ids, y_pred=all_pred_ids))
        print(classification_report(y_true=all_true_ids, y_pred=all_pred_ids))
    model.train()
    return correct.item() / cnt


def train_model(epoch, model, optimizer, train_x, train_y, test_x, test_y,
                best_test, save_path):

    model.train()
    niter = epoch * len(train_x)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for x, y in zip(train_x, train_y):
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    test_acc = eval_model(niter, model, test_x, test_y)

    sys.stdout.write(
        "Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(
            epoch, niter, optimizer.param_groups[0]['lr'], loss.item(),
            test_acc))

    if test_acc > best_test:
        best_test = test_acc
        if save_path:
            print("save modeling ...")
            torch.save(model.state_dict(), save_path)
        eval_model(niter, model, test_x, test_y, reports='all')
    sys.stdout.write("\n")
    return best_test


def save_data(data, labels, path, type='train'):
    with open(os.path.join(path, type + '.txt'), 'w') as ofile:
        for text, label in zip(data, labels):
            ofile.write('{} {}\n'.format(label, ' '.join(text)))


def main(args):
    if args.dataset == 'mr':
        train_x, train_y = dataloader.read_corpus('data/mr_all/train.csv',
                                                  clean=False,
                                                  MR=True,
                                                  shuffle=True)
        test_x, test_y = dataloader.read_corpus('data/mr_all/dev.csv',
                                                clean=False,
                                                MR=True,
                                                shuffle=False)
        if args.noise_type == 'del':
            test_x, test_y = dataloader.read_corpus(
                'data/mr_all/dev.csv',
                clean=False,
                MR=True,
                shuffle=False,
                is_adv=True,
                adv_dict=json.load(open('data/mr_all/naive_adv.dict')))
        elif args.noise_type == 'typos':
            test_x = checklist_data_gen.add_typos(test_x)
        elif args.noise_type == 'inv':
            test_x = checklist_data_gen.add_INV(test_x)
        elif args.noise_type == 'add':
            test_x, test_y = dataloader.read_corpus('data/mr_all/dev.csv',
                                                    clean=False,
                                                    MR=True,
                                                    shuffle=False,
                                                    noise_type='add')
    elif args.dataset == 'ag_news':
        train_x, train_y = dataloader.read_corpus('data/ag_news_all/train.csv',
                                                  csvf=True,
                                                  clean=True,
                                                  del_stop=True,
                                                  MR=False,
                                                  shuffle=True)
        test_x, test_y = dataloader.read_corpus('data/ag_news_all/test.csv',
                                                csvf=True,
                                                clean=True,
                                                del_stop=True,
                                                MR=False,
                                                shuffle=False)
        if args.noise_type == 'del':
            test_x, test_y = dataloader.read_corpus(
                'data/ag_news_all/test.csv',
                csvf=True,
                clean=True,
                MR=False,
                shuffle=False,
                is_adv=True,
                adv_dict=json.load(open('data/ag_news_all/naive_adv.dict')))
        elif args.noise_type == 'typos':
            test_x = checklist_data_gen.add_typos(test_x)
        elif args.noise_type == 'inv':
            test_x = checklist_data_gen.add_INV(test_x)
        elif args.noise_type == 'add':
            test_x, test_y = dataloader.read_corpus('data/ag_news_all/test.csv',
                                                    csvf=True,
                                                    clean=True,
                                                    del_stop=True,
                                                    MR=False,
                                                    shuffle=False,
                                                    noise_type='add')
    elif args.dataset == 'weibo2018':
        train_x, train_y = dataloader.read_weibo_corpus(
            'data/weibo2018/train.txt', shuffle=True)
        test_x, test_y = dataloader.read_weibo_corpus('data/weibo2018/test.txt',
                                                      shuffle=False)

        if args.noise_type == 'del':
            test_x, test_y = dataloader.read_weibo_corpus(
                'data/weibo2018/test.txt',
                shuffle=False,
                is_adv=args.adv,
                adv_dict=json.load(
                    open('data/' + args.dataset + '/naive_adv.dict')))
        elif args.noise_type == 'typos':
            test_x = checklist_data_gen.ch_tongyi_word(test_x)
        elif args.noise_type == 'inv':
            test_x = checklist_data_gen.add_ch_INV(test_x)
        elif args.noise_type == 'add':
            test_x, test_y = dataloader.read_weibo_corpus(
                'data/weibo2018/test.txt', shuffle=False, noise_type='add')
    elif args.dataset == "tnews":
        train_x, train_y = dataloader.read_tnews_corpus(
            data_path="data/tnews/train.json",
            label_path="data/tnews/labels.json",
            shuffle=True)
        test_x, test_y = dataloader.read_tnews_corpus(
            data_path="data/tnews/dev.json",
            label_path="data/tnews/labels.json",
            shuffle=False)

        if args.noise_type == 'del':
            test_x, test_y = dataloader.read_tnews_corpus(
                data_path="data/tnews/dev.json",
                label_path="data/tnews/labels.json",
                is_adv=args.adv,
                adv_dict=json.load(
                    open('data/' + args.dataset + '/naive_adv.dict')),
                shuffle=False)
        elif args.noise_type == 'typos':
            test_x = checklist_data_gen.ch_tongyi_word(test_x)
        elif args.noise_type == 'inv':
            test_x = checklist_data_gen.add_ch_INV(test_x)
        elif args.noise_type == 'add':
            test_x, test_y = dataloader.read_tnews_corpus(
                data_path="data/tnews/dev.json",
                label_path="data/tnews/labels.json",
                shuffle=False,
                noise_type='add')
    elif args.dataset == 'imdb':
        train_x, train_y = dataloader.read_corpus(os.path.join(
            '/data/nlp/datasets/imdb', 'train_tok.csv'),
                                                  clean=False,
                                                  MR=True,
                                                  shuffle=True)
        test_x, test_y = dataloader.read_corpus(os.path.join(
            '/data/nlp/datasets/imdb', 'test_tok.csv'),
                                                clean=False,
                                                MR=True,
                                                shuffle=True)
    else:
        train_x, train_y = dataloader.read_corpus('proj/to_di/data/{}/'
                                                  'train_tok.csv'.format(
                                                      args.dataset),
                                                  clean=False,
                                                  MR=False,
                                                  shuffle=True)
        test_x, test_y = dataloader.read_corpus('proj/to_di/data/{}/'
                                                'test_tok.csv'.format(
                                                    args.dataset),
                                                clean=False,
                                                MR=False,
                                                shuffle=True)

    nclasses = max(train_y) + 1
    # elif args.dataset == 'subj':
    #     data, label = dataloader.read_SUBJ(args.path)
    # elif args.dataset == 'cr':
    #     data, label = dataloader.read_CR(args.path)
    # elif args.dataset == 'mpqa':
    #     data, label = dataloader.read_MPQA(args.path)
    # elif args.dataset == 'trec':
    #     train_x, train_y, test_x, test_y = dataloader.read_TREC(args.path)
    #     data = train_x + test_x
    #     label = None
    # elif args.dataset == 'sst':
    #     train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path)
    #     data = train_x + valid_x + test_x
    #     label = None
    # else:
    #     raise Exception("unknown dataset: {}".format(args.dataset))

    # if args.dataset == 'trec':

    # elif args.dataset != 'sst':
    #     train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.cv_split(
    #         data, label,
    #         nfold = 10,
    #         test_id = args.cv
    #     )

    model = Model(args.embedding, args.d, args.depth, args.dropout, args.cnn,
                  nclasses).cuda()
    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)

    train_x, train_y = dataloader.create_batches(
        train_x,
        train_y,
        args.batch_size,
        model.word2id,
    )
    # valid_x, valid_y = dataloader.create_batches(
    #     valid_x, valid_y,
    #     args.batch_size,
    #     emb_layer.word2id,
    # )
    raw_test_x, raw_test_y = test_x, test_y
    test_x, test_y = dataloader.create_batches(
        test_x,
        test_y,
        args.batch_size,
        model.word2id,
    )
    if args.test or args.noise_type != 'none':
        checkpoint = torch.load(args.save_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
        eval_model(1,
                   model,
                   test_x,
                   test_y,
                   reports='all',
                   raw_test_x=raw_test_x,
                   raw_test_y=raw_test_y)
        exit(0)

    best_test = 0
    # test_err = 1e+8
    for epoch in range(args.max_epoch):
        best_test = train_model(
            epoch,
            model,
            optimizer,
            train_x,
            train_y,
            # valid_x, valid_y,
            test_x,
            test_y,
            best_test,
            args.save_path)
        if args.lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

    # sys.stdout.write("best_valid: {:.6f}\n".format(
    #     best_valid
    # ))
    sys.stdout.write("test_err: {:.6f}\n".format(best_test))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn",
                           action='store_true',
                           help="whether to use cnn")
    argparser.add_argument("--lstm",
                           action='store_true',
                           help="whether to use lstm")
    argparser.add_argument("--noise_type",
                           type=str,
                           default="none",
                           help="del, typos, add, adv")
    argparser.add_argument("--test",
                           action='store_true',
                           help="whether just test")
    argparser.add_argument("--dataset",
                           type=str,
                           default="mr",
                           help="which dataset")
    argparser.add_argument("--embedding",
                           type=str,
                           required=True,
                           help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=70)
    argparser.add_argument("--d", type=int, default=150)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--save_path", type=str, default='')
    argparser.add_argument("--save_data_split",
                           action='store_true',
                           help="whether to save train/test split")
    argparser.add_argument("--gpu_id", type=int, default=0)

    args = argparser.parse_args()
    # args.save_path = os.path.join(args.save_path, args.dataset)
    print(args)
    torch.cuda.set_device(args.gpu_id)
    main(args)
