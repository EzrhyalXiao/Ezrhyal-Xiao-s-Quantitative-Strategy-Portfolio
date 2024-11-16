import sys
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


from qlib.log import get_module_logger
from qlib.utils import get_or_create_path

from qlib.model.base import Model
from qlib.model.utils import ConcatDataset

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter

class iTransformerModel(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.linear = nn.Linear(configs.d_feat, 1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = self.linear(dec_out)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class iTransformer(Model):
    def __init__(
        self,
        d_feat,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        pred_len,
        embed_type="fixed",
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=2048,
        dropout=0.1,
        attn="prob",
        embed="fixed",
        freq="d",
        activation="gelu",
        inverse = False,
        class_strategy='projection',
        channel_independence = False,
        output_attention=False,
        distil=True,
        mix=True,
        use_norm=True,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        factor=5,
        padding=0,
        padding_var=None,
        seq_len_out=None,
        label_len=None,
        pred_len_out=None,
        num_workers=0,
        batch_size=32,
        eval_batch_size=32,
        itr=2,
        train_epochs=10,
        early_stop=5,
        lr=0.0001,
        warmup_prop=0.1,
        weight_decay=0.0001,
        gradient_clip_val=0.5,
        features = 'MS',
        use_amp=False,
        loss="mse",
        l1=0,
        l2=0,
        des = 'Exp',
        metric="mse",
        optimizer="adam",
        scheduler="linear",
        seed=None,
        verbose=True,
        GPU=0,
    ):

        self.logger = get_module_logger("PatchTST")

        self.d_feat = d_feat
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed_type = embed_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.attn = attn
        self.embed = embed
        self.freq = freq
        self.inverse = inverse
        self.activation = activation
        self.class_strategy = class_strategy
        self.channel_independence = channel_independence
        self.output_attention = output_attention
        self.distil = distil
        self.mix = mix
        self.use_norm = use_norm
        self.device = device
        self.factor = factor
        self.padding = padding
        self.padding_var = padding_var
        self.seq_len_out = seq_len_out
        self.label_len = label_len
        self.pred_len_out = pred_len_out
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.itr = itr
        self.n_epochs = train_epochs
        self.early_stop = early_stop
        self.lr = lr
        self.warmup_prop = warmup_prop
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.use_amp = use_amp
        self.features = features
        self.loss = loss
        self.l1 = l1
        self.l2 = l2
        self.des = des
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.seed = seed
        self.verbose = verbose

        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = iTransformerModel(
            self,
        ).to(self.device)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask], weight=None)

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, train_loader):
        self.model.train()

        scores = []
        losses = []

        for _, (input_x, weight_x) in enumerate(tqdm(train_loader, mininterval=2)):

            seq_x, seq_y, emb_mark = input_x

            seq_x = seq_x.to(torch.float32)
            seq_y = seq_y.to(torch.float32)

            # seq_x = torch.concat([seq_x, seq_y], axis=2)
            seq_x = seq_x[:, :-self.pred_len, :]

            dec_x = torch.zeros_like(seq_y[:, -self.pred_len:, :]).float()
            dec_x = torch.cat([seq_y[:, :self.seq_len, :], dec_x], dim=1)


            seq_x_mark = emb_mark[:, :-self.pred_len, :]
            seq_y_mark = emb_mark

            self.train_optimizer.zero_grad()
            if self.device.type == 'cuda':
                seq_x = seq_x.float().cuda()
                seq_y = seq_y.float().cuda()
                dec_x = dec_x.float().cuda()
                seq_x_mark = seq_x_mark.float().cuda()
                seq_y_mark = seq_y_mark.float().cuda()

            pred = self.model(seq_x,seq_x_mark, dec_x ,seq_y_mark)
            # use MSE loss
            outputs = pred[:, -1:, -1:]
            label = seq_y[:, -1:, :]

            loss = self.loss_fn(outputs, label, weight=None)
            losses.append(loss.item())

            if loss.isnan():
                self.logger.info('loss is nan, checking inputs')

            score = self.metric_fn(outputs, label)
            scores.append(score.item())

            loss.backward()
            self.train_optimizer.step()

        return np.mean(losses), np.mean(scores)

    def test_epoch(self, data_loader):
        self.model.eval()
        scores = []
        losses = []
        try:
            for _, (input_x, weight_x) in enumerate(tqdm(data_loader, mininterval=2)):

                seq_x, seq_y, emb_mark = input_x

                seq_x = seq_x.to(torch.float32)
                seq_y = seq_y.to(torch.float32)

                # seq_x = torch.concat([seq_x, seq_y], axis=2)
                seq_x = seq_x[:, :-self.pred_len, :]

                dec_x = torch.zeros_like(seq_y[:, -self.pred_len:, :]).float()
                dec_x = torch.cat([seq_y[:, :self.seq_len, :], dec_x], dim=1)

                seq_x_mark = emb_mark[:, :-self.pred_len, :]
                seq_y_mark = emb_mark

                self.train_optimizer.zero_grad()
                if self.device.type == 'cuda':
                    seq_x = seq_x.float().cuda()
                    seq_y = seq_y.float().cuda()
                    dec_x = dec_x.float().cuda()
                    seq_x_mark = seq_x_mark.float().cuda()
                    seq_y_mark = seq_y_mark.float().cuda()

                pred = self.model(seq_x, seq_x_mark, dec_x, seq_y_mark)
                # use MSE loss
                outputs = pred[:, -1:, -1:]
                label = seq_y[:, -1:, :]


                loss = self.loss_fn(outputs, label, weight=None)
                losses.append(loss.item())

                if loss.isnan():
                    self.logger.info('loss is nan, checking inputs')

                score = self.metric_fn(outputs, label)
                scores.append(score.item())

            return np.mean(losses), np.mean(scores)

        except Exception as e:
            print(e)
            return 1, -1

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):

        sys.stdout.flush()

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            # pin_memory=True,
            # persistent_workers=True,

        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            # pin_memory=True,
            # persistent_workers=True,
        )

        save_path = get_or_create_path(save_path)
        self.fitted = True


        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        ############################## 4. train the model ################################
        start_time = time.time()
        for epoch in range(1, self.n_epochs + 1):
            self.logger.info('batch num: {}'.format(len(train_loader)))

            step = 0

            self.logger.info("Epoch%d:", epoch)
            self.logger.info("training...")
            train_loss, train_score = self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            # train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))

            if abs(train_score) < 1e-3:
                self.logger.info("train score is too small, break")

            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

            # flush the output
            sys.stdout.flush()

        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info('Total running time: {} seconds'.format(total_time))
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.num_workers)
        self.model.eval()

        preds = []
        for _, input_x in enumerate(tqdm(test_loader, mininterval=2)):

            seq_x, seq_y, emb_mark = input_x

            seq_x = seq_x.to(torch.float32)
            seq_y = seq_y.to(torch.float32)

            # seq_x = torch.concat([seq_x, seq_y], axis=2)
            seq_x = seq_x[:, :-self.pred_len, :]

            dec_x = torch.zeros_like(seq_y[:, -self.pred_len:, :]).float()
            dec_x = torch.cat([seq_y[:, :self.seq_len, :], dec_x], dim=1)


            seq_x_mark = emb_mark[:, :-self.pred_len, :]
            seq_y_mark = emb_mark

            self.train_optimizer.zero_grad()
            if self.device.type == 'cuda':
                seq_x = seq_x.float().cuda()
                dec_x = dec_x.float().cuda()
                seq_x_mark = seq_x_mark.float().cuda()
                seq_y_mark = seq_y_mark.float().cuda()

            with torch.no_grad():
                try:
                    pred = self.model(seq_x, seq_x_mark, dec_x, seq_y_mark)
                    outputs = pred[:, -1:, -1:]
                except Exception as e:
                    print(e)
                    print(' _  : ', _)
            preds.append(outputs.detach().cpu().numpy())

        return pd.Series(np.concatenate(preds).flatten(), index=dl_test.get_index())
