import sys
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos,DataEmbedding_wo_pos_temp,DataEmbedding_wo_temp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

from qlib.log import get_module_logger
from qlib.utils import get_or_create_path

from qlib.model.base import Model
from qlib.model.utils import ConcatDataset

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter


class AutoFormerModel(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.linear = nn.Linear(configs.enc_in, 1)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        if self.output_attention:
            return self.linear(dec_out[:, -self.pred_len:, :]), attns
        else:
            return self.linear(dec_out[:, -self.pred_len:, :])  # [B, L, 1]


class Autoformer(Model):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        pred_len: int,
        embed_type: int = 0,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 2048,
        moving_avg: int = 5,
        dropout: float = 0.05,
        attn: str = "prob",
        embed: str = "fixed",
        freq: str = "d",
        activation: str = "gelu",
        output_attention: bool = False,
        distil: bool = False,
        mix: bool = False,
        device: str = "cuda",
        factor: int = 5,
        padding: bool = True,
        padding_var: float = 0.0,
        seq_len_out: int = 1,
        label_len: int = 1,
        pred_len_out: int = 1,
        num_workers: int = 0,
        batch_size: int = 32,
        eval_batch_size: int = 32,
        iter: int = 2,
        n_epochs: int = 10,
        early_stop: int = 2,
        lr: float = 0.001,
        warmup_prop: float = 0.1,
        weight_decay: float = 0.0001,
        gradient_clip_val: float = 0.1,
        loss: str = "mse",
        l1: float = 0.0,
        l2: float = 0.0,
        metric: str = "mse",
        optimizer: str = "adam",
        scheduler: str = "linear",
        seed: int = 42,
        verbose: bool = False,
        GPU: int = 0,
        **kwargs    # other parameters
    ):

        self.logger = get_module_logger("PatchTST")

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
        self.moving_avg = moving_avg
        self.dropout = dropout
        self.attn = attn
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.output_attention = output_attention
        self.distil = distil
        self.mix = mix
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
        self.iter = iter
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.lr = lr
        self.warmup_prop = warmup_prop
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.loss = loss
        self.l1 = l1
        self.l2 = l2
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.seed = seed
        self.verbose = verbose

        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = AutoFormerModel(
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
        mask = ~torch.isnan(label) & ~torch.isnan(pred)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return  self.mse(pred[mask], label[mask], weight[mask])

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