__all__ = ['TimeMixer',]

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

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

from qlib.log import get_module_logger
from qlib.utils import get_or_create_path

from qlib.model.base import Model
from qlib.model.utils import ConcatDataset

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixerModel(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out_list
        else:
            raise ValueError('Only forecast tasks implemented yet')

class TimeMixer(Model):
    def __init__(
        self,
        seq_len: int = 12,
        pred_len: int = 1,
        label_len: int = 1,
        enc_in: int = 7,
        dec_in: int = 7,
        e_layers: int = 2,
        d_layers: int = 1,
        features: str = 'M',
        c_out: int = 1,
        d_model: int = 32,
        d_ff: int=32,
        down_sampling_layers: int = 1,
        down_sampling_window: int = 2,
        channel_independence: int = 1,
        decomp_method: str = 'moving_avg',
        moving_avg: int = 20,
        down_sampling_method: str = 'avg',
        top_k: int = 5,
        dropout: float = 0.05,
        des: str = 'Exp',
        embed: str = "fixed",
        freq: str = "d",
        activation: str = "gelu",
        output_attention: bool = False,
        use_norm: bool = True,
        itr: int = 1,
        num_kernels: int = 6,
        num_workers: int = 20,
        loss: str = 'mse',
        metric: str = '',
        task_name: str = 'short_term_forecast',
        optimizer: str = "adam",
        n_epochs: int = 10,
        batch_size: int = 1024,
        early_stop: int = 2,
        lr: float = 0.001,
        seed: int = 42,
        verbose: bool = False,
        GPU: int = 0,
        **kwargs    # other parameters
    ):

        self.logger = get_module_logger("TimeNet")

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.channel_independence = channel_independence
        self.decomp_method = decomp_method
        self.moving_avg = moving_avg
        self.use_norm = use_norm
        self.down_sampling_method = down_sampling_method

        self.features = features
        self.task_name = task_name
        self.top_k = top_k
        self.des = des
        self.dropout = dropout

        self.freq = freq
        self.activation = activation
        self.output_attention = output_attention

        self.iter = itr
        self.embed = embed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_kernels = num_kernels
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.lr = lr
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.seed = seed
        self.verbose = verbose

        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = TimeMixerModel(
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

            dec_x = torch.zeros_like(seq_y[:, (-self.pred_len):, :]).float()
            dec_x = torch.cat([seq_y[:, :(self.seq_len), :], dec_x], dim=1)


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

                # dec_x = torch.zeros_like(seq_y[:, -self.pred_len:, :]).float()
                # dec_x = torch.cat([seq_y[:, :self.seq_len, :], dec_x], dim=1)

                dec_x = torch.zeros_like(seq_y[:, (-self.pred_len - self.pred_len):, :]).float()
                dec_x = torch.cat([seq_y[:, :(self.seq_len - self.pred_len), :], dec_x], dim=1)

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

            # dec_x = torch.zeros_like(seq_y[:, -self.pred_len:, :]).float()
            # dec_x = torch.cat([seq_y[:, :self.seq_len, :], dec_x], dim=1)

            dec_x = torch.zeros_like(seq_y[:, (-self.pred_len-self.pred_len):, :]).float()
            dec_x = torch.cat([seq_y[:, :(self.seq_len-self.pred_len), :], dec_x], dim=1)


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
