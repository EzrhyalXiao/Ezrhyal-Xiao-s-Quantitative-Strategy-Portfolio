import sys
import time
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader

from qlib.log import get_module_logger
from qlib.utils import get_or_create_path

from qlib.model.base import Model
from qlib.model.utils import ConcatDataset

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearModel(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.linear = nn.Linear(configs.enc_in, 1)

    def forward(self, x, emb_x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return self.linear(x.permute(0,2,1)) # to [Batch, Output length, Channel]

class DLinear(Model):
    def __init__(
        self,
        enc_in: int = 27,
        batch_size: int = 256,
        n_epochs: int = 10,
        early_stop: int = 10,
        optimizer: str = "adam",
        loss: str = "mse",
        metric: str = "",
        lr: float = 1e-3,
        seq_len: int = 12,
        pred_len: int = 1,
        dropout: float = 0.1,
        norm: str = "layernorm",
        individual: bool = False,
        act: str = "gelu",
        subtract_last: bool = False,
        decomposition: bool = False,
        stride: int = 1,
        verbose: bool = False,
        GPU: int = 0,
        n_jobs: int = 8,
        seed: int = 42,
    ):

        self.logger = get_module_logger("PatchTST")

        self.enc_in = enc_in
        self.lr = lr
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout = dropout
        self.norm = norm
        self.individual = individual
        self.act = act
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.stride = stride


        self.batch_size = batch_size

        self.epoch = n_epochs
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.metric = metric


        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # self.lookback_shape = (self.seq_len + self.pred_len, self.pred_len)
        # self.attr_shape = (self.lookback_len, self.d_feat)
        # self.dynCov_shape = (self.seq_len + self.pred_len, self.dyn_feat)



        self.model = DLinearModel(
            self).to(self.device)
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

            seq_x = torch.concat([seq_x, seq_y], axis=2)

            seq_x = seq_x[:, :-self.pred_len, :]
            seq_y = seq_y[:, -self.pred_len:, :]
            # seq_y = seq_train[:, :, [-1]]

            self.train_optimizer.zero_grad()
            if self.device.type == 'cuda':
                seq_x = seq_x.float().cuda()
                seq_y = seq_y.float().cuda()
                emb_mark = emb_mark.float().cuda()

            pred = self.model(seq_x, emb_mark)
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

                seq_x = torch.concat([seq_x, seq_y], axis=2)

                seq_x = seq_x[:, :-self.pred_len, :]
                seq_y = seq_y[:, -self.pred_len:, :]
                # seq_y = seq_train[:, :, [-1]]

                self.train_optimizer.zero_grad()
                if self.device.type == 'cuda':
                    seq_x = seq_x.float().cuda()
                    seq_y = seq_y.float().cuda()
                    emb_mark = emb_mark.float().cuda()

                pred = self.model(seq_x, emb_mark)
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
            num_workers=self.n_jobs,
            drop_last=True,
            # pin_memory=True,
            # persistent_workers=True,

        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
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
        for epoch in range(1, self.epoch + 1):
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
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()

        preds = []
        for _, input_x in enumerate(tqdm(test_loader, mininterval=2)):

            seq_x, seq_y, emb_mark = input_x

            seq_x = seq_x.to(torch.float32)
            seq_y = seq_y.to(torch.float32)

            seq_x = torch.concat([seq_x, seq_y], axis=2)

            seq_x = seq_x[:, :-self.pred_len, :]
            seq_y = seq_y[:, -self.pred_len:, :]
            # seq_y = seq_train[:, :, [-1]]

            self.train_optimizer.zero_grad()
            if self.device.type == 'cuda':
                seq_x = seq_x.float().cuda()
                seq_y = seq_y.float().cuda()
                emb_mark = emb_mark.float().cuda()

            with torch.no_grad():
                try:
                    pred = self.model(seq_x, emb_mark)
                    outputs = pred[:, -1:, -1:]
                except Exception as e:
                    print(e)
                    print(' _  : ', _)
            preds.append(outputs.detach().cpu().numpy())

        return pd.Series(np.concatenate(preds).flatten(), index=dl_test.get_index())

