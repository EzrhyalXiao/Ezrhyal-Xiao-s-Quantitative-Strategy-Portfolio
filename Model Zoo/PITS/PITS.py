__all__ = ['PITS']

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

from typing import Callable, Optional
from layers.PITS_backbone import PITS_backbone
from layers.PITS_layers import series_decomp

class PITSModel(nn.Module):
    def __init__(self, configs, 
                  verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        d_feat = configs.d_feat

        context_window = configs.seq_len
        target_window = configs.pred_len
        
        d_model = configs.d_model
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        c_in = configs.c_in
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        shared_embedding = configs.shared_embedding
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            
        else:
            self.model = PITS_backbone(c_in=c_in, 
                                 context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  d_model=d_model,
                                  shared_embedding=shared_embedding,
                                  head_dropout=head_dropout, 
                                  padding_patch = padding_patch,
                                  individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
        self.linear = nn.Linear(d_feat, 1)

    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        x = self.linear(x)
        return x

class PITS(Model):
    def __init__(
        self,
        d_feat: int = 27,
        enc_in: int = 7,
        dec_in: int = 7,
        batch_size: int = 256,
        epoch: int = 10,
        early_stop: int = 10,
        optimizer: str = "adam",
        loss: str = "mse",
        metric: str = "",
        lr: float = 1e-3,
        seq_len: int = 12,
        pred_len: int = 1,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        dropout: float = 0.1,
        fc_dropout: float = 0.1,
        head_dropout: float = 0.1,
        norm: str = "layernorm",
        act: str = "gelu",
        key_padding_mask: bool = True,
        padding_var: int = None,
        attn_mask: Tensor = None,
        res_attention: bool = True,
        shared_embedding: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        individual: bool = False,
        revin: bool = False,
        affine: bool = False,
        subtract_last: bool = False,
        decomposition: bool = False,
        kernel_size: int = 3,
        patch_len: int = 4,
        stride: int = 1,
        padding_patch: bool = False,
        verbose: bool = False,
        GPU: int = 0,
        n_jobs: int = 8,
        seed: int = 42,
    ):

        self.logger = get_module_logger("PatchTST")

        """
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        d_model = configs.d_model
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        c_in = configs.c_in
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        shared_embedding = configs.shared_embedding
        """

        #init self value

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.d_feat = d_feat
        self.d_model = d_model
        self.head_dropout = head_dropout
        self.individual = individual

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        self.c_in = enc_in
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last

        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.shared_embedding = shared_embedding


        self.dec_in = dec_in
        self.lr = lr

        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.norm = norm
        self.act = act
        self.key_padding_mask = key_padding_mask
        self.padding_var = padding_var
        self.attn_mask = attn_mask
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.pe = pe
        self.learn_pe = learn_pe
        self.pretrain_head = pretrain_head
        self.head_type = head_type


        self.batch_size = batch_size
        self.epoch = epoch

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

        self.logger.info(
            "PatchTST parameters setting:"
            "\nd_model : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_model,
                e_layers,
                dropout,
                epoch,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                GPU,
                self.use_gpu,
                seed
        ))

        self.model = PITSModel(
            self,
            verbose=self.verbose,
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

            seq_x, seq_y, seq_mark, emb_mark = input_x

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

            pred = self.model(seq_x)
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

                seq_x, seq_y, seq_mark, emb_mark = input_x

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

                pred = self.model(seq_x)
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
            pin_memory=True,
            # persistent_workers=True,

        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=False,
            pin_memory=True,
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

            seq_x, seq_y, seq_mark, emb_mark = input_x

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
                    pred = self.model(seq_x)
                    outputs = pred[:, -1:, -1:]
                except Exception as e:
                    print(e)
                    print(' _  : ', _)
            preds.append(outputs.detach().cpu().numpy())

        return pd.Series(np.concatenate(preds).flatten(), index=dl_test.get_index())
