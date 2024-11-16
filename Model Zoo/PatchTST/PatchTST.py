__all__ = ['PatchTST']

# Cell

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

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

class PatchTSTModel(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='LayerNorm', attn_dropout:float=0.,
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        d_feat = configs.d_feat
        emb_feat_in = configs.emb_feat_in
        emb_feat_out = configs.emb_feat_out
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        batch_size = configs.batch_size
        
        
        # model
        self.decomposition = decomposition

        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)

        self.linear = nn.Linear(d_feat, 1)
        self.emb_linear = nn.Linear(emb_feat_in, emb_feat_out)

    def forward(self, x, emb_x):           # x: [Batch, Input length, Channel]
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

        x = x[:, -1:, :]

        return x


class PatchTST(Model):
    def __init__(
        self,
        d_feat: int = 27,
        emb_feat_in: int = 10,
        emb_feat_out: int = 3,
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

        #init self value
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.lr = lr
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.d_feat = d_feat
        self.emb_feat_in = emb_feat_in
        self.emb_feat_out = emb_feat_out
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
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
        self.individual = individual
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

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

        self.model = PatchTSTModel(
            self,
            max_seq_len=self.seq_len + self.pred_len,
            d_k=self.d_model // self.n_heads,
            d_v=self.d_model // self.n_heads,
            norm=self.norm,
            attn_dropout=self.dropout,
            act=self.act,
            key_padding_mask=self.key_padding_mask,
            padding_var=self.padding_var,
            attn_mask=self.attn_mask,
            res_attention=self.res_attention,
            pre_norm=self.pre_norm,
            store_attn=self.store_attn,
            pe=self.pe,
            learn_pe=self.learn_pe,
            pretrain_head=self.pretrain_head,
            head_type=self.head_type,
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
                    pred = self.model(seq_x, emb_mark)
                    outputs = pred[:, -1:, -1:]
                except Exception as e:
                    print(e)
                    print(' _  : ', _)
            preds.append(outputs.detach().cpu().numpy())

        return pd.Series(np.concatenate(preds).flatten(), index=dl_test.get_index())
