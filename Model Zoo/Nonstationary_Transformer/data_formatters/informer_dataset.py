# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Alpha158 dataset.

Defines dataset specific column definitions and data transformations.
"""
import time
import bisect
import pandas as pd
import numpy as np

from copy import copy, deepcopy
from typing import Callable, Union, List, Tuple, Dict, Text, Optional

from utils.timefeatures import time_features
from qlib.data.dataset import TSDataSampler, TSDatasetH, DatasetH

from numba import jit

import warnings
warnings.filterwarnings("ignore")

@jit
def array_match(A, B):
    # find the indices where B in A
    idce = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] in B:
                idce.append([i, j])
    idce = np.array(idce)
    return idce.T

@jit
def search_indice(indices, idx_arr):
    row, col = [], []
    for indice in indices:
        if len(np.unique(indice)) == len(indice):
            idce = []
            for i in range(idx_arr.shape[0]):
                for j in range(idx_arr.shape[1]):
                    if idx_arr[i, j] in indice:
                        idce.append([i, j])
            idce = np.array(idce)
            rows, cols = idce.T
        else:
            rcs = [np.where(idx_arr == id) for id in indice]
            rows, cols = [id[0][0] for id in rcs], [id[1][0] for id in rcs]
        row.append(rows)
        col.append(cols)
    rows = np.concatenate(row)
    cols = np.concatenate(col)
    return rows, cols

class InformerDataSampler(TSDataSampler):
    """Defines and formats data for the Alpha158 dataset.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        start,
        end,
        step_len: int,
        fillna_type: str = "none",
        dtype=None,
        flt_data=None,
    ):

        super().__init__(data, start, end, step_len, fillna_type, dtype, flt_data)

        time_idx = list(self.idx_df.index)
        self.data_stamp = time_features(time_idx)

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
            """
            # We have two method to get the time-series of a sample
            tsds is a instance of TSDataSampler

            # 1) sample by int index directly
            tsds[len(tsds) - 1]

            # 2) sample by <datetime,instrument> index
            tsds['2016-12-31', "SZ300315"]

            # The return value will be similar to the data retrieved by following code
            df.loc(axis=0)['2015-01-01':'2016-12-31', "SZ300315"].iloc[-30:]

            Parameters
            ----------
            idx : Union[int, Tuple[object, str]]
            """

            mtit = (list, np.ndarray)
            if isinstance(idx, mtit):
                indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
                indices = np.concatenate(indices)
            else:
                indices = self._get_indices(*self._get_row_col(idx))

            row, col = self._get_row_col(idx)

            # 1) for better performance, use the last nan line for padding the lost date
            # 2) In case of precision problems. We use np.float64. # TODO: I'm not sure if whether np.float64 will result in
            # precision problems. It will not cause any problems in my tests at least
            indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)
            indices_diff = indices[:-1] - indices[-1]

            rows = np.full(len(indices), row)
            rows[:-1] = rows[:-1] + indices_diff

            if (np.diff(indices) == 1).all():  # slicing instead of indexing for speeding up.
                data = self.data_arr[indices[0]: indices[-1] + 1]
            else:
                data = self.data_arr[indices]
            if isinstance(idx, mtit):
                # if we get multiple indexes, addition dimension should be added.
                # <sample_idx, step_idx, feature_idx>
                data = data.reshape(-1, self.step_len, *data.shape[1:])

            # indices = self.idx_arr[max(row - self.step_len + 1, 0): row + 1, col]
            # rows = [self._get_row_col(idx)[0] for idx in indices]

            data_stamp = self.data_stamp[rows]

            return data[:, :-1], data[:, [-1]], data_stamp

class InformerDatasetH(DatasetH):
    """
    (T)ime-(S)eries Dataset (H)andler


    Convert the tabular data to Time-Series data

    Requirements analysis

    The typical workflow of a user to get time-series data for an sample
    - process features
    - slice proper data from data handler:  dimension of sample <feature, >
    - Build relation of samples by <time, instrument> index
        - Be able to sample times series of data <timestep, feature>
        - It will be better if the interface is like "torch.utils.data.Dataset"
    - User could build customized batch based on the data
        - The dimension of a batch of data <batch_idx, feature, timestep>
    """

    DEFAULT_STEP_LEN = 30

    def __init__(self, step_len=DEFAULT_STEP_LEN, **kwargs):
        self.step_len = step_len
        super().__init__(**kwargs)

    def config(self, **kwargs):
        if "step_len" in kwargs:
            self.step_len = kwargs.pop("step_len")
        super().config(**kwargs)

    def setup_data(self, **kwargs):
        super().setup_data(**kwargs)
        # make sure the calendar is updated to latest when loading data from new config
        cal = self.handler.fetch(col_set=self.handler.CS_RAW).index.get_level_values("datetime").unique()
        self.cal = sorted(cal)

    @staticmethod
    def _extend_slice(slc: slice, cal: list, step_len: int) -> slice:
        # Dataset decide how to slice data(Get more data for timeseries).
        start, end = slc.start, slc.stop
        start_idx = bisect.bisect_left(cal, pd.Timestamp(start))
        pad_start_idx = max(0, start_idx - step_len)
        pad_start = cal[pad_start_idx]
        return slice(pad_start, end)

    def _prepare_seg(self, slc: slice, **kwargs) -> InformerDataSampler:
        """
        split the _prepare_raw_seg is to leave a hook for data preprocessing before creating processing data
        NOTE: TSDatasetH only support slc segment on datetime !!!
        """
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        start, end = slc.start, slc.stop
        flt_col = kwargs.pop("flt_col", None)
        # TSDatasetH will retrieve more data for complete time-series

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super()._prepare_seg(ext_slice, **kwargs)

        flt_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super()._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        tsds = InformerDataSampler(
            data=data,
            start=start,
            end=end,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
        )
        return tsds
