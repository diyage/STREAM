import os
import pandas as pd
import tarfile
import torch
import urllib.request

from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from torch.utils.data import Dataset as TorchDataset
import numpy as np
from abc import abstractmethod
import typing
import collections
import random

from .general_dataset import MinMaxNormalizer, MeanStdNormalizer


class GeneralDatasetForFutureGraph(TorchDataset):
    def __init__(
            self,
            topology: typing.List[dict],
            statics_data: np.ndarray,
            station_data: np.ndarray,
            future_station_data: np.ndarray,
            time_stamps: pd.DatetimeIndex,
            station_normalizer: typing.Union[MinMaxNormalizer, MeanStdNormalizer],
            history_time_step_num: int = 24,
            future_time_step_num: int = 6,
            max_forecast_step_num: int = 1,
            max_direct_upstream_num: int = None,
            max_direct_upstream_distance: float = None,
            root_station_ind: int = 288
    ):
        super().__init__()

        self.rank = int(os.environ.get("RANK", 0))
        """build topology"""
        self.root_station_ind = root_station_ind
        self.topology = topology
        self.has_direct_upstream_station_idx_vec = []
        self.no_direct_upstream_station_idx_vec = []
        self.all_station_idx_vec = []
        for direct_upstream_info in self.topology:
            self.all_station_idx_vec.append(direct_upstream_info["idx"])
            if len(direct_upstream_info["direct_up_idx_vec"]) != 0:
                self.has_direct_upstream_station_idx_vec.append(direct_upstream_info["idx"])
            else:
                self.no_direct_upstream_station_idx_vec.append(direct_upstream_info["idx"])

        self.max_direct_upstream_num = max_direct_upstream_num
        self.max_direct_upstream_distance = max_direct_upstream_distance
        if self.rank == 0:
            print(f"\n>>> we set max_direct_upstream_num: {max_direct_upstream_num}, max_direct_upstream_distance: {max_direct_upstream_distance}", flush=True)

        """build dynamic data"""
        self.time_stamps = time_stamps
        self.station_normalizer = station_normalizer
        if self.rank == 0:
            print(f">>> we use normalizer: {station_normalizer} for station")

        assert isinstance(station_data, np.ndarray) and len(station_data.shape) == 3
        self.station_data = station_normalizer.normalize(station_data, data_dim='sn,t,c')
        assert isinstance(future_station_data, np.ndarray) and len(future_station_data.shape) == 4
        self.max_forecast_step_num = max_forecast_step_num
        assert max_forecast_step_num <= future_station_data.shape[2]
        if self.rank == 0:
            print(f"\n>>> now dataset suppose {future_station_data.shape[2]} max_forecast_step_num, you set: {max_forecast_step_num}")
        self.future_station_data = station_normalizer.normalize(future_station_data[..., :max_forecast_step_num, :], data_dim='sn,t,f,c')

        assert self.station_data.shape[0] == self.future_station_data.shape[0]
        assert self.station_data.shape[1] == self.future_station_data.shape[1]
        assert self.station_data.shape[2] == self.future_station_data.shape[3]

        """build statics data"""
        assert len(statics_data.shape) == 2  # [sn, c]
        self.statics_mean = statics_data.mean(axis=0, keepdims=True)
        self.statics_std = statics_data.std(axis=0, keepdims=True)
        self.statics_data = (statics_data - self.statics_mean) / (self.statics_std + 1e-8)

        """build edge info"""
        self.edge_feature_mean, self.edge_feature_std = self.build_edge_info()
        
        if self.rank == 0:
            print(f"\n>>> already build edge feature mean and std", flush=True)

        self.station_offset_to_node_level = []
        self.edge_features, self.src2dst = self.build_edge_features()

        """others"""
        self.history_time_step_num = history_time_step_num
        self.future_time_step_num = future_time_step_num

        self.station_num, self.station_time_step_num, self.station_feature_dim = self.station_data.shape

        if self.rank == 0:
            print(
                f">>> station info, station num: {self.station_num}, leaf: {len(self.no_direct_upstream_station_idx_vec)}, non-leaf: {len(self.has_direct_upstream_station_idx_vec)}, edge_shape: {self.edge_features.shape}, station_time_step_num: {self.station_time_step_num}, station_feature_dim: {self.station_feature_dim}"
            )

        self.check_data()

        # each_group_data_length = self.history_time_step_num + self.future_time_step_num
        # self.time_stamps_group_ind_vec = []
        # for i in range(0, self.station_time_step_num):
        #     max_forecast_need_time_stamp_length = each_group_data_length + self.max_forecast_step_num - 1
        #     if i + max_forecast_need_time_stamp_length > self.station_time_step_num:
        #         break
        #     else:
        #         self.time_stamps_group_ind_vec.append([i, i + each_group_data_length])
        #
        # self.total_time_step = len(self.time_stamps_group_ind_vec)

        each_group_data_length = self.history_time_step_num + self.future_time_step_num
        self.time_stamps_group_ind_vec = []
        for i in range(0, self.station_time_step_num):
            j = i + each_group_data_length
            k = j + self.max_forecast_step_num - 1
            if k > self.station_time_step_num:
                break
            else:
                self.time_stamps_group_ind_vec.append([i, j])

        self.total_time_step = len(self.time_stamps_group_ind_vec)

    def build_edge_info(self):
        edge_feature_record = collections.defaultdict(list)
        for item in self.topology:
            for var_name, var_value in item.items():
                if var_name in ['idx', "gauge_id", "direct_up_idx_vec", "direct_up_gauge_id_vec"]:
                    continue
                else:
                    edge_feature_record[var_name] += var_value
        
        edge_feature_mean = collections.defaultdict()
        edge_feature_std = collections.defaultdict()
        for var_name, var_value in edge_feature_record.items():
            edge_feature_mean[var_name] = np.mean(var_value)
            edge_feature_std[var_name] = np.std(var_value) + 1e-8

        return edge_feature_mean, edge_feature_std
    

    def build_node_level(self, station_ind, node_level):
        res = {station_ind: node_level}

        for direct_upstream_station_ind in self.topology[station_ind]["direct_up_idx_vec"]:
            res.update(self.build_node_level(direct_upstream_station_ind, node_level + 1))

        return res

    def build_edge_features(self):
        all_station_ind_node_level = self.build_node_level(self.root_station_ind, node_level=0)
        station_ind_2_node_level = [all_station_ind_node_level[station_ind] for station_ind in range(self.station_data.shape[0])]
        assert min(station_ind_2_node_level) == 0
        max_node_level = max(station_ind_2_node_level)
        self.station_offset_to_node_level = station_ind_2_node_level
        src_station_ind_2_edge_level = [max_node_level - node_level for node_level in station_ind_2_node_level]
        if self.rank == 0:
            print(f"\n>>> self.root_station_ind: {self.root_station_ind}, src_station_ind_2_edge_level: [{src_station_ind_2_edge_level[self.root_station_ind]}]", flush=True)
        edge_features_vec = []
        src2dst_vec = []

        for now_station_ind in range(len(self.topology)):

            if now_station_ind not in self.has_direct_upstream_station_idx_vec:
                continue

            direct_upstream_info = self.topology[now_station_ind]
            cache = []
            for i in range(len(direct_upstream_info["direct_up_idx_vec"])):
                cache.append({
                    "direct_up_idx": direct_upstream_info["direct_up_idx_vec"][i],
                    "direct_up_gauge_id": direct_upstream_info["direct_up_gauge_id_vec"][i],
                    "stream_flow_mean": self.station_data[direct_upstream_info["direct_up_idx_vec"][i], :, 0].mean()
                })
                
                cache[-1].update({
                    var_name: (direct_upstream_info[var_name][i] - self.edge_feature_mean[var_name]) / self.edge_feature_std[var_name]
                    for var_name in self.edge_feature_mean.keys()
                })

            if self.max_direct_upstream_distance is not None:
                filter_data = []
                min_distance_ind = 0
                for i in range(len(direct_upstream_info["direct_up_idx_vec"])):
                    if direct_upstream_info["length_km"][i] < direct_upstream_info["length_km"][min_distance_ind]:
                        min_distance_ind = i

                    if direct_upstream_info["length_km"][i] <= self.max_direct_upstream_distance:
                        filter_data.append(cache[i])

                if len(filter_data) == 0 and len(cache) != 0:
                    filter_data = [cache[min_distance_ind]]

                cache = filter_data

            if self.max_direct_upstream_num is not None:
                sorted_data = sorted(cache, key=lambda x: x["stream_flow_mean"], reverse=True)
                cache = sorted_data[:self.max_direct_upstream_num]

            for item in cache:
                edge_features_vec.append([item[var_name] for var_name in self.edge_feature_mean.keys()])
                src2dst_vec.append(
                    [
                        item['direct_up_idx'],
                        now_station_ind,
                        src_station_ind_2_edge_level[item['direct_up_idx']]
                    ]
                )

        return np.array(edge_features_vec, dtype=np.float32), np.array(src2dst_vec, dtype=np.int32)

    def check_data(self):
        if self.rank == 0:
            print(f">>> check data ...")
        assert np.isnan(self.station_data).astype(np.float32).sum() == 0
        assert np.isnan(self.statics_data).astype(np.float32).sum() == 0

    def get_used_station_offset_vec(self):
        return self.all_station_idx_vec

    def __len__(self):
        return len(self.time_stamps_group_ind_vec)

    def __getitem__(self, ind):
        time_step_start_ind, time_step_end_ind = self.time_stamps_group_ind_vec[ind]

        return {
            "basin_statics": torch.tensor(self.statics_data, dtype=torch.float32),
            "basin_weather": torch.tensor(self.station_data[:, time_step_start_ind: time_step_end_ind, 1:], dtype=torch.float32),
            "future_basin_weather": torch.tensor(self.future_station_data[:, time_step_end_ind-1, :, 1:], dtype=torch.float32),
            "station_stream_flow": torch.tensor(self.station_data[:, time_step_start_ind: time_step_end_ind, 0], dtype=torch.float32),
            "future_station_stream_flow": torch.tensor(self.future_station_data[:, time_step_end_ind-1, :, 0], dtype=torch.float32),
            "time_step": torch.tensor(time_step_start_ind, dtype=torch.long),
        }


class MixDatasetForFutureGraph(TorchDataset):
    def __init__(self, sub_basin_datasets: GeneralDatasetForFutureGraph, all_basin_datasets: GeneralDatasetForFutureGraph, seed: int = 42, batch_size: int = 1):
        super().__init__()
        self.sub_basin_datasets = sub_basin_datasets
        self.all_basin_datasets = all_basin_datasets
        assert len(sub_basin_datasets) == len(all_basin_datasets)
        num_len = len(sub_basin_datasets)
        random.seed(seed)
        time_ind_vec = list(range(num_len))
        random.shuffle(time_ind_vec)
        self.group_ind_vec = []
        i = 0
        while i < num_len:
            j = i + batch_size
            if j >= num_len:
                end = num_len
                start = num_len - batch_size
            else:
                start = i
                end = j
            self.group_ind_vec.append(time_ind_vec[start: end])
            i = j


    def __len__(self):
        return len(self.group_ind_vec)

    def __getitem__(self, group_index):

        sub_basin_info_vec = [self.sub_basin_datasets[index] for index in self.group_ind_vec[group_index]]
        all_basin_info_vec = [self.all_basin_datasets[index] for index in self.group_ind_vec[group_index]]

        return {
            "station_stream_flow": torch.stack([sub_basin_info["station_stream_flow"] for sub_basin_info in sub_basin_info_vec], dim=0),
            "future_station_stream_flow": torch.stack([sub_basin_info["future_station_stream_flow"] for sub_basin_info in sub_basin_info_vec], dim=0),

            "src": torch.tensor(self.sub_basin_datasets.src2dst[:, 0], dtype=torch.long),
            "dst": torch.tensor(self.sub_basin_datasets.src2dst[:, 1], dtype=torch.long),
            "edge_features": torch.tensor(self.sub_basin_datasets.edge_features, dtype=torch.float32),

            "non_leaf_mask": torch.tensor([1 if ind in self.sub_basin_datasets.has_direct_upstream_station_idx_vec else 0 for ind in self.sub_basin_datasets.all_station_idx_vec], dtype=torch.float32),

            "sub_basin_statics": torch.stack([sub_basin_info["basin_statics"] for sub_basin_info in sub_basin_info_vec], dim=0),
            "sub_basin_weather": torch.stack([sub_basin_info["basin_weather"] for sub_basin_info in sub_basin_info_vec], dim=0),
            "future_sub_basin_weather": torch.stack([sub_basin_info["future_basin_weather"] for sub_basin_info in sub_basin_info_vec], dim=0),

            "all_basin_statics": torch.stack([all_basin_info["basin_statics"] for all_basin_info in all_basin_info_vec], dim=0),
            "all_basin_weather": torch.stack([all_basin_info["basin_weather"] for all_basin_info in all_basin_info_vec], dim=0),
            "future_all_basin_weather": torch.stack([all_basin_info["future_basin_weather"] for all_basin_info in all_basin_info_vec], dim=0),

            "time_step": torch.stack([sub_basin_info["time_step"] for sub_basin_info in sub_basin_info_vec], dim=0)
        }


class BlendDynamicDataset(TorchDataset):
    def __init__(self, dataset_vec: typing.List[MixDatasetForFutureGraph]):
        super().__init__()
        self.dataset_vec = dataset_vec
        self.time_stamps_group_ind_vec = []
        for dataset_ind in range(len(dataset_vec)):
            for now_data_ind in range(len(dataset_vec[dataset_ind])):
                self.time_stamps_group_ind_vec.append([dataset_ind, now_data_ind])

    def __len__(self):
        return len(self.time_stamps_group_ind_vec)

    def __getitem__(self, ind):
        dataset_ind, now_data_ind = self.time_stamps_group_ind_vec[ind]
        res = self.dataset_vec[dataset_ind][now_data_ind]
        res["dataset_ind"] = torch.tensor(dataset_ind, dtype=torch.long)
        return res
