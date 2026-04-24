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


class Normalizer:
    def __init__(
            self,
            m_=None,
            s_=None
    ):
        if m_ is not None or s_ is not None:
            print(f">>> we use normalized info")
        self.m_ = m_
        self.s_ = s_

    @abstractmethod
    def normalize(self, data: np.ndarray) -> np.ndarray:
        raise RuntimeError(">>> you must implement this method")


class MinMaxNormalizer(Normalizer):
    def __init__(
            self,
            m_=None,
            s_=None
    ):
        super().__init__(m_, s_)

    def normalize(self, data: np.ndarray, data_dim='sn,t,c') -> np.ndarray:
        if data_dim == "sn,t,c":
            return self.normalize_for_sn_t_c(data)
        else:
            return self.normalize_for_sn_t_f_c(data)

    def normalize_for_sn_t_c(self, data):
        assert len(data.shape) == 3
        if self.m_ is None:
            self.m_ = data.min(axis=1, keepdims=True)
        if self.s_ is None:
            self.s_ = data.max(axis=1, keepdims=True) - self.m_
            assert (self.s_ <= 0).astype(np.float32).sum() == 0

        return (data - self.m_) / self.s_

    def normalize_for_sn_t_f_c(self, data):
        assert len(data.shape) == 4
        m_ = self.m_[:, :, None, :]
        s_ = self.s_[:, :, None, :]
        return (data - m_) / s_


class MeanStdNormalizer(Normalizer):
    def __init__(
            self,
            m_=None,
            s_=None
    ):
        super().__init__(m_, s_)

    def normalize(self, data: np.ndarray, data_dim='sn,t,c') -> np.ndarray:
        if data_dim == "sn,t,c":
            return self.normalize_for_sn_t_c(data)
        else:
            return self.normalize_for_sn_t_f_c(data)

    def normalize_for_sn_t_c(self, data):
        assert len(data.shape) == 3
        if self.m_ is None:
            self.m_ = data.mean(axis=1, keepdims=True)
        if self.s_ is None:
            self.s_ = data.std(axis=1, keepdims=True)
            # assert (self.s_ == 0).astype(np.float32).sum() == 0
            mask = self.s_ == 0
            self.m_[mask] = 0.0
            self.s_[mask] = 1.0

        return (data - self.m_) / self.s_

    def normalize_for_sn_t_f_c(self, data):
        assert len(data.shape) == 4
        m_ = self.m_[:, :, None, :]
        s_ = self.s_[:, :, None, :]
        return (data - m_) / s_


class GeneralDataset(TorchDataset):
    def __init__(
            self,
            station_data: np.ndarray,
            station_normalizer: typing.Union[MinMaxNormalizer, MeanStdNormalizer],
            river_data: np.ndarray = None,
            river_normalizer: typing.Union[MinMaxNormalizer, MeanStdNormalizer] = None,
            predict_delta: bool = False,
            data_mean_frequency: int = 1,
            history_time_step_num: int = 24,
            future_time_step_num: int = 6,
            normalize: bool = True
    ):
        super().__init__()
        assert isinstance(station_data, np.ndarray) and len(station_data.shape) == 3
        self.station_normalizer = station_normalizer
        if normalize:
            print(f">>> we use normalizer: {station_normalizer} for station")
            self.station_data = station_normalizer.normalize(station_data)
        else:
            print(f">>> we do not use any normalizer for station")
            self.station_data = station_data

        self.history_time_step_num = history_time_step_num
        self.future_time_step_num = future_time_step_num
        self.predict_delta = predict_delta
        self.data_mean_frequency = data_mean_frequency

        self.station_num, self.station_time_step_num, self.station_feature = self.station_data.shape[0], \
                                                                             self.station_data.shape[1], \
                                                                             self.station_data.shape[2]

        print(
            f">>> station info, station num: {self.station_num}, station_time_step_num: {self.station_time_step_num}, station_feature: {self.station_feature}"
        )

        assert history_time_step_num % data_mean_frequency == 0 and future_time_step_num % data_mean_frequency == 0
        self.river_normalizer = river_normalizer
        if river_data is not None:
            assert isinstance(river_data, np.ndarray) and len(river_data.shape) == 3
            if normalize:
                print(f">>> we use normalizer: {station_normalizer} for river")
                self.river_data = river_normalizer.normalize(river_data)
            else:
                print(f">>> we do not use any normalizer for river")
                self.river_data = river_data
            self.river_num, self.river_time_step_num, self.river_feature = self.river_data.shape[0], \
                                                                           self.river_data.shape[1], \
                                                                           self.river_data.shape[2]
            assert self.river_time_step_num == self.station_time_step_num
        else:
            self.river_data = None
            self.river_num, self.river_time_step_num, self.river_feature = -1, -1, -1

        self.check_data()

    def check_data(self):
        print(f">>> check data ...")
        if self.river_data is not None:
            assert np.isnan(self.river_data).astype(np.float32).sum() == 0

        if self.station_data is not None:
            assert np.isnan(self.station_data).astype(np.float32).sum() == 0

    def __len__(self):
        if self.predict_delta:
            return self.station_time_step_num - (self.history_time_step_num + self.future_time_step_num + self.data_mean_frequency) + 1
        else:
            return self.station_time_step_num - (self.history_time_step_num + self.future_time_step_num) + 1

    def __getitem__(self, index):

        res = {}

        # for station
        start_ind = index
        if self.predict_delta:
            end_ind = start_ind + self.history_time_step_num + self.future_time_step_num + self.data_mean_frequency
        else:
            end_ind = start_ind + self.history_time_step_num + self.future_time_step_num
        station_data_ = self.station_data[:, start_ind: end_ind, :]
        station_data_vec = np.split(station_data_, indices_or_sections=station_data_.shape[1] // self.data_mean_frequency, axis=1)
        station_data_vec = [np.mean(v, axis=1, keepdims=True) for v in station_data_vec]
        station_data_ = np.concatenate(station_data_vec, axis=1)

        if self.predict_delta:
            res["start"] = station_data_[:, [0], :]
            station_data_ = station_data_[:, 1:, :] - station_data_[:, :-1, :]

        res["station_past_data"] = torch.tensor(
            station_data_[:, :self.history_time_step_num // self.data_mean_frequency, :],
            dtype=torch.float32
        )
        res["station_future_data"] = torch.tensor(
            station_data_[:, self.history_time_step_num // self.data_mean_frequency:, :],
            dtype=torch.float32
        )

        if self.river_data is not None:
            # for river
            end_ind = start_ind + self.history_time_step_num + self.future_time_step_num

            river_data_ = self.river_data[:, start_ind: end_ind, :]
            river_data_vec = np.split(river_data_, indices_or_sections=river_data_.shape[1] // self.data_mean_frequency, axis=1)
            river_data_vec = [np.mean(v, axis=1, keepdims=True) for v in river_data_vec]
            river_data_ = np.concatenate(river_data_vec, axis=1)
            res["river_data"] = torch.tensor(river_data_, dtype=torch.float32)

        return res


class GeneralDatasetWithTopology(TorchDataset):
    def __init__(
            self,
            topology: typing.List[dict],
            use_edge_feature: bool,
            station_data: np.ndarray,
            station_normalizer: typing.Union[MinMaxNormalizer, MeanStdNormalizer],
            river_data: np.ndarray = None,
            river_normalizer: typing.Union[MinMaxNormalizer, MeanStdNormalizer] = None,
            predict_delta: bool = False,
            data_mean_frequency: int = 1,
            history_time_step_num: int = 24,
            future_time_step_num: int = 6,
            normalize: bool = True,
    ):
        super().__init__()
        self.topology = topology
        self.use_edge_feature = use_edge_feature
        res = self.build_edge_info()
        self.dist_hdn_mean, self.dist_hdn_std = res[0]
        self.elev_diff_mean, self.elev_diff_std = res[1]
        self.slope_mean, self.slope_std = res[2]

        assert isinstance(station_data, np.ndarray) and len(station_data.shape) == 3
        self.station_normalizer = station_normalizer
        if normalize:
            print(f">>> we use normalizer: {station_normalizer} for station")
            self.station_data = station_normalizer.normalize(station_data)
        else:
            print(f">>> we do not use any normalizer for station")
            self.station_data = station_data

        self.history_time_step_num = history_time_step_num
        self.future_time_step_num = future_time_step_num
        self.predict_delta = predict_delta
        self.data_mean_frequency = data_mean_frequency

        self.station_num, self.station_time_step_num, self.station_feature = self.station_data.shape[0], \
                                                                             self.station_data.shape[1], \
                                                                             self.station_data.shape[2]

        print(
            f">>> station info, station num: {self.station_num}, station_time_step_num: {self.station_time_step_num}, station_feature: {self.station_feature}"
        )

        assert history_time_step_num % data_mean_frequency == 0 and future_time_step_num % data_mean_frequency == 0
        self.river_normalizer = river_normalizer
        if river_data is not None:
            assert isinstance(river_data, np.ndarray) and len(river_data.shape) == 3
            if normalize:
                print(f">>> we use normalizer: {station_normalizer} for river")
                self.river_data = river_normalizer.normalize(river_data)
            else:
                print(f">>> we do not use any normalizer for river")
                self.river_data = river_data
            self.river_num, self.river_time_step_num, self.river_feature = self.river_data.shape[0], \
                                                                           self.river_data.shape[1], \
                                                                           self.river_data.shape[2]
            assert self.river_time_step_num == self.station_time_step_num
        else:
            self.river_data = None
            self.river_num, self.river_time_step_num, self.river_feature = -1, -1, -1

        self.check_data()

        assert not self.predict_delta
        assert self.river_data is None
        self.total_time_step = self.station_time_step_num - (self.history_time_step_num + self.future_time_step_num) + 1

    def build_edge_info(self):
        all_dist_hdn = []
        all_elev_diff = []
        all_slope = []
        if self.use_edge_feature:
            for item in self.topology:
                all_dist_hdn += item["dist_hdn"]
                all_elev_diff += item["elev_diff"]
                all_slope += item["slope"]

            return (np.mean(all_dist_hdn), np.std(all_dist_hdn)), (np.mean(all_elev_diff), np.std(all_elev_diff)), (np.mean(all_slope), np.std(all_slope))
        else:
            return (None, None), (None, None), (None, None)

    def check_data(self):
        print(f">>> check data ...")
        if self.river_data is not None:
            assert np.isnan(self.river_data).astype(np.float32).sum() == 0

        if self.station_data is not None:
            assert np.isnan(self.station_data).astype(np.float32).sum() == 0

    def __len__(self):
        return self.total_time_step * self.station_num

    def __getitem__(self, global_idx):

        station_id = global_idx // self.total_time_step
        index = global_idx % self.total_time_step

        # for station
        start_ind = index
        end_ind = start_ind + self.history_time_step_num + self.future_time_step_num

        cache = []

        for idx in [station_id] + self.topology[station_id]["all_up_idx_vec"]:
            if self.use_edge_feature:
                if idx == station_id:
                    edge_info = np.zeros(shape=(end_ind - start_ind, 3), dtype=self.station_data.dtype)
                else:
                    i = self.topology[station_id]["all_up_idx_vec"].index(idx)
                    edge_info = [
                        (self.topology[station_id]["dist_hdn"][i] - self.dist_hdn_mean) / self.dist_hdn_std,
                        (self.topology[station_id]["elev_diff"][i] - self.elev_diff_mean) / self.elev_diff_std,
                        (self.topology[station_id]["slope"][i] - self.slope_mean) / self.slope_std,
                    ]
                    edge_info = np.array([edge_info], dtype=self.station_data.dtype).repeat(repeats=end_ind - start_ind, axis=0)

                data = np.concatenate(
                    [
                        self.station_data[idx, start_ind: end_ind, :],
                        edge_info
                    ],
                    axis=-1
                )
            else:
                data = self.station_data[idx, start_ind: end_ind, :]

            cache.append(data)

        for _ in range(self.station_num - 1 - len(self.topology[station_id]["all_up_idx_vec"])):
            cache.append(np.zeros_like(cache[0]))

        part_one = [cache[0]]
        part_two = cache[1:]
        random.shuffle(part_two)
        cache = part_one + part_two
        station_data_ = np.stack(cache, axis=0)

        station_data_vec = np.split(station_data_, indices_or_sections=station_data_.shape[1] // self.data_mean_frequency, axis=1)
        station_data_vec = [np.mean(v, axis=1, keepdims=True) for v in station_data_vec]
        station_data_ = np.concatenate(station_data_vec, axis=1)

        res = {
            "station_idx": station_id,
            "all_up_idx_vec": self.topology[station_id]["all_up_idx_vec"],
            "time_step_start_idx": start_ind,
            "station_past_data": torch.tensor(
                station_data_[:, :self.history_time_step_num // self.data_mean_frequency, :],
                dtype=torch.float32
            ),
            "station_future_data": torch.tensor(
                station_data_[:, self.history_time_step_num // self.data_mean_frequency:, :],
                dtype=torch.float32
            )
        }

        return res


class GeneralDatasetWithTopologyDirectUpStream(TorchDataset):
    def __init__(
            self,
            topology: typing.List[dict],
            statics_data: np.ndarray,
            station_data: np.ndarray,
            station_normalizer: typing.Union[MinMaxNormalizer, MeanStdNormalizer],
            history_time_step_num: int = 24,
            future_time_step_num: int = 6,
            node_type: str = "leaf",
            max_direct_upstream_num: int = None,
            max_direct_upstream_distance: float = None
    ):
        super().__init__()
        self.topology = topology
        self.max_direct_upstream_num = max_direct_upstream_num
        self.max_direct_upstream_distance = max_direct_upstream_distance
        print(f"\n>>> we set max_direct_upstream_num: {max_direct_upstream_num}, max_direct_upstream_distance: {max_direct_upstream_distance}")
        assert node_type in ["leaf", "non-leaf", "all"]
        self.node_type = node_type
        res = self.build_edge_info()
        self.dist_hdn_mean, self.dist_hdn_std = res[0]
        self.elev_diff_mean, self.elev_diff_std = res[1]
        self.slope_mean, self.slope_std = res[2]

        assert len(statics_data.shape) == 2  # [sn, c]
        self.statics_mean = statics_data.mean(axis=0, keepdims=True)
        self.statics_std = statics_data.std(axis=0, keepdims=True)
        self.statics_data = (statics_data - self.statics_mean) / (self.statics_std + 1e-8)

        assert isinstance(station_data, np.ndarray) and len(station_data.shape) == 3
        self.station_normalizer = station_normalizer

        print(f">>> we use normalizer: {station_normalizer} for station")
        self.station_data = station_normalizer.normalize(station_data)

        self.history_time_step_num = history_time_step_num
        self.future_time_step_num = future_time_step_num

        self.station_num, self.station_time_step_num, self.station_feature = self.station_data.shape[0], \
                                                                             self.station_data.shape[1], \
                                                                             self.station_data.shape[2]
        self.has_direct_upstream_station_idx_vec = []
        self.no_direct_upstream_station_idx_vec = []
        self.all_station_idx_vec = []
        for direct_upstream_info in self.topology:
            self.all_station_idx_vec.append(direct_upstream_info["idx"])
            if len(direct_upstream_info["direct_up_idx_vec"]) != 0:
                self.has_direct_upstream_station_idx_vec.append(direct_upstream_info["idx"])
            else:
                self.no_direct_upstream_station_idx_vec.append(direct_upstream_info["idx"])
        print(
            f">>> station info, station num: {self.station_num}, leaf: {len(self.no_direct_upstream_station_idx_vec)}, non-leaf: {len(self.has_direct_upstream_station_idx_vec)}, station_time_step_num: {self.station_time_step_num}, station_feature: {self.station_feature}"
        )

        self.check_data()

        self.total_time_step = self.station_time_step_num - (self.history_time_step_num + self.future_time_step_num) + 1
        self.select_station_info = []

        for sn_offset in self.get_used_station_offset_vec():
            for now_time_step_offset in range(self.total_time_step):
                self.select_station_info.append({"now_station_ind": sn_offset, "time_step_start_ind": now_time_step_offset})

    def build_edge_info(self):
        all_dist_hdn = []
        all_elev_diff = []
        all_slope = []
        for item in self.topology:
            all_dist_hdn += item["dist_hdn"]
            all_elev_diff += item["elev_diff"]
            all_slope += item["slope"]

        return (np.mean(all_dist_hdn), np.std(all_dist_hdn)), (np.mean(all_elev_diff), np.std(all_elev_diff)), (np.mean(all_slope), np.std(all_slope))

    def check_data(self):
        print(f">>> check data ...")

        assert np.isnan(self.station_data).astype(np.float32).sum() == 0
        assert np.isnan(self.statics_data).astype(np.float32).sum() == 0

    def get_used_station_offset_vec(self):
        if self.node_type == "all":
            return self.all_station_idx_vec
        elif self.node_type == "leaf":
            return self.no_direct_upstream_station_idx_vec
        else:
            return self.has_direct_upstream_station_idx_vec

    def __len__(self):
        return len(self.select_station_info)

    def get_station_idx_and_time_step_idx(self, global_idx):
        now_station_ind = self.select_station_info[global_idx]["now_station_ind"]
        time_step_start_ind = self.select_station_info[global_idx]["time_step_start_ind"]

        time_step_end_ind = time_step_start_ind + self.history_time_step_num + self.future_time_step_num
        return now_station_ind, time_step_start_ind, time_step_end_ind

    def __getitem__(self, global_idx):

        now_station_ind, time_step_start_ind, time_step_end_ind = self.get_station_idx_and_time_step_idx(global_idx)

        direct_upstream_info = self.topology[now_station_ind]

        res = {
            "now_station_ind": now_station_ind,
            "now_gauge_id": direct_upstream_info["gauge_id"],
            "time_step": [time_step_start_ind, time_step_end_ind],
            "xd": self.station_data[now_station_ind, time_step_start_ind: time_step_end_ind],
            "xs": self.statics_data[now_station_ind],
            "direct_upstream": []
        }
        for i in range(len(direct_upstream_info["direct_up_idx_vec"])):

            res["direct_upstream"].append({
                "direct_up_idx": direct_upstream_info["direct_up_idx_vec"][i],
                "direct_up_gauge_id": direct_upstream_info["direct_up_gauge_id_vec"][i],
                "dist_hdn": (direct_upstream_info["dist_hdn"][i] - self.dist_hdn_mean) / self.dist_hdn_std,
                "elev_diff": (direct_upstream_info["elev_diff"][i] - self.elev_diff_mean) / self.elev_diff_std,
                "slope": (direct_upstream_info["slope"][i] - self.slope_mean) / self.slope_std,
                "stream_flow": self.station_data[direct_upstream_info["direct_up_idx_vec"][i], time_step_start_ind: time_step_end_ind, 0],
                "stream_flow_mean": self.station_data[direct_upstream_info["direct_up_idx_vec"][i], time_step_start_ind: time_step_end_ind, 0].mean()
            })

        if self.max_direct_upstream_distance is not None:
            filter_data = []
            min_distance_ind = 0
            for i in range(len(direct_upstream_info["direct_up_idx_vec"])):
                if direct_upstream_info["dist_hdn"][i] < direct_upstream_info["dist_hdn"][min_distance_ind]:
                    min_distance_ind = i

                if direct_upstream_info["dist_hdn"][i] <= self.max_direct_upstream_distance:
                    filter_data.append(res["direct_upstream"][i])

            if len(filter_data) == 0 and len(res["direct_upstream"]) != 0:
                filter_data = [res["direct_upstream"][min_distance_ind]]

            res["direct_upstream"] = filter_data

        if self.max_direct_upstream_num is not None:
            sorted_data = sorted(res["direct_upstream"], key=lambda x: x["stream_flow_mean"], reverse=True)
            res["direct_upstream"] = sorted_data[:self.max_direct_upstream_num]

        return res


class MixDataset(TorchDataset):
    def __init__(self, sub_basin_datasets: GeneralDatasetWithTopologyDirectUpStream, all_basin_datasets: GeneralDatasetWithTopologyDirectUpStream):
        super().__init__()
        self.sub_basin_datasets = sub_basin_datasets
        self.all_basin_datasets = all_basin_datasets

    def __len__(self):
        return len(self.sub_basin_datasets)

    def __getitem__(self, index):
        sub_basin_info = self.sub_basin_datasets[index]

        time_step_start_ind, time_step_end_ind = sub_basin_info["time_step"]

        for i, item in enumerate(sub_basin_info["direct_upstream"]):

            sub_basin_info["direct_upstream"][i]["xd"] = self.all_basin_datasets.station_data[item["direct_up_idx"], time_step_start_ind: time_step_end_ind]
            sub_basin_info["direct_upstream"][i]["xs"] = self.all_basin_datasets.statics_data[item["direct_up_idx"]]

        return sub_basin_info


class GeneralDatasetForGraph(TorchDataset):
    def __init__(
            self,
            topology: typing.List[dict],
            statics_data: np.ndarray,
            station_data: np.ndarray,
            station_normalizer: typing.Union[MinMaxNormalizer, MeanStdNormalizer],
            history_time_step_num: int = 24,
            future_time_step_num: int = 6,
            max_direct_upstream_num: int = None,
            max_direct_upstream_distance: float = None,
            root_station_ind: int = 288
    ):
        super().__init__()
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
        print(f"\n>>> we set max_direct_upstream_num: {max_direct_upstream_num}, max_direct_upstream_distance: {max_direct_upstream_distance}")

        """build dynamic data"""
        assert isinstance(station_data, np.ndarray) and len(station_data.shape) == 3
        self.station_normalizer = station_normalizer

        print(f">>> we use normalizer: {station_normalizer} for station")
        self.station_data = station_normalizer.normalize(station_data)

        """build statics data"""
        assert len(statics_data.shape) == 2  # [sn, c]
        self.statics_mean = statics_data.mean(axis=0, keepdims=True)
        self.statics_std = statics_data.std(axis=0, keepdims=True)
        self.statics_data = (statics_data - self.statics_mean) / (self.statics_std + 1e-8)

        """build edge info"""
        res = self.build_edge_info()
        self.dist_hdn_mean, self.dist_hdn_std = res[0]
        self.elev_diff_mean, self.elev_diff_std = res[1]
        self.slope_mean, self.slope_std = res[2]

        self.edge_features, self.src2dst = self.build_edge_features()

        """others"""
        self.history_time_step_num = history_time_step_num
        self.future_time_step_num = future_time_step_num

        self.station_num, self.station_time_step_num, self.station_feature_dim = self.station_data.shape

        print(
            f">>> station info, station num: {self.station_num}, leaf: {len(self.no_direct_upstream_station_idx_vec)}, non-leaf: {len(self.has_direct_upstream_station_idx_vec)}, edge_num: {self.edge_features.shape[0]}, station_time_step_num: {self.station_time_step_num}, station_feature_dim: {self.station_feature_dim}"
        )

        self.check_data()

        self.total_time_step = self.station_time_step_num - (self.history_time_step_num + self.future_time_step_num) + 1

    def build_edge_info(self):
        all_dist_hdn = []
        all_elev_diff = []
        all_slope = []
        for item in self.topology:
            all_dist_hdn += item["dist_hdn"]
            all_elev_diff += item["elev_diff"]
            all_slope += item["slope"]

        return (np.mean(all_dist_hdn), np.std(all_dist_hdn)), (np.mean(all_elev_diff), np.std(all_elev_diff)), (np.mean(all_slope), np.std(all_slope))

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
        src_station_ind_2_edge_level = [max_node_level - node_level for node_level in station_ind_2_node_level]
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
                    "dist_hdn": (direct_upstream_info["dist_hdn"][i] - self.dist_hdn_mean) / self.dist_hdn_std,
                    "elev_diff": (direct_upstream_info["elev_diff"][i] - self.elev_diff_mean) / self.elev_diff_std,
                    "slope": (direct_upstream_info["slope"][i] - self.slope_mean) / self.slope_std,
                    "stream_flow_mean": self.station_data[direct_upstream_info["direct_up_idx_vec"][i], :, 0].mean()
                })

            if self.max_direct_upstream_distance is not None:
                filter_data = []
                min_distance_ind = 0
                for i in range(len(direct_upstream_info["direct_up_idx_vec"])):
                    if direct_upstream_info["dist_hdn"][i] < direct_upstream_info["dist_hdn"][min_distance_ind]:
                        min_distance_ind = i

                    if direct_upstream_info["dist_hdn"][i] <= self.max_direct_upstream_distance:
                        filter_data.append(cache[i])

                if len(filter_data) == 0 and len(cache) != 0:
                    filter_data = [cache[min_distance_ind]]

                cache = filter_data

            if self.max_direct_upstream_num is not None:
                sorted_data = sorted(cache, key=lambda x: x["stream_flow_mean"], reverse=True)
                cache = sorted_data[:self.max_direct_upstream_num]

            for item in cache:
                edge_features_vec.append([item['dist_hdn'], item['elev_diff'], item['slope']])
                src2dst_vec.append(
                    [
                        item['direct_up_idx'],
                        now_station_ind,
                        src_station_ind_2_edge_level[item['direct_up_idx']]
                    ]
                )

        return np.array(edge_features_vec, dtype=np.float32), np.array(src2dst_vec, dtype=np.int32)

    def check_data(self):
        print(f">>> check data ...")

        assert np.isnan(self.station_data).astype(np.float32).sum() == 0
        assert np.isnan(self.statics_data).astype(np.float32).sum() == 0

    def get_used_station_offset_vec(self):
        return self.all_station_idx_vec

    def __len__(self):
        return self.total_time_step


    def __getitem__(self, time_step_start_ind):
        time_step_end_ind = time_step_start_ind + self.history_time_step_num + self.future_time_step_num

        return {
            "sub_basin_weather": torch.tensor(self.station_data[:, time_step_start_ind: time_step_end_ind, 1:], dtype=torch.float32),
            "station_stream_flow": torch.tensor(self.station_data[:, time_step_start_ind: time_step_end_ind, 0], dtype=torch.float32),
            "time_step": torch.tensor(time_step_start_ind, dtype=torch.long),
        }


class MixDatasetForGraph(TorchDataset):
    def __init__(self, sub_basin_datasets: GeneralDatasetForGraph, all_basin_datasets: GeneralDatasetForGraph):
        super().__init__()
        self.sub_basin_datasets = sub_basin_datasets
        self.all_basin_datasets = all_basin_datasets

    def __len__(self):
        return len(self.sub_basin_datasets)

    def __getitem__(self, index):
        sub_basin_info = self.sub_basin_datasets[index]
        all_basin_info = self.all_basin_datasets[index]

        return {
            "sub_basin_weather": sub_basin_info["sub_basin_weather"],
            "all_basin_weather": all_basin_info["sub_basin_weather"],
            "station_stream_flow": sub_basin_info["station_stream_flow"],
            "time_step": sub_basin_info["time_step"]
        }


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
        res = self.build_edge_info()
        self.dist_hdn_mean, self.dist_hdn_std = res[0]
        self.elev_diff_mean, self.elev_diff_std = res[1]
        self.slope_mean, self.slope_std = res[2]

        self.station_offset_to_node_level = []
        self.edge_features, self.src2dst = self.build_edge_features()

        """others"""
        self.history_time_step_num = history_time_step_num
        self.future_time_step_num = future_time_step_num

        self.station_num, self.station_time_step_num, self.station_feature_dim = self.station_data.shape

        if self.rank == 0:
            print(
                f">>> station info, station num: {self.station_num}, leaf: {len(self.no_direct_upstream_station_idx_vec)}, non-leaf: {len(self.has_direct_upstream_station_idx_vec)}, edge_num: {self.edge_features.shape[0]}, station_time_step_num: {self.station_time_step_num}, station_feature_dim: {self.station_feature_dim}"
            )

        self.check_data()

        each_group_data_length = self.history_time_step_num + self.future_time_step_num
        self.time_stamps_group_ind_vec = []
        for i in range(0, self.station_time_step_num):
            max_forecast_need_time_stamp_length = each_group_data_length + self.max_forecast_step_num - 1
            if i + max_forecast_need_time_stamp_length > self.station_time_step_num:
                break
            else:
                self.time_stamps_group_ind_vec.append([i, i + each_group_data_length])

        self.total_time_step = len(self.time_stamps_group_ind_vec)

    def build_edge_info(self):
        all_dist_hdn = []
        all_elev_diff = []
        all_slope = []
        for item in self.topology:
            all_dist_hdn += item["dist_hdn"]
            all_elev_diff += item["elev_diff"]
            all_slope += item["slope"]

        return (np.mean(all_dist_hdn), np.std(all_dist_hdn)), (np.mean(all_elev_diff), np.std(all_elev_diff)), (np.mean(all_slope), np.std(all_slope))

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
                    "dist_hdn": (direct_upstream_info["dist_hdn"][i] - self.dist_hdn_mean) / self.dist_hdn_std,
                    "elev_diff": (direct_upstream_info["elev_diff"][i] - self.elev_diff_mean) / self.elev_diff_std,
                    "slope": (direct_upstream_info["slope"][i] - self.slope_mean) / self.slope_std,
                    "stream_flow_mean": self.station_data[direct_upstream_info["direct_up_idx_vec"][i], :, 0].mean()
                })

            if self.max_direct_upstream_distance is not None:
                filter_data = []
                min_distance_ind = 0
                for i in range(len(direct_upstream_info["direct_up_idx_vec"])):
                    if direct_upstream_info["dist_hdn"][i] < direct_upstream_info["dist_hdn"][min_distance_ind]:
                        min_distance_ind = i

                    if direct_upstream_info["dist_hdn"][i] <= self.max_direct_upstream_distance:
                        filter_data.append(cache[i])

                if len(filter_data) == 0 and len(cache) != 0:
                    filter_data = [cache[min_distance_ind]]

                cache = filter_data

            if self.max_direct_upstream_num is not None:
                sorted_data = sorted(cache, key=lambda x: x["stream_flow_mean"], reverse=True)
                cache = sorted_data[:self.max_direct_upstream_num]

            for item in cache:
                edge_features_vec.append([item['dist_hdn'], item['elev_diff'], item['slope']])
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
            "basin_weather": torch.tensor(self.station_data[:, time_step_start_ind: time_step_end_ind, 1:], dtype=torch.float32),
            "future_basin_weather": torch.tensor(self.future_station_data[:, time_step_end_ind-1, :, 1:], dtype=torch.float32),
            "station_stream_flow": torch.tensor(self.station_data[:, time_step_start_ind: time_step_end_ind, 0], dtype=torch.float32),
            "future_station_stream_flow": torch.tensor(self.future_station_data[:, time_step_end_ind-1, :, 0], dtype=torch.float32),
            "time_step": torch.tensor(time_step_start_ind, dtype=torch.long),
        }


class MixDatasetForFutureGraph(TorchDataset):
    def __init__(self, sub_basin_datasets: GeneralDatasetForFutureGraph, all_basin_datasets: GeneralDatasetForFutureGraph):
        super().__init__()
        self.sub_basin_datasets = sub_basin_datasets
        self.all_basin_datasets = all_basin_datasets

    def __len__(self):
        return len(self.sub_basin_datasets)

    def __getitem__(self, index):
        sub_basin_info = self.sub_basin_datasets[index]
        all_basin_info = self.all_basin_datasets[index]

        return {
            "station_stream_flow": sub_basin_info["station_stream_flow"],
            "future_station_stream_flow": sub_basin_info["future_station_stream_flow"],

            "sub_basin_weather": sub_basin_info["basin_weather"],
            "future_sub_basin_weather": sub_basin_info["future_basin_weather"],
            "all_basin_weather": all_basin_info["basin_weather"],
            "future_all_basin_weather": all_basin_info["future_basin_weather"],

            "time_step": sub_basin_info["time_step"]
        }
