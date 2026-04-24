import yaml
import numpy as np
import os
import torch
import random
import collections
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
# from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import gc
import warnings
from functools import partial
import matplotlib.pyplot as plt
import torch.nn as nn
from stream_tool.config_module import MixConfig, build_config_from_args, build_default_args_parser
from stream_tool import nh_metric
from stream_tool.other_tools import TimeCount
from data_define.hydrodynamic_dataset import MeanStdNormalizer, GeneralDatasetForFutureGraph, MixDatasetForFutureGraph, BlendDynamicDataset
import utils
import torch.utils.checkpoint as cp
from mamba_ssm import Mamba2
import pandas as pd
import shutil
from torch_scatter import scatter
import deepspeed


LAYER_NORM_TYPE = nn.LayerNorm


def set_seed_and_device(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)


def build_dataset(dataset_config_dict, sub_all: str, is_train: bool = False, station_normalizer=None) -> GeneralDatasetForFutureGraph:
    assert sub_all in ["sub", "all"]
    if sub_all == "all":
        root_dir = dataset_config_dict["all_basin_data_dir"]
        future_dir = dataset_config_dict["future_all_basin_data_dir"]
    else:
        root_dir = dataset_config_dict["sub_basin_data_dir"]
        future_dir = dataset_config_dict["future_sub_basin_data_dir"]

    if is_train:
        years = dataset_config_dict["train_years"]
        assert station_normalizer is None
        station_normalizer = MeanStdNormalizer()
    else:
        years = dataset_config_dict["val_years"]

    station_vec = []
    future_station_vec = []

    all_time_stamps_vec = []
    for year_id, year in enumerate(years):
        gt_station_data = np.load(os.path.join(root_dir, f"{year}", "station.bin.npy"))
        try:
            future_station_data = np.load(os.path.join(future_dir, f"{year}", "station.bin.npy"))
        except Exception as e:
            future_station_data = np.load(os.path.join(future_dir, f"{year}_epand_with_era5_land", "station.bin.npy"))

        real_data_time_stamp_len = min(gt_station_data.shape[1], future_station_data.shape[1])

        now_year_time_stamps = pd.date_range(f"{year}-01-01T00", f"{year}-12-31T18", freq=args.time_stamps_frequency)
        if len(now_year_time_stamps) != real_data_time_stamp_len:
            if year_id == 0:
                all_time_stamps_vec.append(now_year_time_stamps[-real_data_time_stamp_len:])
                gt_station_data = gt_station_data[:, -real_data_time_stamp_len:]
                future_station_data = future_station_data[:, -real_data_time_stamp_len:]
            elif year_id == len(years) - 1:
                all_time_stamps_vec.append(now_year_time_stamps[0:real_data_time_stamp_len])
                gt_station_data = gt_station_data[:, 0:real_data_time_stamp_len]
                future_station_data = future_station_data[:, 0:real_data_time_stamp_len]
            else:
                print(f"\n>>> year: {year} gt_station_data is : {gt_station_data.shape}, future_station_data is: {future_station_data.shape}", flush=True)
                raise RuntimeError('error')
        else:
            all_time_stamps_vec.append(now_year_time_stamps)
            
        station_vec.append(gt_station_data)
        future_station_vec.append(future_station_data)

    station_data = np.concatenate(station_vec, axis=1)
    future_data = np.concatenate(future_station_vec, axis=1)
    all_time_stamps = np.concatenate(all_time_stamps_vec)
    if RANK == 0:
        print(f"\n>>> all_time_stamps: {all_time_stamps.shape} station_data is : {station_data.shape}, future_data is: {future_data.shape}", flush=True)
    
    if not is_train:
        eval_step_num = dataset_config_dict["evaluation_max_time_steps"]
        if eval_step_num is None:
            if RANK == 0:
                print(f"\n>>> evaluation may cost too much time, be carefully", flush=True)
        else:

            if station_data.shape[1] >= eval_step_num:
                station_data = station_data[:, -eval_step_num:, :]
                future_data = future_data[:, -eval_step_num:, :]
                all_time_stamps = all_time_stamps[-eval_step_num:]

            if RANK == 0:
                print(f"\n>>> evaluation may cost too much time, so we just eval last {eval_step_num} time_steps, time_stamp from {all_time_stamps[0]} to {all_time_stamps[-1]}", flush=True)

    topology = utils.read_info_from_json_or_json_line(os.path.join(root_dir, "topology.json"))
    statics = utils.read_info_from_json_or_json_line(os.path.join(root_dir, "static.json"))
    statics_vec = []
    for item in statics:
        cache = []
        for key, value in item.items():
            if key not in ["idx", "gauge_id"]:
                cache.append(value)

        statics_vec.append(cache)
    statics = np.array(statics_vec, dtype=np.float32)

    return GeneralDatasetForFutureGraph(
        topology=topology,
        statics_data=statics,
        station_data=station_data,
        future_station_data=future_data,
        time_stamps=all_time_stamps,
        station_normalizer=station_normalizer,
        history_time_step_num=dataset_config_dict["history_time_step_num"] + 1,
        future_time_step_num=dataset_config_dict["future_time_step_num"],
        max_forecast_step_num=future_data.shape[2],
        max_direct_upstream_distance=dataset_config_dict["max_direct_upstream_distance"],
        root_station_ind=dataset_config_dict["root_offset_ind"]
    )


class NaiveMLP(nn.Module):
    def __init__(self, in_dim: int, mid_dim, out_dim, hidden_layers=1, use_norm: bool = True):
        super().__init__()
        self.out_dim = out_dim

        vec = [
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.SiLU(inplace=True),
        ]
        for _ in range(hidden_layers - 1):
            vec += [
                nn.Linear(mid_dim, mid_dim, bias=True),
                nn.SiLU(inplace=True),
            ]
        vec.append(nn.Linear(mid_dim, out_dim, bias=True))

        self.model = nn.Sequential(*vec)
        if use_norm:
            self.norm = LAYER_NORM_TYPE(out_dim)
        else:
            self.norm = None

    def forward(self, x):
        if self.norm:
            return self.norm(self.model(x))
        else:
            return self.model(x)


class SelfLstm(nn.Module):
    def __init__(self, num_hidden, num_layers=1):
        super().__init__()
        self.self_lstm = nn.LSTM(input_size=num_hidden, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)

    def forward(self, hs):
        output, _ = self.self_lstm(hs)
        return output[:, -1]


class EdgeResidualPredict(torch.nn.Module):
    def __init__(self, time_step, edge_feature_dim, mid_dim, out_dim, used_time_step_num: int = None, time_embed_dim: int = 32, use_norm: bool = True):
        super().__init__()
        self.time_step = time_step
        if used_time_step_num is not None:
            assert used_time_step_num <= time_step
            self.used_time_step_num = used_time_step_num
        else:
            self.used_time_step_num = time_step

        # self.time_delta_embed = nn.Parameter(data=torch.randn(size=(self.used_time_step_num, time_embed_dim)), requires_grad=True)

        self.edge_info_embed_layer = nn.Sequential(
            nn.Linear(1 + edge_feature_dim, mid_dim),
            nn.SiLU(),
            nn.LayerNorm(mid_dim),
            SelfLstm(mid_dim),
            nn.Linear(mid_dim, mid_dim),
        )

    def forward(self, x_stream_flow: torch.Tensor, edge_features: torch.Tensor):
        """
        :param x_stream_flow: [b, n, time_step]
        :param edge_features: [b, n, edge_feature_dim]
        :return:
        """
        assert x_stream_flow.shape[-1] == self.time_step
        x_stream_flow = x_stream_flow[..., -self.used_time_step_num:]
        b, n, t = x_stream_flow.shape

        x_stream_flow = x_stream_flow.unsqueeze(dim=-1)  # [b, n, t, 1]

        edge_features = edge_features.unsqueeze(dim=2)   # [b, n, 1, edge_feature_dim]
        edge_features = edge_features.repeat(repeats=(1, 1, t, 1))   # [b, n, t, edge_feature_dim]

        # time_delta_embed = self.time_delta_embed[None, None]  # [1, 1, t, edge_feature_dim]
        # time_delta_embed = time_delta_embed.repeat(repeats=(b, n, 1, 1))  # [b, n, t, edge_feature_dim]

        features = torch.cat([x_stream_flow, edge_features], dim=-1)
        features = features.view(b*n, t, -1)
        features = self.edge_info_embed_layer(features)  # [b*n, mid]
        return features.view(b, n, -1)


class BasinPredict(torch.nn.Module):
    def __init__(self, time_step, xd_feature_dim, xs_feature_dim, mid_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Linear(xd_feature_dim + xs_feature_dim, mid_dim),
            nn.SiLU(),
            nn.LayerNorm(mid_dim),
            SelfLstm(num_hidden=mid_dim),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, xd: torch.Tensor, xs: torch.Tensor):
        """

        :param xd: [b, sn, time_step, xd_feature_dim]
        :param xs: [b, sn, xs_feature_dim]
        :return:
        """
        assert len(xd.shape) == 4 and len(xs.shape) == 3
        b, sn, t, c1 = xd.shape

        xs = xs[:, :, None, :].repeat(repeats=(1, 1, t, 1))
        x = torch.cat([xd, xs], dim=-1)
        x = x.view(b*sn, t, -1)
        out = self.model(x)   # [b*sn, out_dim]
        return out.view(b, sn, self.out_dim)


class STREAM(nn.Module):
    def __init__(
            self,
            time_step,
            sub_basin_weather_dim,
            sub_basin_statics_dim,
            hidden_dim,
            edge_features_dim=3,
            edge_features_used_time_step=None,
            edge_features_time_embed_dim: int = 32,
            message_passing_num: int = 20,
            use_checkpointing: bool = False,
            station_num_vec: list = None
    ):
        super().__init__()

        self.message_passing_num = message_passing_num
        self.use_checkpointing = use_checkpointing

        self.edge_embed_layer = NaiveMLP(in_dim=edge_features_dim, mid_dim=hidden_dim, out_dim=hidden_dim, use_norm=True)
        self.sub_basin_model = BasinPredict(time_step=time_step, xd_feature_dim=sub_basin_weather_dim, xs_feature_dim=sub_basin_statics_dim, mid_dim=hidden_dim, out_dim=hidden_dim)

        self.edge_model = EdgeResidualPredict(
            time_step=time_step, edge_feature_dim=hidden_dim, mid_dim=hidden_dim, out_dim=hidden_dim,
            used_time_step_num=edge_features_used_time_step,
            time_embed_dim=edge_features_time_embed_dim,
            use_norm=True
        )

        self.node_model = NaiveMLP(in_dim=hidden_dim + hidden_dim, mid_dim=hidden_dim, out_dim=hidden_dim, use_norm=True)

        self.head = NaiveMLP(in_dim=hidden_dim, mid_dim=hidden_dim, out_dim=1, use_norm=False)

    def _balance_water(self, predict_from_sub_basin_water, predict_from_all_basin_water, dataset_ind):

        predict_stream_flow = torch.mean(predict_from_sub_basin_water, dim=-1, keepdim=True)
        return predict_stream_flow

    def forecast_one_step(self, inputs):
        def custom_forward(dataset_ind, src, dst, sub_basin_weather, sub_basin_statics, all_basin_weather, all_basin_statics, station_stream_flow, non_leaf_mask, edge_embedding):
            station_weather_embedding = self.sub_basin_model(sub_basin_weather, sub_basin_statics)
            # predict_from_all_basin_water = self.all_basin_model(all_basin_weather, all_basin_statics)
            predict_from_all_basin_water = None
            # (b, station_num, t, c1)   (b, station_num, c2)   -->   (b, station_num, c)

            for layer_ind in range(self.message_passing_num):
                """update edge using time-series stream-flow and edge_embedding"""
                predict_stream_flow = self._balance_water(self.head(station_weather_embedding), predict_from_all_basin_water, dataset_ind)
                replaced_stream_flow = torch.cat([station_stream_flow, predict_stream_flow], dim=-1)

                # edge_update_info = self.edge_model_vec[layer_ind](replaced_stream_flow[:, self.src], edge_embedding)
                edge_update_info = self.edge_model(replaced_stream_flow[:, src], edge_embedding)
                # (b, edge_num, t)  (b, edge_num, c) --> (b, edge_num, c)

                edge_embedding = edge_embedding + edge_update_info
                # (b, edge_num, c)

                """update node using edge"""
                edge_agg = scatter(edge_embedding, dst, dim=1, dim_size=station_weather_embedding.shape[1], reduce='sum')
                # (b, station_num, c)

                node_update_info = torch.cat([station_weather_embedding, edge_agg], dim=-1)
                # node_update_info = self.node_model_vec[layer_ind](node_update_info)
                node_update_info = self.node_model(node_update_info)
                # (b, station_num, c + c)  --> (b, station_num, c)

                station_weather_embedding = station_weather_embedding + non_leaf_mask * node_update_info

            out = self._balance_water(self.head(station_weather_embedding), predict_from_all_basin_water, dataset_ind)
            # (b, station_num, c)  --> (b, station_num, 1)
            return out
        if self.use_checkpointing:
            return cp.checkpoint(custom_forward, *inputs, use_reentrant=True)
        else:
            return custom_forward(*inputs)

    def forward(self, dataset_ind, non_leaf_mask, src, dst, edge_features, sub_basin_statics, sub_basin_weather, future_sub_basin_weather, all_basin_statics, all_basin_weather, future_all_basin_weather, station_stream_flow, max_forecast_step_num=0):
        sub_basin_statics = sub_basin_statics.squeeze(0)
        sub_basin_weather = sub_basin_weather.squeeze(0)
        future_sub_basin_weather = future_sub_basin_weather.squeeze(0)
        all_basin_statics = all_basin_statics.squeeze(0)
        all_basin_weather = all_basin_weather.squeeze(0)
        future_all_basin_weather = future_all_basin_weather.squeeze(0)
        station_stream_flow = station_stream_flow.squeeze(0)

        b = sub_basin_weather.shape[0]
        non_leaf_mask = non_leaf_mask[..., None]
        edge_features = edge_features[0]
        src = src[0]
        dst = dst[0]

        edge_embedding = self.edge_embed_layer(edge_features)
        # (edge_num, c3)  -->   (edge_num, c)
        edge_embedding = edge_embedding[None, ...].repeat(repeats=(b, 1, 1))
        # (edge_num, c)  -->   (b, edge_num, c)

        zero_day_sub_basin_weather = sub_basin_weather[:, :, 0:-1]
        forecast_day_sub_basin_weather = sub_basin_weather[:, :, 1:]

        zero_day_all_basin_weather = all_basin_weather[:, :, 0:-1]
        forecast_day_all_basin_weather  = all_basin_weather [:, :, 1:]

        zero_day_station_stream_flow = station_stream_flow[:, :, 0:-1]
        forecast_day_station_stream_flow = station_stream_flow[:, :, 1:]

        model_output_vec_for_0day = []
        now_input_sub_basin_weather = zero_day_sub_basin_weather
        now_input_all_basin_weather = zero_day_all_basin_weather
        now_input_station_stream_flow = zero_day_station_stream_flow[:, :, 0:-1]
        predict_stream_flow = self.forecast_one_step((dataset_ind, src, dst, now_input_sub_basin_weather, sub_basin_statics, now_input_all_basin_weather, all_basin_statics, now_input_station_stream_flow, non_leaf_mask, edge_embedding))
        model_output_vec_for_0day.append(predict_stream_flow)

        model_output_vec_for_future_vec = []
        for forecast_step_ind in range(max_forecast_step_num):
            now_input_sub_basin_weather = torch.cat([forecast_day_sub_basin_weather[:, :, forecast_step_ind:-1, :], future_sub_basin_weather[:, :, :forecast_step_ind + 1, :]], dim=2)
            now_input_all_basin_weather = torch.cat([forecast_day_all_basin_weather[:, :, forecast_step_ind:-1, :], future_all_basin_weather[:, :, :forecast_step_ind + 1, :]], dim=2)
            now_input_station_stream_flow = torch.cat([forecast_day_station_stream_flow[:, :, forecast_step_ind:-1]] + model_output_vec_for_future_vec, dim=2)
            """last-time-stamp stream_flow will be predicted"""
            predict_stream_flow = self.forecast_one_step((dataset_ind, src, dst, now_input_sub_basin_weather, sub_basin_statics, now_input_all_basin_weather, all_basin_statics, now_input_station_stream_flow, non_leaf_mask, edge_embedding))
            model_output_vec_for_future_vec.append(predict_stream_flow)

        return torch.cat(model_output_vec_for_0day, dim=-1), torch.cat(model_output_vec_for_future_vec, dim=-1)


def save_checkpoint(epoch, global_iteration, engine: deepspeed.DeepSpeedEngine, now_epoch_is_best: bool = False):

    checkpoint_save_dir_vec = [
        os.path.join(os.path.join(EXP_CONFIG.save_dir, "checkpoints"), 'all_epoch_' + f'{epoch:0>8}' + '_iter_' + f'{global_iteration:0>8}'),
    ]
    if now_epoch_is_best:
        checkpoint_save_dir_vec.append(
            os.path.join(os.path.join(EXP_CONFIG.save_dir, "best_checkpoint"), 'all_epoch_' + f'{epoch:0>8}' + '_iter_' + f'{global_iteration:0>8}'),
        )
    if dist.get_rank() == 0:
        record = {
            "epoch": epoch,
            "global_iteration": global_iteration
        }
        for save_dir in checkpoint_save_dir_vec:
            utils.write_info_to_json([record], os.path.join(save_dir, "record.json"))

            if dist.get_rank() == 0:
                sd_dir = os.path.join(save_dir, 'sd')
                os.makedirs(sd_dir, exist_ok=True)
                sd = engine.state_dict()
                torch.save(sd, os.path.join(sd_dir, f"rank_{0:0>3}.pth"))
                del sd

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()


def load_checkpoint(model: STREAM, reload_checkpoint_from, device):
    if reload_checkpoint_from not in ["", None]:
        load_from = os.path.join(reload_checkpoint_from, "sd", "rank_000.pth")
        sd = torch.load(
            load_from,
            map_location=device,
            weights_only=False
        )

        new_sd = {}
        inited_sd = model.state_dict()
        for k, v in sd.items():
            new_k = k[7:]
            if args.just_evaluation or "balance_weight" not in new_k:
                new_sd[new_k] = v
            else:
                """balance_weight will be reset during train-phase"""
                # new_sd[new_k] = inited_sd[new_k]

        if args.just_evaluation != 1:
            print(f"\n>>> balance_weight will be reset during train-phase", flush=True)
            for k, v in inited_sd.items():
                if "balance_weight" in k:
                    new_sd[k] = v

        model.load_state_dict(new_sd, strict=True)
        del sd
        gc.collect()
        torch.cuda.empty_cache()
        if dist.get_rank() == 0:
            print(f"\n>>> load from {load_from} success !!! ", flush=True)


def compute_predict_metric(epoch_idx, dataset, pred_np, gt_np, now_forecast_step_num: int):
    now_station_idx_vec = dataset.sub_basin_datasets.get_used_station_offset_vec()
    all_metric = nh_metric.call_all_metric(
        pred_np,
        gt_np,
        metric_name_list=nh_metric.nh_eval_metric.get_available_metrics() # + ["Acc-0.03", "Acc-0.04", "Acc-0.10", "Acc-0.15", "Acc-0.20", "Refine-Peak-Timing"]
    )

    if dist.get_rank() == 0:
        figs_save_dir = os.path.join(EXP_CONFIG.save_dir, f"figs/epoch_{epoch_idx:0>3}/forecast_step_{now_forecast_step_num:0>3}")
        os.makedirs(figs_save_dir, exist_ok=True)
        time_step_num, sn_num = pred_np.shape[0], pred_np.shape[1]
        pred_np = np.reshape(pred_np, newshape=(time_step_num, sn_num))
        gt_np = np.reshape(gt_np, newshape=(time_step_num, sn_num))
        for sn_idx in range(sn_num):
            plt.figure(figsize=(30, 6))
            plt.plot(pred_np[:, sn_idx], label='pred')
            plt.plot(gt_np[:, sn_idx], label='gt')

            plt.tight_layout()
            plt.legend()

            offset_id = now_station_idx_vec[sn_idx]
            gauge_id = dataset.sub_basin_datasets.topology[offset_id]["gauge_id"]
            plt.savefig(f'{figs_save_dir}/gauge_{gauge_id:0>3}.png')
            plt.close()

    total_loss = np.mean((pred_np - gt_np) ** 2)

    print_info = "\n" + "*" * 50 + "\n"
    print_info += f"epoch: {epoch_idx}, now_forecast_step_num: {now_forecast_step_num} \n"
    print_info += f"loss: {total_loss} \n\n"

    assert "NSE" in all_metric.keys()
    nse_good_mask = all_metric["NSE"] > 0
    nse_bad_mask = all_metric["NSE"] <= 0

    record = collections.defaultdict()

    for metric_name, metric_values in all_metric.items():

        if metric_name in ["NSE"]:
            good = metric_values[nse_good_mask]
            bad = metric_values[nse_bad_mask]
            print_info += f"metric_name: {metric_name} \n\t-metric_shape: {metric_values.shape} \n\t-metric_values: {metric_values.mean():.6f} \n\t-good: {good.shape[0]} \n\t-bad: {bad.shape[0]} \n\t-good station nse: {good.mean():.6f} \n\n"
            record["good_station_nse"] = good.mean()
            record['good_station_num'] = good.shape[0]

        elif metric_name in ["Refine-Peak-Timing"]:

            peak_time_diff_vec = []
            peak_value_diff_vec = []
            peak_pred_vec = []
            peak_gt_vec = []
            for sn_idx in range(len(metric_values)):
                peak_time_diff_vec += metric_values[sn_idx][0]
                peak_value_diff_vec += metric_values[sn_idx][1]
                peak_gt_vec += metric_values[sn_idx][2]
                peak_pred_vec += metric_values[sn_idx][3]

            if len(peak_time_diff_vec) > 0:
                peak_time_diff_np = np.array(peak_time_diff_vec, dtype=np.float32)
                peak_value_diff_np = np.array(peak_value_diff_vec, dtype=np.float32)

                """level"""
                rate = [60, 20, 15, 2.5, 1.5, 1]
                level_num = len(rate)
                peak_level_vec = nh_metric.get_peak_level(peak_gt_vec, level_num=level_num, rate=rate)
                peak_level_np = np.array(peak_level_vec, dtype=np.float32)
                peak_gt_np = np.array(peak_gt_vec, dtype=np.float32)
                for level in range(1, level_num + 1):
                    mask = peak_level_np == level
                    select_peak_time_np = peak_time_diff_np[mask]
                    select_peak_value_np = peak_value_diff_np[mask]
                    select_peak_gt = peak_gt_np[mask]

                    select_peak_predict_eq_zero_mask = select_peak_time_np == 0

                    select_peak_predict_eq_zero_num = select_peak_predict_eq_zero_mask.astype(dtype=np.float32).sum()

                    print_info += f"metric_name: {metric_name}, level: {level}({select_peak_time_np.shape[0] / peak_time_diff_np.shape[0]:.2%})"
                    print_info += f"\n\t-Peak-Num: {select_peak_time_np.shape[0]}, [{select_peak_gt.min()}, {select_peak_gt.max()}]"

                    if select_peak_predict_eq_zero_num > 0:
                        print_info += f"\n\t-Peak-EQ-Zero: {select_peak_predict_eq_zero_num}({select_peak_predict_eq_zero_num / select_peak_time_np.shape[0]:.6f}), Peak-Value-Diff: {select_peak_value_np[select_peak_predict_eq_zero_mask].mean():.6f}\n\n"
                        record[f'Level-{level}-PeakAcc'] = select_peak_predict_eq_zero_num / select_peak_time_np.shape[0]
                        record[f'Level-{level}-PeakError'] = select_peak_value_np[select_peak_predict_eq_zero_mask].mean()
                    else:
                        print_info += f"\n\t-Peak-EQ-Zero: {select_peak_predict_eq_zero_num}({select_peak_predict_eq_zero_num / select_peak_time_np.shape[0]:.6f})\n\n"
                        record[f'Level-{level}-PeakAcc'] = select_peak_predict_eq_zero_num / select_peak_time_np.shape[0]
                        record[f'Level-{level}-PeakError'] = -1
                """global"""
                peak_predict_lt_zero_mask = peak_time_diff_np < 0
                peak_predict_eq_zero_mask = peak_time_diff_np == 0
                peak_predict_gt_zero_mask = (peak_time_diff_np > 0) & (peak_time_diff_np < 1)

                peak_time_diff_np = np.abs(peak_time_diff_np)
                peak_predict_lt_zero_num = peak_predict_lt_zero_mask.astype(dtype=np.float32).sum()
                peak_predict_eq_zero_num = peak_predict_eq_zero_mask.astype(dtype=np.float32).sum()
                peak_predict_gt_zero_num = peak_predict_gt_zero_mask.astype(dtype=np.float32).sum()

                print_info += f"metric_name: {metric_name}"
                print_info += f"\n\t-Peak-Num: {peak_time_diff_np.shape[0]}"

                if peak_predict_lt_zero_num > 0:
                    print_info += f"\n\t-Peak-LT-Zero: {peak_predict_lt_zero_num}({peak_predict_lt_zero_num / peak_time_diff_np.shape[0]:.6f}), Peak-Time-Diff: {peak_time_diff_np[peak_predict_lt_zero_mask].mean()}, Peak-Value-Diff: {peak_value_diff_np[peak_predict_lt_zero_mask].mean():.6f}"
                else:
                    print_info += f"\n\t-Peak-LT-Zero: {peak_predict_lt_zero_num}({peak_predict_lt_zero_num / peak_time_diff_np.shape[0]:.6f})"

                if peak_predict_eq_zero_num > 0:
                    print_info += f"\n\t-Peak-EQ-Zero: {peak_predict_eq_zero_num}({peak_predict_eq_zero_num / peak_time_diff_np.shape[0]:.6f}), Peak-Time-Diff: {peak_time_diff_np[peak_predict_eq_zero_mask].mean()}, Peak-Value-Diff: {peak_value_diff_np[peak_predict_eq_zero_mask].mean():.6f}"
                    record[f'PeakAcc'] = peak_predict_eq_zero_num / peak_time_diff_np.shape[0]
                    record[f'PeakError'] = peak_value_diff_np[peak_predict_eq_zero_mask].mean()
                else:
                    print_info += f"\n\t-Peak-EQ-Zero: {peak_predict_eq_zero_num}({peak_predict_eq_zero_num / peak_time_diff_np.shape[0]:.6f})"
                    record[f'PeakAcc'] = peak_predict_eq_zero_num / peak_time_diff_np.shape[0]
                    record[f'PeakError'] = -1

                if peak_predict_gt_zero_num > 0:
                    print_info += f"\n\t-Peak-GT-Zero: {peak_predict_gt_zero_num}({peak_predict_gt_zero_num / peak_time_diff_np.shape[0]:.6f}), Peak-Time-Diff: {peak_time_diff_np[peak_predict_gt_zero_mask].mean()}, Peak-Value-Diff: {peak_value_diff_np[peak_predict_gt_zero_mask].mean():.6f}\n\n"
                else:
                    print_info += f"\n\t-Peak-GT-Zero: {peak_predict_gt_zero_num}({peak_predict_gt_zero_num / peak_time_diff_np.shape[0]:.6f})\n\n"

            else:
                print_info += f"metric_name: {metric_name}, no peaks to compute metric \n\n"
        else:
            record[metric_name] = metric_values[nse_good_mask].mean()
            print_info += f"metric_name: {metric_name} \n\t-metric_shape: {metric_values.shape} \n\t-metric_values: {metric_values[nse_good_mask].mean():.6f} \n\n"

    print_info += "*" * 50
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"\nvalidation results:", flush=True)
        print(print_info, flush=True)

    record["print_info"] = print_info
    record["epoch"] = epoch_idx
    record['mix'] = record['good_station_nse'] + record['KGE']
    return record


@torch.no_grad()
def compute_predict_results(epoch_idx, model, dataset: MixDatasetForFutureGraph):
    _m = torch.tensor(
        dataset.sub_basin_datasets.station_normalizer.m_[:, :, 0:1],
        dtype=torch.float32,
    )[None, ...].cuda()
    _s = torch.tensor(
        dataset.sub_basin_datasets.station_normalizer.s_[:, :, 0:1],
        dtype=torch.float32,
    )[None, ...].cuda()
    now_station_idx_vec = dataset.sub_basin_datasets.get_used_station_offset_vec()
    model.eval()

    cache = np.linspace(-0.499, dist.get_world_size() - 1 + 0.499, len(dataset))
    cache = cache.round().astype(dtype=np.int32)

    data_ind_belong_to_now_rank = []
    for data_ind in range(len(dataset)):
        if cache[data_ind] == dist.get_rank():
            data_ind_belong_to_now_rank.append(data_ind)

    sub_dataset = torch.utils.data.Subset(dataset=dataset, indices=data_ind_belong_to_now_rank)

    loader = torch.utils.data.DataLoader(
        sub_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=EXP_CONFIG.num_workers
    )

    now_rank_dict = collections.defaultdict(dict)
    max_forecast_step_num = -1
    for batch_data in tqdm(loader, position=dist.get_rank(), desc=f"Rank[{dist.get_rank()}]-Epoch-{epoch_idx}-Eval"):

        inputs = {key: value.cuda() for key, value in batch_data.items() if key not in ["time_step"]}
     
        targets_future = inputs.pop("future_station_stream_flow")  # [1, b, sn, f]
        targets_future = targets_future[0]
        targets_0day = inputs["station_stream_flow"][..., -2:-1]  # [1, b, sn, 1]
        targets_0day = targets_0day[0]
        
        inputs["max_forecast_step_num"] = targets_future.shape[2]

        max_forecast_step_num = targets_future.shape[2]
        inputs["dataset_ind"] = torch.tensor(EVAL_DATASET_BELONG_TO_WHICH_GRAPH, dtype=torch.long)

        outputs_0day, outputs_future = model(**inputs)
        outputs = torch.cat([outputs_0day, outputs_future], dim=-1)
        targets = torch.cat([targets_0day, targets_future], dim=-1)
        max_forecast_step_num += 1

        for ind in range(outputs.shape[0]):
            target_time_step_ind = batch_data["time_step"][ind].item()
            for station_idx in now_station_idx_vec:
                now_rank_dict[station_idx][target_time_step_ind] = [outputs[ind, station_idx], targets[ind, station_idx]]  # forecast_step_num tensor

        inputs, targets, outputs = None, None, None
        gc.collect()
        torch.cuda.empty_cache()

    # write eval info to cache dir
    cache_dir = os.path.join(EXP_CONFIG.save_dir, "cache")
    if dist.get_rank() == 0:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)

    dist.barrier()
    torch.save(now_rank_dict, os.path.join(cache_dir, f"rank_{dist.get_rank():0>4}.pt"))
    dist.barrier()
    # read from other rank
    for other_rank in range(dist.get_world_size()):
        if other_rank != dist.get_rank():
            other_rank_dict = torch.load(os.path.join(cache_dir, f"rank_{other_rank:0>4}.pt"), map_location='cpu', weights_only=False)
            for station_idx in other_rank_dict.keys():
                now_rank_dict[station_idx].update(other_rank_dict[station_idx])

    total_pred = torch.empty(size=(dataset.sub_basin_datasets.total_time_step, len(now_station_idx_vec), max_forecast_step_num, 1), dtype=torch.float32).cuda()
    total_gt = torch.empty(size=(dataset.sub_basin_datasets.total_time_step, len(now_station_idx_vec), max_forecast_step_num, 1), dtype=torch.float32).cuda()
    status_matrix = torch.ones(size=(dataset.sub_basin_datasets.total_time_step, len(now_station_idx_vec)), dtype=torch.float32).cuda()

    for j, station_idx in enumerate(now_station_idx_vec):
        time_step_ind_vec = list(now_rank_dict[station_idx].keys())
        time_step_ind_vec = sorted(time_step_ind_vec)

        assert time_step_ind_vec[-1] - time_step_ind_vec[0] + 1 == len(time_step_ind_vec) and len(time_step_ind_vec) == dataset.sub_basin_datasets.total_time_step
        for i, target_time_step_ind in enumerate(time_step_ind_vec):
            total_pred[i, j, :, 0] = now_rank_dict[station_idx][target_time_step_ind][0].cuda()
            total_gt[i, j, :, 0] = now_rank_dict[station_idx][target_time_step_ind][1].cuda()
            status_matrix[i, j] = 0.0

    assert status_matrix.sum() == 0.0

    total_pred = torch.clamp(total_pred * _s[:, now_station_idx_vec] + _m[:, now_station_idx_vec], min=0.0)  # b, sn, f, 1
    total_gt = torch.clamp(total_gt * _s[:, now_station_idx_vec] + _m[:, now_station_idx_vec], min=0.0)  # b, sn, f, 1

    pred_np = total_pred.cpu().numpy()
    gt_np = total_gt.cpu().numpy()

    return pred_np, gt_np


@torch.no_grad()
def evaluation(epoch_idx, engine: deepspeed.DeepSpeedEngine, dataset: MixDatasetForFutureGraph):
    pred_np, gt_np = compute_predict_results(epoch_idx=epoch_idx, model=engine, dataset=dataset)
    vec = []
    data_dict = collections.defaultdict(list)
    max_eval_forecast_steps = 1
    # max_eval_forecast_steps = pred_np.shape[2]
    for forecast_step_ind in range(pred_np.shape[2]-1, pred_np.shape[2]):
        res = compute_predict_metric(
            epoch_idx=epoch_idx,
            dataset=dataset,
            pred_np=pred_np[:, :, forecast_step_ind: forecast_step_ind + 1, :],
            gt_np=gt_np[:, :, forecast_step_ind: forecast_step_ind + 1, :],
            now_forecast_step_num=forecast_step_ind
        )
        vec.append(res)
        data_dict["forecast_days"].append(forecast_step_ind)
        for key in ["good_station_num", "good_station_nse", "KGE"]:
            if key in res.keys():
                data_dict[key].append(res[key])

    df = pd.DataFrame(data_dict)
    if dist.get_rank() == 0:
        eval_results_save_to = os.path.join(EXP_CONFIG.save_dir, f"eval_results/epoch_{epoch_idx:0>3}.csv")
        os.makedirs(os.path.dirname(eval_results_save_to), exist_ok=True)
        df.to_csv(eval_results_save_to, na_rep='nan', index=False)
        print(f"\n>>> eval results save to: {eval_results_save_to} success !!!", flush=True)
    return vec


def train(engine, train_dataloader, train_sample, val_dataset):

    time_counter = TimeCount(max_cnt=EXP_CONFIG.now_rank_total_batch_num)
    global_iteration = 0
    best_epoch = 0
    all_val_results = []
    for epoch_idx in range(EXP_CONFIG.max_epoch):
        train_sample.set_epoch(epoch_idx)

        for batch_idx, batch_data in enumerate(train_dataloader):
            engine.train()
            global_iteration += 1

            inputs = {key: value.cuda() for key, value in batch_data.items() if key not in ["time_step"]}

            targets_future = inputs.pop("future_station_stream_flow")  # [1, b, sn, f]
            targets_future = targets_future[0]
            targets_0day = inputs["station_stream_flow"][..., -2:-1]  # [1, b, sn, 1]
            targets_0day = targets_0day[0]

            inputs["max_forecast_step_num"] = targets_future.shape[2]
            outputs_0day, outputs_future = engine(**inputs)

            loss = torch.nn.functional.mse_loss(outputs_0day, targets_0day) + torch.nn.functional.mse_loss(outputs_future, targets_future)
            engine.backward(loss)
            engine.step()

            time_counter.update()
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            if global_iteration % EXP_CONFIG.log_frequency == 0 and dist.get_rank() == 0:
                current_lr = engine.optimizer.param_groups[0]['lr']
                print_info = f">>> Epoch: {epoch_idx}/{EXP_CONFIG.max_epoch}, "
                print_info += f"Local batch: {(batch_idx + 1)}/{len(train_dataloader)}({(batch_idx + 1) / len(train_dataloader):.2%}), "
                print_info += f"Global batch: {global_iteration}/{EXP_CONFIG.now_rank_total_batch_num}({global_iteration / EXP_CONFIG.now_rank_total_batch_num:.2%}), "
                print_info += f"Current lr: {current_lr}, Total loss: {loss.item()}, "
                print_info += f"Train speed(per batch): {time_counter.speed()}, Still need time: {time_counter.still_need()}"
                print(print_info, flush=True)

        if epoch_idx >= args.eval_start_epoch and (epoch_idx + 1) % args.eval_frequency == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val_results = evaluation(epoch_idx, engine, dataset=val_dataset)[0]
                all_val_results.append(val_results)
                if all_val_results[best_epoch] is not None and val_results[args.chose_best_epoch_metric_name] >= all_val_results[best_epoch][args.chose_best_epoch_metric_name]:
                    best_epoch = epoch_idx
                if all_val_results[best_epoch] is None:
                    best_epoch = epoch_idx
                if dist.get_rank() == 0:
                    print(f"\n>>> best[{args.chose_best_epoch_metric_name}] val results: \n", flush=True)
                    print(all_val_results[best_epoch]["print_info"])

            dist.barrier()
        else:
            all_val_results.append(None)

        save_checkpoint(epoch=epoch_idx, global_iteration=global_iteration, engine=engine, now_epoch_is_best=best_epoch == epoch_idx)


def main():
    import socket
    import datetime
    hostname = socket.gethostname()
    deepspeed.init_distributed(dist_backend='nccl', timeout=datetime.timedelta(seconds=1800))
    os.environ["TRITON_CACHE_DIR"] = f"/mnt/inaisfs/data/home/yangyh_criait/.triton/cache_{hostname}_{dist.get_rank():0>5}"
    local_rank = os.environ["LOCAL_RANK"]
    device = torch.device(f"cuda:{local_rank}")

    set_seed_and_device(EXP_CONFIG.global_seed, device)

    """build data"""
    dataset_vec = []
    for data_config in TRAIN_DATASET_CONF_DICT_VEC:
        dataset_vec.append(
            MixDatasetForFutureGraph(
                sub_basin_datasets=build_dataset(dataset_config_dict=data_config, sub_all='sub', is_train=True),
                all_basin_datasets=build_dataset(dataset_config_dict=data_config, sub_all='all', is_train=True),
                seed=EXP_CONFIG.data_seed,
                batch_size=EXP_CONFIG.batch_size
            )
        )

    train_dataset = BlendDynamicDataset(dataset_vec=dataset_vec)

    train_sample = DistributedSampler(
        dataset=train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=EXP_CONFIG.data_seed

    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=EXP_CONFIG.num_workers,
        sampler=train_sample
    )
    cache_train_dataset = MixDatasetForFutureGraph(
        sub_basin_datasets=build_dataset(dataset_config_dict=EVAL_DATASET_CONF_DICT, sub_all='sub', is_train=True),
        all_basin_datasets=build_dataset(dataset_config_dict=EVAL_DATASET_CONF_DICT, sub_all='all', is_train=True),
        seed=EXP_CONFIG.data_seed,
        batch_size=EXP_CONFIG.batch_size
    )
    val_dataset = MixDatasetForFutureGraph(
        sub_basin_datasets=build_dataset(dataset_config_dict=EVAL_DATASET_CONF_DICT, sub_all='sub', is_train=False, station_normalizer=cache_train_dataset.sub_basin_datasets.station_normalizer),
        all_basin_datasets=build_dataset(dataset_config_dict=EVAL_DATASET_CONF_DICT, sub_all='all', is_train=False, station_normalizer=cache_train_dataset.all_basin_datasets.station_normalizer),
        seed=EXP_CONFIG.data_seed,
        batch_size=1
    )
    EXP_CONFIG.update({
        "now_rank_total_batch_num": EXP_CONFIG.max_epoch * len(train_dataloader)
    })
    """build model and optimizer"""

    model = STREAM(
        time_step=EXP_CONFIG.history_time_step_num + EXP_CONFIG.future_time_step_num,
        sub_basin_weather_dim=val_dataset.sub_basin_datasets.station_feature_dim - 1,
        sub_basin_statics_dim=val_dataset.sub_basin_datasets.statics_data.shape[-1],
        hidden_dim=EXP_CONFIG.num_hidden,
        edge_features_dim=val_dataset.sub_basin_datasets.edge_features.shape[-1],
        edge_features_used_time_step=args.edge_features_used_time_step,
        edge_features_time_embed_dim=32,
        message_passing_num=20,
        use_checkpointing=EXP_CONFIG.use_checkpointing,
        station_num_vec=[item.sub_basin_datasets.statics_data.shape[0] for item in dataset_vec]
    ).cuda()

    load_checkpoint(model, reload_checkpoint_from=EXP_CONFIG.reload_checkpoint_from, device=device)

    total_learnable_params = 0
    for param_name, param_value in model.named_parameters():
        if param_value.requires_grad and "balance_layers" not in param_name:
            total_learnable_params += param_value.numel()
    if dist.get_rank() == 0:
        print("*" * 50 + "\n" + EXP_CONFIG.get_print_params().strip() + "\n" + "*" * 50, flush=True)
        EXP_CONFIG.check_params()
        print(f"\n>>> total_learnable_params: {total_learnable_params / 1e+6}M")
    # Initialize optimizer and scheduler
    config = {
        "train_micro_batch_size_per_gpu": 1,  # batch size will be implemented in dataset
        "gradient_accumulation_steps": 1,  # no gradient accumulation
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": EXP_CONFIG.lr,
                "weight_decay": EXP_CONFIG.weight_decay,
                "betas": [0.9, 0.95],
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": EXP_CONFIG.lr * 0.1,
                "warmup_max_lr": EXP_CONFIG.lr,
                "warmup_num_steps": 100,
                "total_num_steps": (len(train_dataset) // dist.get_world_size()) * EXP_CONFIG.max_epoch  # no gradient accumulation, sp, tp
            }
        },
        "gradient_clipping": EXP_CONFIG.gradient_clip,
        "zero_optimization": True,
        "steps_per_print": EXP_CONFIG.log_frequency,
        "flops_profiler": {
            "enabled": False,
            "profile_step": EXP_CONFIG.log_frequency,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
        }
    }

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=config,
    )

    if EXP_CONFIG.just_evaluation:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluation(-1, engine, dataset=val_dataset)
    else:
        train(engine, train_dataloader, train_sample, val_dataset)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    parser = build_default_args_parser()
    parser.add_argument('--train_dataset_yaml_file_vec', type=str)
    parser.add_argument('--val_dataset_yaml_file', type=str)
    parser.add_argument('--eval_start_epoch', type=int, default=0)
    parser.add_argument('--eval_frequency', type=int, default=1)
    parser.add_argument('--edge_features_used_time_step', type=int)
    parser.add_argument('--chose_best_epoch_metric_name', type=str, default='good_station_nse')

    args = parser.parse_args()

    TRAIN_DATASET_CONF_DICT_VEC = []
    EVAL_DATASET_BELONG_TO_WHICH_GRAPH = None
    if args.train_dataset_yaml_file_vec.endswith('.txt'):
        with open(args.train_dataset_yaml_file_vec, mode='r', encoding='utf-8') as f_read:
            train_dataset_yaml_file_vec = f_read.readlines()
            train_dataset_yaml_file_vec = [item.strip() for item in train_dataset_yaml_file_vec if item.strip() != '']
    else:
        train_dataset_yaml_file_vec = [args.train_dataset_yaml_file_vec]

    if args.val_dataset_yaml_file.endswith('.txt'):
        with open(args.val_dataset_yaml_file, mode='r', encoding='utf-8') as f_read:
            val_dataset_yaml_file_vec = f_read.readlines()
            val_dataset_yaml_file_vec = [item.strip() for item in val_dataset_yaml_file_vec if item.strip() != '']
            assert len(val_dataset_yaml_file_vec) == 1
    else:
        val_dataset_yaml_file_vec = [args.val_dataset_yaml_file]

    for ind, dataset_yaml_file in enumerate(train_dataset_yaml_file_vec):
        if dataset_yaml_file == val_dataset_yaml_file_vec[0]:
            EVAL_DATASET_BELONG_TO_WHICH_GRAPH = ind
        with open(dataset_yaml_file, mode='r') as f:
            TRAIN_DATASET_CONF_DICT_VEC.append(yaml.safe_load(f))

    if RANK == 0:
        print(f"\n>>> EVAL_DATASET_BELONG_TO_WHICH_GRAPH: {EVAL_DATASET_BELONG_TO_WHICH_GRAPH}", flush=True)
    assert EVAL_DATASET_BELONG_TO_WHICH_GRAPH is not None
    
    with open(val_dataset_yaml_file_vec[0], mode='r') as f:
        EVAL_DATASET_CONF_DICT = yaml.safe_load(f)

    EXP_CONFIG = build_config_from_args(args)
    EXP_CONFIG.update(EVAL_DATASET_CONF_DICT)
    main()
