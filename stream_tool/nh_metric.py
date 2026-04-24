import neuralhydrology.evaluation.metrics as nh_eval_metric
import numpy as np
from neuralhydrology.datautils.utils import DataArray
import collections
import pandas as pd


PEAKING_DISTANCE = 10
MAX_WINDOW = 3

def acc_k(
        obs: np.ndarray, sim: np.ndarray,
        threshold: float
):
    abs_delta = np.abs(obs - sim)
    info = (abs_delta / (obs + 1e-8)) <= threshold
    acc = info.astype(dtype=np.float32).mean()
    return acc


def get_peak_level(gt, level_num, rate=None):
    peak_num = len(gt)
    index = np.arange(0, peak_num, step=1).tolist()
    v = list(zip(gt, index))
    v = sorted(v, key=lambda s: s[0])
    gt, index = zip(*v)

    if rate is None:
        rate = [1] * level_num

    assert len(rate) == level_num
    rate = np.array(rate, dtype=np.float64)
    rate = rate / rate.sum()  # sum will be 1
    rate = np.cumsum(rate)

    cache = np.linspace(0, 1, num=peak_num)
    level = []
    for item in cache:
        flag = False
        for l, r in enumerate(rate):
            if item <= r:
                flag = True
                level.append(l + 1)
                break
        if not flag:
            level.append(level_num)

    assert len(level) == peak_num

    v = list(zip(gt, index, level))
    v = sorted(v, key=lambda s: s[1])
    gt, index, level = zip(*v)
    return level


def get_peak_idx_vec(
        obs: DataArray,
        sim: DataArray,
):
    # verify inputs
    nh_eval_metric._validate_inputs(obs, sim)

    # get time series with only valid observations (scipy's find_peaks doesn't guarantee correctness with NaNs)
    obs, sim = nh_eval_metric._mask_valid(obs, sim)

    # heuristic to get indices of peaks and their corresponding height.
    obs_peaks, _ = nh_eval_metric.signal.find_peaks(obs.values, distance=PEAKING_DISTANCE, prominence=np.std(obs.values))
    sim_peaks, _ = nh_eval_metric.signal.find_peaks(sim.values, distance=PEAKING_DISTANCE, prominence=np.std(sim.values))
    return obs_peaks, sim_peaks


def refine_mean_peak_timing(
    obs: DataArray,
    sim: DataArray,
    window: int = None,
    resolution: str = '1D',
    datetime_coord: str = None
) -> tuple:

    # verify inputs
    nh_eval_metric._validate_inputs(obs, sim)

    # get time series with only valid observations (scipy's find_peaks doesn't guarantee correctness with NaNs)
    obs, sim = nh_eval_metric._mask_valid(obs, sim)

    # heuristic to get indices of peaks and their corresponding height.
    peaks, _ = nh_eval_metric.signal.find_peaks(obs.values, distance=PEAKING_DISTANCE, prominence=np.std(obs.values))

    # infer name of datetime index
    if datetime_coord is None:
        datetime_coord = nh_eval_metric.utils.infer_datetime_coord(obs)

    if window is None:
        # infer a reasonable window size
        window = max(int(nh_eval_metric.utils.get_frequency_factor('12H', resolution)), MAX_WINDOW)

    # evaluate timing
    timing_errors = []
    peak_errors = []
    sim_values = []
    obs_values = []
    for idx in peaks:
        # skip peaks at the start and end of the sequence and peaks around missing observations
        # (NaNs that were removed in obs & sim would result in windows that span too much time).
        if (idx - window < 0) or (idx + window >= len(obs)) or (pd.date_range(obs[idx - window][datetime_coord].values,
                                                                              obs[idx + window][datetime_coord].values,
                                                                              freq=resolution).size != 2 * window + 1):
            continue

        # check if the value at idx is a peak (both neighbors must be smaller)
        if (sim[idx] > sim[idx - 1]) and (sim[idx] > sim[idx + 1]):
            peak_sim = sim[idx]
        else:
            # define peak around idx as the max value inside of the window
            values = sim[idx - window:idx + window + 1]
            peak_sim = values[values.argmax()]

        # get xarray object of qobs peak, for getting the date and calculating the datetime offset
        peak_obs = obs[idx]

        # calculate the time difference between the peaks
        delta = peak_sim.coords[datetime_coord] - peak_obs.coords[datetime_coord]

        timing_error = delta.values / pd.to_timedelta(resolution)
        timing_errors.append(timing_error)

        peak_error = np.abs(peak_obs.data - peak_sim.data) / (peak_obs.data + 1e-8)
        peak_errors.append(peak_error)
        sim_values.append(peak_sim.data)
        obs_values.append(peak_obs.data)

    return timing_errors, peak_errors, obs_values, sim_values


def calculate_acc_metrics(pred, gt, metric_name_list: list):
    record = collections.defaultdict(np.ndarray)
    time_step, sn, _, _ = pred.shape

    for metric_name in metric_name_list:
        if metric_name.startswith("Acc-"):
            res = np.zeros(shape=(sn,))
            threshold = float(metric_name[4:])
            for sn_ind in range(sn):
                res[sn_ind] = acc_k(
                    np.reshape(gt[:, sn_ind], newshape=(time_step,)),
                    np.reshape(pred[:, sn_ind], newshape=(time_step,)),
                    threshold=threshold
                )
            record[metric_name] = res
        else:
            raise RuntimeError(f">>> wrong metric_name: {metric_name}")

    return record


def calculate_official_metrics(pred, gt, metric_name_list: list, resolution: str = "1d"):

    time_step, sn, _, _ = pred.shape

    record = collections.defaultdict(np.ndarray)

    nh_metric_name_list = []

    for metric_name in metric_name_list:
        if metric_name in nh_eval_metric.get_available_metrics():
            res = np.zeros(shape=(sn,))
            record[metric_name] = res
            nh_metric_name_list.append(metric_name)
        else:
            raise RuntimeError(f">>> wrong metric_name: {metric_name}")

    time_range = pd.date_range("2000-01-01", periods=time_step, freq=resolution)
    for sn_ind in range(pred.shape[1]):
        obs = DataArray(
            data=np.reshape(gt[:, sn_ind], newshape=(time_step,)),
            coords={"datetime": time_range},
            dims=["datetime"],
        )
        sim = DataArray(
            data=np.reshape(pred[:, sn_ind], newshape=(time_step,)),
            coords={"datetime": time_range},
            dims=["datetime"],
        )

        now_station_metric = nh_eval_metric.calculate_metrics(obs, sim, metrics=nh_metric_name_list)
        for metric_name, metric_value in now_station_metric.items():
            record[metric_name][sn_ind] = metric_value
    return record


def calculate_refine_mean_peak_timing_metrics(pred, gt, time_range = None, metric_name: str = "Refine-Peak-Timing", window: int = None, resolution: str = '1D', datetime_coord: str = None):
    assert metric_name == "Refine-Peak-Timing"
    time_step, sn, _, _ = pred.shape
    if time_range is None:
        time_range = pd.date_range("2000-01-01", periods=time_step, freq=resolution)
    else:
        assert len(time_range) == time_step
    res = []
    for sn_ind in range(pred.shape[1]):
        obs = DataArray(
            data=np.reshape(gt[:, sn_ind], newshape=(time_step,)),
            coords={"datetime": time_range},
            dims=["datetime"],
        )
        sim = DataArray(
            data=np.reshape(pred[:, sn_ind], newshape=(time_step,)),
            coords={"datetime": time_range},
            dims=["datetime"],
        )
        cache = refine_mean_peak_timing(obs, sim, window=window, resolution=resolution, datetime_coord=datetime_coord)

        res.append(cache)

    return {metric_name: res}


def call_all_metric(pred, gt, metric_name_list: list, freq: str = "1d", time_range=None):
    # print(f">>> pred: {pred.shape}, gt: {gt.shape}")
    assert len(pred.shape) == len(gt.shape) and len(pred.shape) == 4
    assert pred.shape[2] == 1 and pred.shape[3] == 1
    res = {}
    official_metric_name_vec = []
    acc_metric_name_vec = []
    for metric_name in metric_name_list:

        if metric_name in nh_eval_metric.get_available_metrics():
            official_metric_name_vec.append(metric_name)
        elif metric_name.startswith("Acc-"):
            acc_metric_name_vec.append(metric_name)
        elif metric_name == "Refine-Peak-Timing":
            res.update(calculate_refine_mean_peak_timing_metrics(pred, gt, resolution=freq, time_range=time_range))
        else:
            raise RuntimeError(f">>> wrong metric_name: {metric_name}")
    if len(official_metric_name_vec) > 0:
        res.update(calculate_official_metrics(pred, gt, metric_name_list=official_metric_name_vec, resolution=freq))

    if len(acc_metric_name_vec) > 0:
        res.update(calculate_acc_metrics(pred, gt, metric_name_list=acc_metric_name_vec))
    return res


if __name__ == '__main__':
    pass
