# Functions for extracting and averaging raw data
from __future__ import annotations

import json
import os

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter

import UnitMatchPy.utils as util


# Decompressed data functions
def read_meta(meta_path):
    """
    Reads in the meta data as a dictionary

    Parameters
    ----------
    meta_path : str
        The absolute path to the meta data dictionary

    Returns
    -------
    dict
        The meta data dictionary
    """
    meta_dict = {}
    with meta_path.open() as f:
        meta_list = f.read().splitlines()
        # convert the list entries into key value pairs
        for m in meta_list:
            cs_list = m.split(sep="=")
            if cs_list[0][0] == "~":
                curr_key = cs_list[0][1 : len(cs_list[0])]
            else:
                curr_key = cs_list[0]
            meta_dict.update({curr_key: cs_list[1]})

    return meta_dict


def get_sample_idx(spike_times, unit_ids, sample_amount, units):
    """
    Uses data from KiloSort to choose a even subset of spikes for each unit

    Parameters
    ----------
    spike_times : ndarray (n_spikes)
        The times (in samples) for each spike
    unit_ids : ndarays
        The id's for each unit
    sample_amount : int
        The number of spikes sampled for each amount
    units : ndarray
        The ids for each unit

    Returns
    -------
    sample_idx
        The idxs of the spikes to be sampled for each unit
    """
    unique_unit_ids = np.unique(unit_ids)
    nunits_all = len(unique_unit_ids)

    sample_idx = np.zeros((nunits_all, sample_amount))
    # Process ALL unit
    for i, idx in enumerate(units):
        unit_times = spike_times[unit_ids == idx]
        if sample_amount < len(unit_times):
            chosen_idxs = np.linspace(
                0, len(unit_times) - 1, sample_amount, dtype=int
            )  # -1 so can't index out of region
            sample_idx[i, :] = unit_times[chosen_idxs]
        else:
            sample_idx[i, : len(unit_times)] = unit_times
            sample_idx[i, len(unit_times) :] = np.nan

    return sample_idx


def extract_a_unit(
    sample_idx, data, half_width, spike_width, n_channels, sample_amount
):
    """
    Extract an average waveform for a single unit.

    Parameters
    ----------
    sample_idx : ndarray (sample_amount)
        The spike index's to be sampled for this unit
    data : memmap
        The memmap array of raw data
    half_width : int
        The half width value for this extraction
    spike_width : int
        The width of each unit in samples
    n_channels : int
        The number of channels to extract (to exclude sync channels)
    sample_amount : int
        The number of spike to extract for each unit

    Returns
    -------
    ndarray (spike_width, n_channels, 2)
        Two average waveforms for each unit
    """

    channels = np.arange(0, n_channels)

    all_sample_waveforms = np.zeros((sample_amount, spike_width, n_channels))
    for i, idx in enumerate(sample_idx[:]):
        if np.isnan(idx):
            continue
        tmp = data[
            int(idx - half_width - 1) : int(idx + half_width - 1), channels
        ]  # -1, to better fit with ML
        tmp.astype(np.float32)
        # gaussian smooth, over time gaussian window = 5, sigma = window size / 5
        tmp = gaussian_filter(
            tmp, 1, radius=2, axes=0
        )  # edges are handled differently to ML
        # window ~ radius *2 + 1
        tmp = tmp - np.mean(tmp[:20, :], axis=0)
        all_sample_waveforms[i] = tmp

    # median and split CV's
    n_waves = np.sum(~np.isnan(sample_idx[:]))
    cv_limit = np.floor(n_waves / 2).astype(int)

    # find median over samples
    avg_waveforms = np.zeros((spike_width, n_channels, 2))
    avg_waveforms[:, :, 0] = np.median(
        all_sample_waveforms[:cv_limit, :, :], axis=0
    )  # median over samples
    avg_waveforms[:, :, 1] = np.median(
        all_sample_waveforms[cv_limit:n_waves, :, :], axis=0
    )  # median over samples
    return avg_waveforms


def extract_a_unit_KS4(
    sample_idx,
    data,
    samples_before,
    samples_after,
    spike_width,
    n_channels,
    sample_amount,
):
    """
    Extract a single unit's average waveform from KS4 data

    Parameters
    ----------
    sample_idx : ndarray (sample_amount)
        The spike index's to be sampled for this unit
    data : memmap
        The memmap array of raw data
    samples_before : int
        The number of samples before the spike to sample
    samples_after : int
        The number of samples after the spike to sample
    spike_width : int
        The width of each unit in samples
    n_channels : int
        The number of channels to extract (to exclude sync channels)
    sample_amount : int
        The number of spikes to extract for each unit

    Returns
    -------
    ndarray (spike_width, n_channels, 2)
        Two average waveforms for each unit
    """
    channels = np.arange(0, n_channels)

    all_sample_waveforms = np.zeros((sample_amount, spike_width, n_channels))
    for i, idx in enumerate(sample_idx[:]):
        if np.isnan(idx):
            continue

        start_idx = int(idx - samples_before - 1)
        end_idx = int(idx + samples_after - 1)

        # Extract the data segment
        tmp = data[max(0, start_idx) : min(data.shape[0], end_idx), channels]
        tmp = tmp.astype(np.float32)

        # Pad the data if necessary
        if start_idx < 0:
            pad_width = (-start_idx, 0)
            tmp = np.pad(
                tmp, ((pad_width[0], 0), (0, 0)), mode="constant", constant_values=0
            )
        if end_idx > data.shape[0]:
            pad_width = (0, end_idx - data.shape[0])
            tmp = np.pad(
                tmp, ((0, pad_width[1]), (0, 0)), mode="constant", constant_values=0
            )

        # Gaussian smooth, over time gaussian window = 5, sigma = window size / 5
        tmp = gaussian_filter(
            tmp, 1, radius=2, axes=0
        )  # edges are handled differently to ML
        # window ~ radius * 2 + 1
        tmp = tmp - np.mean(tmp[:samples_before, :], axis=0)

        all_sample_waveforms[i] = tmp

    # Median and split CVs
    n_waves = np.sum(~np.isnan(sample_idx[:]))
    cv_lim = np.floor(n_waves / 2).astype(int)

    # Find median over samples
    avg_waveforms = np.zeros((spike_width, n_channels, 2))
    avg_waveforms[:, :, 0] = np.median(
        all_sample_waveforms[:cv_lim, :, :], axis=0
    )  # median over samples
    avg_waveforms[:, :, 1] = np.median(
        all_sample_waveforms[cv_lim:n_waves, :, :], axis=0
    )  # median over samples
    return avg_waveforms


def save_avg_waveforms(
    avg_waveforms, save_dir, all_unit_ids, good_units, extract_good_units_only=False
):
    """
    Saves the average waveforms as a unique .npy file called "UnitX_RawSpikes.npy" in a folder called
    RawWaveforms in the save_dir.

    Parameters
    ----------
    avg_waveforms : ndarray (n_units, spike_width, n_channels, 2)
        The extracts waveforms for all units
    save_dir : str
        The absolute path to the directory where the results are to be saved, recommend the KS results directory
    good_units : ndarray
        A list of the good units in the session
    extract_good_units_only : bool, optional
        If True will only save the good units, by default False
    """
    current_dir = os.getcwd()
    os.chdir(save_dir)
    dir_list = os.listdir()
    if "RawWaveforms" in dir_list:
        tmp_path = os.path.join(save_dir, "RawWaveforms")
    else:
        os.mkdir("RawWaveforms")
        tmp_path = os.path.join(save_dir, "RawWaveforms")

    os.chdir(tmp_path)

    # ALL waveforms from 0->nUnits
    if extract_good_units_only == False:
        for i, idx in enumerate(all_unit_ids):
            np.save(f"Unit{idx}_RawSpikes.npy", avg_waveforms[i, :, :, :])
        print(
            f"Saved {avg_waveforms.shape[0]} units to RawWaveforms directory, saving all units"
        )

    # If only extracting GoodUnits
    else:
        for i, idx in enumerate(good_units):
            # need idx, to select value so saves with correct name
            np.save(f"Unit{idx}_RawSpikes.npy", avg_waveforms[i, :, :, :])
        print(
            f"Saved {good_units.shape[0]} units to RawWaveforms directory, only saving good units"
        )
    os.chdir(current_dir)


# Load in necessary files from KS directory and raw data directory
# extracting n Sessions
def get_raw_data_paths(raw_data_dir_paths):
    """
    This function will look in the raw data directory to find the necessary files

    Parameters
    ----------
    raw_data_dir_paths : list
        Each value is the path to the raw data for the session

    Returns
    -------
    lists
        lists where the values are the paths to the necessary files for each case
    """
    cbin_paths = []
    ch_paths = []
    meta_paths = []

    for i in range(len(raw_data_dir_paths)):
        for f in os.listdir(raw_data_dir_paths[i]):
            name, ext = os.path.splitext(f)

            if ext == ".cbin":
                cbin_paths.append(os.path.join(raw_data_dir_paths[i], name + ext))

            if ext == ".ch":
                ch_paths.append(os.path.join(raw_data_dir_paths[i], name + ext))

            if ext == ".meta":
                meta_paths.append(os.path.join(raw_data_dir_paths[i], name + ext))

    return cbin_paths, ch_paths, meta_paths


def extract_KS_data(KS_dirs, extract_good_units_only=False):
    """
    This function will look in each KS directory to find the needed files

    Parameters
    ----------
    KS_dirs : list
        each value is the path to a KS directory for each session
    extract_good_units_only : bool, optional
        If True will extract good units only, by default False

    Returns
    -------
    lists
        The lists of path to the files for each session
    """
    n_sessions = len(KS_dirs)

    # Load Spike Times
    spike_times = []
    for i in range(n_sessions):
        path_tmp = os.path.join(KS_dirs[i], "spike_times.npy")
        spike_times_tmp = np.load(path_tmp)
        spike_times.append(spike_times_tmp)

    # Load Spike ID's
    spike_ids = []
    all_unit_ids = []
    for i in range(n_sessions):
        path_tmp = os.path.join(KS_dirs[i], "spike_clusters.npy")
        spike_ids_tmp = np.load(path_tmp)
        spike_ids.append(spike_ids_tmp)
        all_unit_ids.append(np.unique(spike_ids_tmp))

    if extract_good_units_only:
        # Good unit ID's
        unit_labels_paths = []

        # load Good unit Paths
        for i in range(n_sessions):
            unit_labels_paths.append(os.path.join(KS_dirs[i], "cluster_group.tsv"))

        good_units = util.get_good_units(unit_labels_paths)

        return spike_ids, spike_times, good_units, all_unit_ids
    else:
        return spike_ids, spike_times, [None for s in range(n_sessions)], all_unit_ids


def find_last_valid_sample(path, n_channels_tot, block=3000):
    """Return the number of leading samples in `path` that contain real data.

    Returns n_samples when the file is intact. Raises if the zero region is
    not a single contiguous tail (that would mean a different problem).
    """
    n_samples = os.path.getsize(path) // 2 // n_channels_tot
    d = np.memmap(path, dtype="int16", mode="r", shape=(n_samples, n_channels_tot))
    try:

        # `d` comes from the enclosing scope -- do NOT take it as a parameter.
        # It was briefly a parameter while only two of the four call sites were
        # updated, which raised TypeError on the binary-search path below. That
        # path only runs for a TRUNCATED file, so intact files returned early and
        # the break stayed hidden until it mattered.
        def is_zero(i):
            i = min(max(int(i), 0), n_samples - block)
            return not np.any(d[i : i + block])

        if not is_zero(n_samples - block):
            return n_samples  # ends with real data -> intact
        if is_zero(0):
            raise ValueError(f"{path} looks empty at the start")

        # binary search for the start of the zero tail
        lo, hi = 0, n_samples - block
        while hi - lo > block:
            mid = (lo + hi) // 2
            if is_zero(mid):
                hi = mid
            else:
                lo = mid

        seg = np.asarray(d[lo : min(lo + 2 * block, n_samples)])
        nz = np.flatnonzero(np.any(seg != 0, axis=1))
        last = lo + int(nz[-1]) if nz.size else lo

        # confirm the remainder really is one contiguous run of zeros
        for probe in np.linspace(last + block, n_samples - block, 50):
            if probe > last and not is_zero(probe):
                raise ValueError(
                    f"{path}: found real data after sample {last}; the zero "
                    "region is not a simple truncated tail - inspect manually."
                )
        return last + 1
    finally:
        del d


def extraction_pipeline(
    data_paths: list,
    meta_paths: list,
    ks_paths: list,
    n_sessions: int,
    spike_ids: list,
    all_unit_ids: list,
    spike_times: list,
    sample_amount: int,
    samples_before: int,
    samples_after: int,
    half_width: int,
    spike_width: int,
    max_width: int,
    n_channels: int,
    KS4_data: bool = True,
    extract_good_units_only: bool = False,
    good_units: list | None = None,
):

    # first loop: ensure good binary files and only use readable portion
    valid_until = []
    for sid in range(n_sessions):
        with open(meta_paths[sid], "r") as file:
            meta = json.load(file)
        n_ch_tot = int(meta["continuous"][0]["num_channels"])
        n_samples = os.path.getsize(data_paths[sid]) // 2 // n_ch_tot
        v = find_last_valid_sample(data_paths[sid], n_ch_tot)
        valid_until.append(v)
        pct = 100 * v / n_samples
        flag = "OK" if v == n_samples else "TRUNCATED"
        print(
            f"session {sid}: {v:,} / {n_samples:,} samples valid "
            f"({pct:.2f}%, {v / 30000 / 3600:.2f} h of {n_samples / 30000 / 3600:.2f} h)  [{flag}]"
        )
        if v < n_samples:
            st_all = np.asarray(spike_times[sid]).squeeze()
            n_lost = int((st_all >= v).sum())
            print(
                f"           -> discarding {n_lost:,} of {st_all.size:,} spikes "
                f"({100 * n_lost / st_all.size:.1f}%) that fall in the missing region"
            )

    # second loop: extract spike waveforms
    for sid in range(n_sessions):
        # units to extract for this session
        if extract_good_units_only:
            units = np.asarray(good_units[sid]).squeeze()
        else:
            units = np.unique(spike_ids[sid])
        n_units = len(units)

        # load metadata
        with open(meta_paths[sid], "r") as file:
            meta = json.load(file)
        n_bytes = os.path.getsize(data_paths[sid])
        n_channels_tot = int(meta["continuous"][0]["num_channels"])
        n_samples = int(n_bytes / (2 * n_channels_tot))

        # create memmap to raw data, for that session.
        # mode="r" -- read only, so the raw recording can never be written to.
        data = np.memmap(
            data_paths[sid], dtype="int16", mode="r", shape=(n_samples, n_channels_tot)
        )

        # Keep only spikes that have a full waveform of REAL data behind them:
        #   - far enough from the start/end for a complete window, and
        #   - before valid_until[sid], so we never sample the zero-filled tail of
        #     a truncated file (see the previous cell).
        st = np.asarray(spike_times[sid]).squeeze()
        si = np.asarray(spike_ids[sid]).squeeze()
        upper = min(data.shape[0], valid_until[sid]) - max_width
        keep = (st >= max_width) & (st < upper)
        spike_times_tmp = st[keep]
        spike_ids_tmp = si[keep]
        print(
            f"session {sid}: using {int(keep.sum()):,} of {st.size:,} spikes "
            f"({100.0 * int(keep.sum()) / st.size:.1f}%), {n_units} units"
        )

        sample_idx = get_sample_idx(
            spike_times_tmp, spike_ids_tmp, sample_amount, units=units
        )

        if KS4_data:
            avg_waveforms = Parallel(
                n_jobs=-1, verbose=10, mmap_mode="r", max_nbytes=None
            )(
                delayed(extract_a_unit_KS4)(
                    sample_idx[uid],
                    data,
                    samples_before,
                    samples_after,
                    spike_width,
                    n_channels,
                    sample_amount,
                )
                for uid in range(n_units)
            )
        else:
            avg_waveforms = Parallel(
                n_jobs=-1, verbose=10, mmap_mode="r", max_nbytes=None
            )(
                delayed(extract_a_unit)(
                    sample_idx[uid],
                    data,
                    half_width,
                    spike_width,
                    n_channels,
                    sample_amount,
                )
                for uid in range(n_units)
            )
        avg_waveforms = np.asarray(avg_waveforms)

        # sanity check: no unit should come out flat on either cross-validation half
        for cv in (0, 1):
            peak = np.nanmax(np.abs(avg_waveforms[:, :, :, cv]), axis=(1, 2))
            flat = np.flatnonzero(peak == 0)
            if flat.size:
                print(
                    f"  WARNING session {sid} CV{cv}: {flat.size}/{n_units} units are all "
                    f"zero (first few: {units[flat[:5]]}) -- check the raw data."
                )

        # Save in file named 'RawWaveforms' in the KS Directory
        save_avg_waveforms(
            avg_waveforms,
            ks_paths[sid],
            all_unit_ids[sid],
            good_units=good_units[sid],
            extract_good_units_only=extract_good_units_only,
        )
        del data
