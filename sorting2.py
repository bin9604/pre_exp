# sorting_server.py
import os
import sys
import time
import logging
import traceback

# —— 服务器无 GUI ——
import matplotlib
matplotlib.use("Agg")

# —— 限制线程，避免多通道并行时抢满 CPU ——
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface as si
from spikeinterface import sorters as ss
from spikeinterface.curation import remove_excess_spikes

# ================= 配置区（只改这里） =================
PLX_FILE = "/fastdata/home/libin/data/subjects_2.exps_29.origin.20260928131241_565310.20260928__ce.00000.plx"
OUT_DIR  = "/fastdata/home/libin/data/sorting_out"
SORTER_FOLDER = os.path.join(OUT_DIR, "ms5_by_channel_filtered")

# QC 阈值
MIN_SPIKES = 20
MIN_FIRING_RATE = 0.1
MIN_PRESENCE_RATIO = 0.8
MIN_SNR = 4.0
MAX_ISI_VIOLATION_RATIO = 0.01
MIN_SPIKES_FOR_ISI = 50
# =====================================================

os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "sorting.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sorting")


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("START  plx=%s", PLX_FILE)
    log.info("OUT    %s", OUT_DIR)

    # ========== Cell 1: 读取 ==========
    recording = se.read_plexon(PLX_FILE)
    log.info("recording: %s", recording)

    # ========== Cell 2: 设置属性 ==========
    n = recording.get_num_channels()
    recording.delete_property("physical_unit")
    recording.delete_property("gain_to_physical_unit")
    recording.delete_property("offset_to_physical_unit")
    recording.set_channel_gains(np.full(n, 0.19501953125))
    recording.set_channel_offsets(np.zeros(n))

    dummy_locations = np.zeros((n, 2))
    dummy_locations[:, 0] = np.arange(n)
    recording.set_channel_locations(dummy_locations)
    recording.set_property("group", np.arange(n))
    log.info("channels=%d", n)

    # ========== Cell 3: 滤波 ==========
    recording_filtered = spre.bandpass_filter(
        recording, freq_min=300, freq_max=6000
    )
    log.info("filtered: %s", recording_filtered)

    # ========== Cell 4: sorting ==========
    sorting_test = ss.run_sorter_by_property(
        sorter_name="mountainsort5",
        recording=recording_filtered,
        grouping_property="group",
        folder=SORTER_FOLDER,
        remove_existing_folder=True,
    )
    log.info("sorting: %s", sorting_test)

    # ========== Cell 5: 清理越界 spike ==========
    sorting_clean = remove_excess_spikes(sorting_test, recording_filtered)
    log.info("cleaned: %s", sorting_clean)

    # ========== Cell 6: 存内存 + 拆分 ==========
    recording_mem = recording_filtered.save_to_memory()
    recording_dict = recording_mem.split_by("group", outputs="dict")

    # ========== Cell 7: analyzer ==========
    analyzers_by_group = {}
    # 用 clean 后的 sorting，单元属性从 clean 里取
    unit_groups = sorting_clean.get_property("group")

    for group_id, sub_rec in recording_dict.items():
        try:
            unit_ids_in_group = [
                uid for uid, g in zip(sorting_clean.get_unit_ids(), unit_groups)
                if g == group_id
            ]
            if len(unit_ids_in_group) == 0:
                log.info("group %s: no units, skip", group_id)
                continue

            sorting_sub = sorting_clean.select_units(unit_ids_in_group)
            analyzer_sub = si.create_sorting_analyzer(
                sorting=sorting_sub,
                recording=sub_rec,
                format="memory",
            )
            analyzers_by_group[group_id] = analyzer_sub
        except Exception:
            log.error("group %s analyzer failed:\n%s", group_id, traceback.format_exc())

    log.info("analyzers built: %d groups", len(analyzers_by_group))

    # ========== Cell 8: QC ==========
    qc_results = {}
    for group_id, analyzer in analyzers_by_group.items():
        try:
            analyzer.compute("random_spikes")
            analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
            analyzer.compute("templates")
            analyzer.compute("noise_levels")
            analyzer.compute("spike_amplitudes")
            analyzer.compute(
                "quality_metrics",
                metric_names=[
                    "num_spikes", "firing_rate", "presence_ratio", "snr",
                    "amplitude_median", "amplitude_cutoff", "isi_violation",
                ],
            )
            qm = analyzer.get_extension("quality_metrics").get_data()
            qm["group"] = group_id
            qc_results[group_id] = qm
            log.info("QC done group %s (%d units)", group_id, len(qm))
        except Exception:
            log.error("group %s QC failed:\n%s", group_id, traceback.format_exc())

    if not qc_results:
        log.error("no QC results, abort")
        return

    # ========== Cell 9: 合并 QC ==========
    qc_table = pd.concat(qc_results.values(), axis=0)
    qc_table.index.name = "unit_id"
    qc_table = qc_table.reset_index()

    # ========== Cell 10: QC 阈值 ==========
    qc_table["basic_qc"] = (
        (qc_table["num_spikes"] >= MIN_SPIKES)
        & (qc_table["firing_rate"] >= MIN_FIRING_RATE)
        & (qc_table["presence_ratio"] >= MIN_PRESENCE_RATIO)
    )
    qc_table["signal_qc"] = qc_table["snr"] >= MIN_SNR
    qc_table["physiology_qc"] = (
        qc_table["isi_violations_ratio"] <= MAX_ISI_VIOLATION_RATIO
    )
    qc_table["isi_reliable"] = qc_table["num_spikes"] >= MIN_SPIKES_FOR_ISI
    qc_table["pass_all"] = (
        qc_table["basic_qc"] & qc_table["signal_qc"] & qc_table["physiology_qc"]
    )

    def get_failure_reason(row):
        reasons = []
        if row["num_spikes"] < MIN_SPIKES: reasons.append("too_few_spikes")
        if row["firing_rate"] < MIN_FIRING_RATE: reasons.append("low_firing_rate")
        if row["presence_ratio"] < MIN_PRESENCE_RATIO: reasons.append("low_presence_ratio")
        if row["snr"] < MIN_SNR: reasons.append("low_snr")
        if row["isi_violations_ratio"] > MAX_ISI_VIOLATION_RATIO: reasons.append("high_isi_violation")
        return "PASS" if len(reasons) == 0 else "; ".join(reasons)

    qc_table["failure_reason"] = qc_table.apply(get_failure_reason, axis=1)

    def classify_unit(row):
        if not row["basic_qc"]: return "FAIL"
        if not row["signal_qc"]: return "FAIL"
        if not row["physiology_qc"]: return "FAIL"
        if not row["isi_reliable"]: return "PASS_LOW_CONFIDENCE"
        return "PASS"

    qc_table["qc_status"] = qc_table.apply(classify_unit, axis=1)

    qc_csv = os.path.join(OUT_DIR, "qc_table.csv")
    qc_table.to_csv(qc_csv, index=False)
    log.info("QC table saved: %s (%d units)", qc_csv, len(qc_table))

    print(qc_table[["group", "unit_id", "qc_status", "failure_reason"]].to_string(index=False))

    # ========== Cell 11: 批量画波形（不再手动 unit_id=9） ==========
    plot_dir = os.path.join(OUT_DIR, "waveforms")
    os.makedirs(plot_dir, exist_ok=True)

    # 只画通过 QC 的单元
    pass_units = qc_table[qc_table["qc_status"].isin(["PASS", "PASS_LOW_CONFIDENCE"])]

    for _, row in pass_units.iterrows():
        unit_id = int(row["unit_id"])
        group_id = row["group"]
        try:
            rec = analyzers_by_group[group_id].recording
            fs = rec.get_sampling_frequency()
            spikes = sorting_clean.get_unit_spike_train(unit_id)

            pre = int(1.0 / 1000 * fs)
            post = int(2.0 / 1000 * fs)

            manual_waveforms = []
            for spike in spikes:
                start = int(spike) - pre
                end = int(spike) + post
                if start < 0 or end > rec.get_num_frames():
                    continue
                w = rec.get_traces(start_frame=start, end_frame=end,
                                   return_in_uV=True)[:, 0]
                baseline = np.median(w[:pre])
                w = w - baseline
                manual_waveforms.append(w)

            manual_waveforms = np.asarray(manual_waveforms)
            if manual_waveforms.size == 0:
                continue

            time_ms = (np.arange(manual_waveforms.shape[1]) - pre) / fs * 1000
            mean_waveform = np.mean(manual_waveforms, axis=0)

            fig, ax = plt.subplots(figsize=(7, 4))
            for w in manual_waveforms:
                ax.plot(time_ms, w, linewidth=0.5, alpha=0.15)
            ax.plot(time_ms, mean_waveform, linewidth=2.5)
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.axhline(0, linewidth=0.8)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title(f"Unit {unit_id} | Group {group_id} | {row['qc_status']}")
            plt.tight_layout()
            out_png = os.path.join(plot_dir, f"unit_g{group_id}_u{unit_id}.png")
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            log.error("plot unit %s/%s failed:\n%s",
                      group_id, unit_id, traceback.format_exc())
            plt.close("all")

    log.info("DONE in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()