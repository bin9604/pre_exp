# ========== Cell 1: 读取 ==========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface as si
from spikeinterface import sorters as ss
from spikeinterface.curation import remove_excess_spikes

PLX_FILE = r"D:\raw_data\orga_pre_exp\20260928\plx\subjects_2.exps_29.origin.20260928131241_565310.20260928__ce.00000.mex_20260928_181616\subjects_2.exps_29.origin.20260928131241_565310.20260928__ce.00000.plx"
recording = se.read_plexon(PLX_FILE)
print(recording)

# ========== Cell 2: 设置属性（合并） ==========
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

# ========== Cell 3: 滤波 ==========
recording_filtered = spre.bandpass_filter(
    recording, freq_min=300, freq_max=6000
)
print(recording_filtered)

# ========== Cell 4: sorting ==========
sorting_test = ss.run_sorter_by_property(
    sorter_name="mountainsort5",
    recording=recording_filtered,
    grouping_property="group",
    folder="ms5_by_channel_filtered",
    remove_existing_folder=True
)
print(sorting_test)

# ========== Cell 5: 清理越界 spike（用滤波 recording） ==========
sorting_clean = remove_excess_spikes(sorting_test, recording_filtered)

# ========== Cell 6: 滤波信号存内存 + 拆分 ==========
recording_mem = recording_filtered.save_to_memory()
recording_dict = recording_mem.split_by("group", outputs="dict")

# ========== Cell 7: 创建 analyzer ==========
analyzers_by_group = {}
unit_groups = sorting_test.get_property("group")

for group_id, sub_rec in recording_dict.items():
    unit_ids_in_group = [
        uid for uid, g in zip(sorting_test.get_unit_ids(), unit_groups)
        if g == group_id
    ]
    if len(unit_ids_in_group) == 0:
        continue

    sorting_sub = sorting_test.select_units(unit_ids_in_group)

    analyzer_sub = si.create_sorting_analyzer(
        sorting=sorting_sub,
        recording=sub_rec,
        format="memory"
    )
    analyzers_by_group[group_id] = analyzer_sub

# ========== Cell 8: QC ==========
qc_results = {}
for group_id, analyzer in analyzers_by_group.items():
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
        ]
    )
    qm = analyzer.get_extension("quality_metrics").get_data()
    qm["group"] = group_id
    qc_results[group_id] = qm

# ========== Cell 9: 合并 QC 表 ==========
qc_table = pd.concat(qc_results.values(), axis=0)
qc_table.index.name = "unit_id"
qc_table = qc_table.reset_index()

# ========== Cell 10: QC 阈值 ==========
MIN_SPIKES = 20
MIN_FIRING_RATE = 0.1
MIN_PRESENCE_RATIO = 0.8
MIN_SNR = 4.0
MAX_ISI_VIOLATION_RATIO = 0.01
MIN_SPIKES_FOR_ISI = 50

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
    qc_table["basic_qc"]
    & qc_table["signal_qc"]
    & qc_table["physiology_qc"]
)

def get_failure_reason(row):
    reasons = []
    if row["num_spikes"] < MIN_SPIKES:
        reasons.append("too_few_spikes")
    if row["firing_rate"] < MIN_FIRING_RATE:
        reasons.append("low_firing_rate")
    if row["presence_ratio"] < MIN_PRESENCE_RATIO:
        reasons.append("low_presence_ratio")
    if row["snr"] < MIN_SNR:
        reasons.append("low_snr")
    if row["isi_violations_ratio"] > MAX_ISI_VIOLATION_RATIO:
        reasons.append("high_isi_violation")
    return "PASS" if len(reasons) == 0 else "; ".join(reasons)

qc_table["failure_reason"] = qc_table.apply(get_failure_reason, axis=1)

def classify_unit(row):
    if not row["basic_qc"]: return "FAIL"
    if not row["signal_qc"]: return "FAIL"
    if not row["physiology_qc"]: return "FAIL"
    if not row["isi_reliable"]: return "PASS_LOW_CONFIDENCE"
    return "PASS"

qc_table["qc_status"] = qc_table.apply(classify_unit, axis=1)
print(qc_table[["group", "unit_id", "qc_status", "failure_reason"]].to_string(index=False))

# ========== Cell 11: 画波形 ==========
unit_id = 9
group_id = 12

rec = analyzers_by_group[group_id].recording
fs = rec.get_sampling_frequency()
spikes = sorting_test.get_unit_spike_train(unit_id)

pre = int(1.0 / 1000 * fs)
post = int(2.0 / 1000 * fs)

manual_waveforms = []
for spike in spikes:
    start = int(spike) - pre
    end = int(spike) + post
    if start < 0 or end > rec.get_num_frames():
        continue
    w = rec.get_traces(start_frame=start, end_frame=end, return_in_uV=True)[:, 0]
    baseline = np.median(w[:pre])
    w = w - baseline
    manual_waveforms.append(w)

manual_waveforms = np.asarray(manual_waveforms)
print("shape:", manual_waveforms.shape)

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
ax.set_title(f"Unit {unit_id} | Group {group_id}")
plt.tight_layout()
plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # 必须在 show 之前
