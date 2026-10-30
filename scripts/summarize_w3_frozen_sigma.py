#!/usr/bin/env python3
"""Read-only postcheck of sigma-error association and static velocity bins.

This analysis is retrospective; it never fits or selects a model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from factr2_next.data_collection.quality import sha256, write_json
from diagnose_w3_next_u import aligned_states
import run_w3_next_u_ablation as core
from run_w3_frozen_sigma import verify


def summarize(folder):
    folder = Path(folder).resolve()
    verify(folder)
    state = json.loads((folder/"state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("frozen_sigma_postcheck_requires_complete")
    lock = json.loads((folder/"lock.json").read_text())
    parent = Path(lock["parent"])
    cfg, manifest = core.verify(parent)
    comparison = json.loads((folder/"evaluation/comparison.json").read_text())
    diagnosis = json.loads((folder/"evaluation/previous_diagnosis.json").read_text())
    result = dict(scope="retrospective associations; no p-values for correlated windows; no model selection",
                  source_sha256=sha256(__file__), comparison_sha256=sha256(folder/"evaluation/comparison.json"),
                  original_parent_lock_sha256=sha256(parent/"lock.json"), velocity={}, sigma_error={})
    for role in ("train", "val", "test"):
        ds = core.dataset(manifest, cfg, role)
        states = aligned_states(ds, cfg["data"]["keys"])
        static = states["kind"] == "static"
        items = {}
        for j in (0, 3):
            velocity = np.rad2deg(states["x"][static, 7+j])
            spans = []
            for ep in np.unique(states["episode_index"][static]):
                q = states["x"][(states["episode_index"] == ep) & static, j][-100:]
                spans.append(float(np.rad2deg(np.ptp(q))))
            if not len(velocity):
                continue
            items[f"J{j+1}"] = dict(static_frames=int(static.sum()),
                velocity_deg_s_quantiles={str(p):float(np.percentile(velocity,p)) for p in (0,25,50,75,100)},
                abs_velocity_deg_s_quantiles={str(p):float(np.percentile(abs(velocity),p)) for p in (50,95,99.5)},
                static_tail_at_most_1s_position_span_deg=spans,
                fraction_abs_velocity_le_diagnostic_deadband_0_1_deg_s=float(np.mean(abs(velocity) <= .1)))
        result["velocity"][role] = items
        if role in ("val", "test"):
            with np.load(folder/"evaluation"/f"{role}_LEARNED_SIGMA.npz", allow_pickle=False) as h:
                error, sigma = h["error_nm"], np.sqrt(h["variance_free_nm2"])
                np.testing.assert_array_equal(h["timestamp_ns"], states["timestamp_ns"])
                np.testing.assert_array_equal(h["episode_index"], states["episode_index"])
            result["sigma_error"][role] = dict(
                spearman_sigma_vs_abs_error=[float(spearmanr(sigma[:,j],abs(error[:,j])).statistic) for j in range(7)],
                interpretation="association only; negative value means larger errors tend to receive smaller sigma")
    # Compare observed bins to nominal 12-bit symmetric mapping. Matching is
    # evidence of quantization, not proof of exact physical velocity or cause.
    result["nominal_zero_bin_reference_deg_s"] = dict(
        J1_45rad_s_range=float(np.rad2deg(45/4095)), J4_20rad_s_range=float(np.rad2deg(20/4095)))
    output = folder/"evaluation"
    write_json(output/"postcheck.json", result)
    a = comparison["splits"]["test"]["arms"]["FIXED_SIGMA"]["overall"]
    b = comparison["splits"]["test"]["arms"]["LEARNED_SIGMA"]["overall"]
    lines = ["# 本轮结论与下一步", "",
        f"已完成旧模型 J1/J4 分组诊断及冻结均值的方差消融。方差头只含 {comparison['head_parameters']} 个参数，"
        f"实际训练 {state['completed_epochs']} 轮，第 {state['best_epoch']} 轮验证 NLL 最佳；完整运行约 {state['elapsed_seconds']:.1f} 秒。"
        f"两组使用同一均值与残差，回放与原 NEXT_MSE 最大差异为 "
        f"{max(s['mean_max_abs_replay_difference_nm'] for s in comparison['splits'].values()):.8g} Nm，"
        f"测试 MSE 保持 {a['physical_mse_nm2']:.8f} Nm²。", "",
        "## 1. 学习 sigma 尚未优于固定 sigma", "",
        f"测试 NLL：固定 {a['marginal_nll']:.6f}，学习 {b['marginal_nll']:.6f}（越小越好）。验证 NLL 虽有改善，测试 NLL 反而变差。", "",
        "覆盖率和区间宽度必须一起判断，当前不能据此提供可靠的 95% 外力置信区间。逐关节数字见 COMPARISON_ZH.md。", "",
        "学习 sigma 与绝对残差的 Spearman 相关性：", "",
        "|关节|验证集|测试集|", "|---|---:|---:|"]
    for j in (0,3,5):
        lines.append(f"|J{j+1}|{result['sigma_error']['val']['spearman_sigma_vs_abs_error'][j]:.3f}|"
                     f"{result['sigma_error']['test']['spearman_sigma_vs_abs_error'][j]:.3f}|")
    bounds = diagnosis["splits"]["test"]["variance_bound_fractions"]
    lines += ["", "负相关意味着较大的绝对残差反而倾向得到较小的 sigma，误差排序也可能不可靠；上述统计仅描述关联，不对重叠窗口作独立显著性检验。", "",
        "## 2. 原联合 NEXT-U 的均值与方差边界", "",
        f"旧方差下限占比：{bounds['floor_per_joint']}；上限占比：{bounds['ceiling_per_joint']}。", ""]
    for joint, group in (("J1", "kind:single_joint"), ("J4", "kind:static")):
        m = diagnosis["splits"]["test"]["joints"][joint].get(group)
        if m is not None:
            lines += [f"{joint} 的 {group} 片段，NEXT RMSE 为 {m['next_rmse_nm']:.5f} Nm，"
                      f"联合 NEXT-U 为 {m['u_rmse_nm']:.5f} Nm。", ""]
    lines += ["该配对比较使用同一反馈力矩，均值退化不能仅归因为显示了更多电流噪声。损失权重或会话差异的具体因果仍未证明。", "",
        f"测试最终帧状态超出训练逐轴 min/max 的比例为 "
        f"{diagnosis['splits']['test']['outside_any_final_state_fraction']*100:.3f}%。"
        "单轴范围检查不能排除组合状态、历史或会话分布偏移。", "",
        "## 3. 静置速度反馈存在明显离散档位", "",
        "|关节|测试静置速度中位数 deg/s|静置尾段最大位置跨度 deg|", "|---|---:|---:|"]
    for joint in ("J1", "J4"):
        if joint not in result["velocity"]["test"]:
            continue
        m = result["velocity"]["test"][joint]
        lines.append(f"|{joint}|{m['velocity_deg_s_quantiles']['50']:.6f}|{max(m['static_tail_at_most_1s_position_span_deg']):.6f}|")
    lines += ["", "该中位数与 ±45 rad/s、±20 rad/s 的 12-bit 对称速度编码近零半档位相符，提示明显的量化影响；这不是证明机械臂实际持续以该速度运动。静置标签来自原采集协议，位置跨度也只受编码器分辨率约束，不能证明真实位置严格不变。", "",
        "本轮原始速度分组死区预设为 0.1 deg/s，低于这些近零档位。因此‘正向/负向/换向’仅是反馈值符号标签，不能直接当成真实机械运动方向。这是诊断标签的限制，本轮没有修改输入速度或重新训练模型。", "",
        "## 下一步", "",
        "优先核对速度反馈量化、时间对齐和独立会话差异，在新的独立标定会话上评估区间校准；静置与换向标签应结合位置变化和反馈分辨率验证。先维持现有均值模型，不将本轮方差头用于自动降权、清零或在线报警。已知接触参考仍是验证外力绝对精度、加载增益和卸载恢复的必要后续数据。完整 MOB＋GRU 继续作为独立后续对照。", "",
        "本轮为单 seed、既有会话的回顾性试验。均值和方差的选模复用了同一验证集，验证指标可能偏乐观；仍需新的独立会话验证。没有在这些测试结果上继续调整方差头或超参数。"]
    (output/"INTERPRETATION_ZH.md").write_text("\n".join(lines)+"\n")
    return output/"INTERPRETATION_ZH.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    print(summarize(args.experiment))
