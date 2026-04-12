#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_episode_group(monitor, group_name, group_name_en, prefix):
    """Add train/val episode metric group with a fixed field schema.

    按给定前缀（train/val）添加同构指标面板。
    """
    p = f"{prefix}_"
    return (
        monitor.add_group(group_name=group_name, group_name_en=group_name_en)
        .add_panel(name="总奖励与总分", name_en=f"{prefix}_reward_score", type="line")
        .add_metric(metrics_name=f"{p}reward", expr=f"avg({p}reward{{}})")
        .add_metric(metrics_name=f"{p}total_score", expr=f"avg({p}total_score{{}})")
        .end_panel()
        .add_panel(name="得分拆分", name_en=f"{prefix}_score_breakdown", type="line")
        .add_metric(metrics_name=f"{p}step_score", expr=f"avg({p}step_score{{}})")
        .add_metric(metrics_name=f"{p}treasure_score", expr=f"avg({p}treasure_score{{}})")
        .add_metric(metrics_name=f"{p}treasures", expr=f"avg({p}treasures{{}})")
        .end_panel()
        .add_panel(name="对局进度", name_en=f"{prefix}_progress", type="line")
        .add_metric(metrics_name=f"{p}steps", expr=f"avg({p}steps{{}})")
        .add_metric(metrics_name=f"{p}speedup_reached", expr=f"avg({p}speedup_reached{{}})")
        .end_panel()
        .add_panel(name="阶段步数", name_en=f"{prefix}_phase_steps", type="line")
        .add_metric(metrics_name=f"{p}pre_steps", expr=f"avg({p}pre_steps{{}})")
        .add_metric(metrics_name=f"{p}post_steps", expr=f"avg({p}post_steps{{}})")
        .end_panel()
        .add_panel(name="阶段总奖励", name_en=f"{prefix}_phase_total_r", type="line")
        .add_metric(metrics_name=f"{p}pre_total_r", expr=f"avg({p}pre_total_r{{}})")
        .add_metric(metrics_name=f"{p}post_total_r", expr=f"avg({p}post_total_r{{}})")
        .end_panel()
        .add_panel(name="阶段塑形奖励", name_en=f"{prefix}_phase_shaped", type="line")
        .add_metric(metrics_name=f"{p}pre_shaped_r", expr=f"avg({p}pre_shaped_r{{}})")
        .add_metric(metrics_name=f"{p}post_shaped_r", expr=f"avg({p}post_shaped_r{{}})")
        .end_panel()
        .add_panel(name="步数分增量", name_en=f"{prefix}_step_gain", type="line")
        .add_metric(metrics_name=f"{p}pre_step_gain", expr=f"avg({p}pre_step_gain{{}})")
        .add_metric(metrics_name=f"{p}post_step_gain", expr=f"avg({p}post_step_gain{{}})")
        .end_panel()
        .add_panel(name="宝箱分增量", name_en=f"{prefix}_trea_gain", type="line")
        .add_metric(metrics_name=f"{p}pre_trea_gain", expr=f"avg({p}pre_trea_gain{{}})")
        .add_metric(metrics_name=f"{p}post_trea_gain", expr=f"avg({p}post_trea_gain{{}})")
        .end_panel()
        .add_panel(name="总分增量", name_en=f"{prefix}_total_gain", type="line")
        .add_metric(metrics_name=f"{p}pre_total_gain", expr=f"avg({p}pre_total_gain{{}})")
        .add_metric(metrics_name=f"{p}post_total_gain", expr=f"avg({p}post_total_gain{{}})")
        .end_panel()
        .add_panel(name="终局奖励", name_en=f"{prefix}_terminal_reward", type="line")
        .add_metric(metrics_name=f"{p}pre_terminal", expr=f"avg({p}pre_terminal{{}})")
        .add_metric(metrics_name=f"{p}post_terminal", expr=f"avg({p}post_terminal{{}})")
        .end_panel()
        .add_panel(name="结果占比", name_en=f"{prefix}_result_ratio", type="line")
        .add_metric(metrics_name=f"{p}post_terminated", expr=f"avg({p}post_terminated{{}})")
        .add_metric(metrics_name=f"{p}terminated_rate", expr=f"avg({p}terminated_rate{{}})")
        .add_metric(metrics_name=f"{p}completed_rate", expr=f"avg({p}completed_rate{{}})")
        .end_panel()
        .add_panel(name="异常截断", name_en=f"{prefix}_abnormal", type="line")
        .add_metric(metrics_name=f"{p}abnormal_trunc", expr=f"avg({p}abnormal_trunc{{}})")
        .end_panel()
        .add_panel(name="末帧风险与目标", name_en=f"{prefix}_final_state", type="line")
        .add_metric(metrics_name=f"{p}final_danger", expr=f"avg({p}final_danger{{}})")
        .add_metric(metrics_name=f"{p}final_trea_dist", expr=f"avg({p}final_trea_dist{{}})")
        .add_metric(metrics_name=f"{p}final_visible_tre", expr=f"avg({p}final_visible_tre{{}})")
        .end_panel()
        .add_panel(name="闪现使用", name_en=f"{prefix}_flash_usage", type="line")
        .add_metric(metrics_name=f"{p}flash_count", expr=f"avg({p}flash_count{{}})")
        .add_metric(metrics_name=f"{p}last_flash_used", expr=f"avg({p}last_flash_used{{}})")
        .add_metric(metrics_name=f"{p}last_flash_ready", expr=f"avg({p}last_flash_ready{{}})")
        .end_panel()
        .add_panel(name="闪现合法性", name_en=f"{prefix}_flash_legal", type="line")
        .add_metric(metrics_name=f"{p}last_flash_legal", expr=f"avg({p}last_flash_legal{{}})")
        .end_panel()
        .add_panel(name="奖励诊断一", name_en=f"{prefix}_reward_diag_1", type="line")
        .add_metric(metrics_name=f"{p}dist_shaping_mean", expr=f"avg({p}dist_shaping_mean{{}})")
        .add_metric(metrics_name=f"{p}treasure_approach_mean", expr=f"avg({p}treasure_approach_mean{{}})")
        .add_metric(metrics_name=f"{p}danger_penalty_mean", expr=f"avg({p}danger_penalty_mean{{}})")
        .end_panel()
        .add_panel(name="奖励诊断二", name_en=f"{prefix}_reward_diag_2", type="line")
        .add_metric(metrics_name=f"{p}flash_reward_mean", expr=f"avg({p}flash_reward_mean{{}})")
        .add_metric(metrics_name=f"{p}flash_penalty_mean", expr=f"avg({p}flash_penalty_mean{{}})")
        .end_panel()
        .end_group()
    )


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    monitor = (
        monitor.title("峡谷追猎")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="回报与总损失", name_en="reward_total_loss", type="line")
        .add_metric(metrics_name="cum_reward", expr="avg(cum_reward{})")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="策略与价值损失", name_en="policy_value_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="探索与更新强度", name_en="explore_update", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .add_metric(metrics_name="grad_clip_norm", expr="avg(grad_clip_norm{})")
        .add_metric(metrics_name="clip_frac", expr="avg(clip_frac{})")
        .end_panel()
        .add_panel(name="优势与回报拟合", name_en="adv_ret_fit", type="line")
        .add_metric(metrics_name="explained_var", expr="avg(explained_var{})")
        .add_metric(metrics_name="adv_mean", expr="avg(adv_mean{})")
        .add_metric(metrics_name="ret_mean", expr="avg(ret_mean{})")
        .end_panel()
        .end_group()
    )

    monitor = _add_episode_group(
        monitor=monitor,
        group_name="Train 指标",
        group_name_en="train_metrics",
        prefix="train",
    )

    monitor = _add_episode_group(
        monitor=monitor,
        group_name="Val 指标",
        group_name_en="val_metrics",
        prefix="val",
    )

    return monitor.build()
