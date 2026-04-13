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


def _add_single_metric_panel(monitor, panel_name, panel_name_en, metric_key):
    """Add one panel with exactly one metric.

    每个图只放一个变量，便于快速定位波动来源。
    """
    return (
        monitor.add_panel(name=panel_name, name_en=panel_name_en, type="line")
        .add_metric(metrics_name=metric_key, expr=f"avg({metric_key}{{}})")
        .end_panel()
    )


def _add_algorithm_group(monitor):
    """Build algorithm monitor group.

    算法指标组：每个指标单独一个图。
    """
    metrics = [
        ("CumReward", "cum_reward"),
        ("TotalLoss", "total_loss"),
        ("ValueLoss", "value_loss"),
        ("PolicyLoss", "policy_loss"),
        ("EntropyLoss", "entropy_loss"),
        ("GradClipNorm", "grad_clip_norm"),
        ("ClipFrac", "clip_frac"),
        ("ExplainedVar", "explained_var"),
        ("AdvMean", "adv_mean"),
        ("RetMean", "ret_mean"),
    ]

    monitor = monitor.add_group(group_name="算法指标", group_name_en="algorithm")
    for panel_name, metric_key in metrics:
        monitor = _add_single_metric_panel(
            monitor=monitor,
            panel_name=panel_name,
            panel_name_en=f"algorithm_{metric_key}",
            metric_key=metric_key,
        )
    return monitor.end_group()


def _add_episode_group(monitor, group_name, group_name_en, prefix):
    """Add train/val episode metric group.

    按给定前缀（train/val）添加同构指标面板，每个指标单独一个图。
    """
    p = f"{prefix}_"
    metrics = [
        ("Reward", f"{p}reward"),
        ("TotalScore", f"{p}total_score"),
        ("StepScore", f"{p}step_score"),
        ("TreasureScore", f"{p}treasure_score"),
        ("Treasures", f"{p}treasures"),
        ("Steps", f"{p}steps"),
        ("SpeedupReached", f"{p}speedup_reached"),
        ("Pre_Steps", f"{p}pre_steps"),
        ("Post_Steps", f"{p}post_steps"),
        ("Pre_TotalR", f"{p}pre_total_r"),
        ("Post_TotalR", f"{p}post_total_r"),
        ("Pre_ShapedR", f"{p}pre_shaped_r"),
        ("Post_ShapedR", f"{p}post_shaped_r"),
        ("Pre_StepGain", f"{p}pre_step_gain"),
        ("Post_StepGain", f"{p}post_step_gain"),
        ("Pre_TreaGain", f"{p}pre_trea_gain"),
        ("Post_TreaGain", f"{p}post_trea_gain"),
        ("Pre_TotalGain", f"{p}pre_total_gain"),
        ("Post_TotalGain", f"{p}post_total_gain"),
        ("Pre_Terminal", f"{p}pre_terminal"),
        ("Post_Terminal", f"{p}post_terminal"),
        ("Post_Terminated", f"{p}post_terminated"),
        ("TerminatedRate", f"{p}terminated_rate"),
        ("CompletedRate", f"{p}completed_rate"),
        ("AbnormalTrunc", f"{p}abnormal_trunc"),
        ("Final_Danger", f"{p}final_danger"),
        ("Final_TreaDist", f"{p}final_trea_dist"),
        ("FlashCount", f"{p}flash_count"),
        ("Last_FlashUsed", f"{p}last_flash_used"),
        ("Last_FlashReady", f"{p}last_flash_ready"),
        ("Last_FlashLegal", f"{p}last_flash_legal"),
        ("Final_VisibleTre", f"{p}final_visible_tre"),
        # 诊断项：保留调参所需可解释性。
        ("DistShapingMean", f"{p}dist_shaping_mean"),
        ("TreasureApproachMean", f"{p}treasure_approach_mean"),
        ("DangerPenaltyMean", f"{p}danger_penalty_mean"),
        ("FlashRewardMean", f"{p}flash_reward_mean"),
        ("FlashPenaltyMean", f"{p}flash_penalty_mean"),
    ]

    monitor = monitor.add_group(
        group_name=group_name, group_name_en=group_name_en)
    for panel_name, metric_key in metrics:
        monitor = _add_single_metric_panel(
            monitor=monitor,
            panel_name=panel_name,
            panel_name_en=f"{prefix}_{metric_key}",
            metric_key=metric_key,
        )
    return monitor.end_group()


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    monitor = monitor.title("峡谷追猎")
    monitor = _add_algorithm_group(monitor)

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
