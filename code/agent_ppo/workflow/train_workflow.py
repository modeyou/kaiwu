#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import os
import time
import copy

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(
            "usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.train_episode_cnt = 0
        self.val_episode_cnt = 0
        self.val_interval = max(
            int(getattr(Config, "TRAIN_VAL_INTERVAL", 10)), 1)
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

        # 训练/验证配置拆分：train 用前 80%，val 用后 20% 地图
        self.train_usr_conf, self.val_usr_conf = self._build_train_val_confs(
            usr_conf)
        train_maps = self.train_usr_conf["env_conf"].get("map", [])
        val_maps = self.val_usr_conf["env_conf"].get("map", [])
        self.logger.info(
            f"[train/val split] interval={self.val_interval} train_maps={train_maps} val_maps={val_maps}"
        )

        # 特征分段索引，用于从 obs_data.feature 中提取监控统计
        names = [
            "hero_self",
            "monster_1",
            "monster_2",
            "map_local",
            "legal_action",
            "progress",
            "nearest_monster_dyn",
            "second_nearest_monster_dyn",
            "nearest_treasure_target",
            "nearest_buff_target",
            "phase_hint",
        ]
        self.feature_slices = {}
        start = 0
        for name, dim in zip(names, Config.FEATURES):
            self.feature_slices[name] = (start, start + dim)
            start += dim

    def _feature_group(self, feature, name):
        """Extract a named feature group from flattened feature vector.

        从拼接后的特征向量中提取指定分组。
        """
        start, end = self.feature_slices[name]
        return np.asarray(feature, dtype=np.float32)[start:end]

    def _build_train_val_confs(self, usr_conf):
        """Build train/val env configs from user config.

        基于用户配置拆分 train/val 地图，其他环境参数保持一致。
        """
        train_conf = copy.deepcopy(usr_conf)
        val_conf = copy.deepcopy(usr_conf)

        env_conf = usr_conf.get("env_conf", {})
        maps = list(env_conf.get("map", []))

        if len(maps) >= 4:
            split_idx = max(1, int(round(len(maps) * 0.8)))
            split_idx = min(split_idx, len(maps) - 1)
            train_maps = maps[:split_idx]
            val_maps = maps[split_idx:]
        elif len(maps) >= 2:
            train_maps = maps[:-1]
            val_maps = maps[-1:]
        elif len(maps) == 1:
            train_maps = maps
            val_maps = maps
        else:
            # 空地图配置兜底，避免 reset 失败
            train_maps = [1]
            val_maps = [1]

        train_conf.setdefault("env_conf", {})["map"] = train_maps
        val_conf.setdefault("env_conf", {})["map"] = val_maps

        # 为避免顺序偏置，默认训练和验证都随机抽图
        train_conf["env_conf"]["map_random"] = True
        val_conf["env_conf"]["map_random"] = True
        return train_conf, val_conf

    def _should_run_val(self):
        """Whether current episode should be validation.

        每累计 val_interval 局训练后，插入 1 局验证。
        """
        expected_val_cnt = self.train_episode_cnt // self.val_interval
        return self.val_episode_cnt < expected_val_cnt

    def _prefix_metrics(self, metrics, prefix):
        """Add mode prefix for monitor keys.

        统一为 train_/val_ 前缀，避免与环境面板指标重名。
        """
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # 决定当前局模式：训练 or 验证
            is_val = self._should_run_val()
            run_conf = self.val_usr_conf if is_val else self.train_usr_conf
            mode_str = "VAL" if is_val else "TRAIN"
            mode_prefix = "val" if is_val else "train"

            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset env / 重置环境
            env_obs = self.env.reset(run_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, _remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            if is_val:
                self.val_episode_cnt += 1
            else:
                self.train_episode_cnt += 1

            done = False
            step = 0
            total_reward = 0.0

            # train/val 统一统计字段（与需求清单一一对应）
            speedup_reached = 0.0
            pre_steps = 0.0
            post_steps = 0.0
            pre_total_r = 0.0
            post_total_r = 0.0
            pre_shaped_r = 0.0
            post_shaped_r = 0.0
            pre_step_gain = 0.0
            post_step_gain = 0.0
            pre_trea_gain = 0.0
            post_trea_gain = 0.0
            pre_terminal = 0.0
            post_terminal = 0.0
            post_terminated = 0.0
            terminated_rate = 0.0
            completed_rate = 0.0
            abnormal_trunc = 0.0
            flash_count = 0.0
            last_flash_used = 0.0
            last_flash_ready = 0.0
            last_flash_legal = 0.0
            final_visible_tre = 0.0
            final_danger = 0.0
            final_trea_dist = -1.0
            final_total_score = 0.0
            final_step_score = 0.0
            final_treasure_score = 0.0
            final_treasures = 0.0

            # 奖励诊断分项（用于调参回归）
            sum_dist_shaping = 0.0
            sum_treasure_approach = 0.0
            sum_danger_penalty = 0.0
            sum_flash_reward = 0.0
            sum_flash_penalty = 0.0

            self.logger.info(
                f"[{mode_str}] total_episode:{self.episode_cnt} train_episode:{self.train_episode_cnt} val_episode:{self.val_episode_cnt} start"
            )

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                # 训练局走随机采样，验证局走贪心动作
                act = self.agent.action_process(
                    act_data, is_stochastic=(not is_val))

                # Step env / 与环境交互
                _env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(
                    env_obs)

                # 从当前/下一时刻特征提取阶段与可见性信息
                curr_phase = self._feature_group(
                    obs_data.feature, "phase_hint")
                next_treasure = self._feature_group(
                    _obs_data.feature, "nearest_treasure_target")
                next_phase = self._feature_group(
                    _obs_data.feature, "phase_hint")
                post_speedup = float(curr_phase[0] > 0.5)
                speedup_reached = max(speedup_reached, post_speedup)
                if post_speedup > 0.5:
                    post_steps += 1.0
                else:
                    pre_steps += 1.0

                flash_count += float(act >= 8)
                last_flash_used = float(act >= 8)

                # 末帧可见宝箱比例（这里只记录“最近宝箱是否可见”）
                final_visible_tre = float(next_treasure[0] > 0.5)

                # 末帧闪现可用性（legal_action[8:16]）
                flash_legal = np.array(
                    _obs_data.legal_action[8:16], dtype=np.float32)
                last_flash_legal = float(
                    flash_legal.mean()) if flash_legal.size > 0 else 0.0
                last_flash_ready = float(flash_legal.sum() > 0.0)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get(
                    "reward", [0.0]), dtype=np.float32)
                reward_info = _remain_info.get("reward_info", {})

                # 记录奖励分项，便于诊断哪一项在主导训练。
                sum_dist_shaping += float(reward_info.get("dist_shaping", 0.0))
                sum_treasure_approach += float(
                    reward_info.get("treasure_approach", 0.0))
                sum_danger_penalty += float(
                    reward_info.get("danger_penalty", 0.0))
                sum_flash_reward += float(reward_info.get("flash_reward", 0.0))
                sum_flash_penalty += float(
                    reward_info.get("flash_penalty", 0.0))

                total_reward += float(reward[0])

                # 分阶段累计总奖励
                if post_speedup > 0.5:
                    post_total_r += float(reward[0])
                else:
                    pre_total_r += float(reward[0])

                # shaping 奖励：去掉得分增量项后的策略塑形部分
                shaped_r = (
                    float(reward_info.get("dist_shaping", 0.0))
                    + float(reward_info.get("treasure_approach", 0.0))
                    + float(reward_info.get("danger_penalty", 0.0))
                    + float(reward_info.get("double_pressure_penalty", 0.0))
                    + float(reward_info.get("flash_reward", 0.0))
                    + float(reward_info.get("flash_penalty", 0.0))
                    + float(reward_info.get("invalid_move_penalty", 0.0))
                )

                step_gain = float(reward_info.get("step_gain", 0.0))
                trea_gain = float(reward_info.get("treasure_gain", 0.0))
                if post_speedup > 0.5:
                    post_shaped_r += shaped_r
                    post_step_gain += step_gain
                    post_trea_gain += trea_gain
                else:
                    pre_shaped_r += shaped_r
                    pre_step_gain += step_gain
                    pre_trea_gain += trea_gain

                # 记录末帧危险度与最近宝箱距离
                min_monster_dist = float(
                    reward_info.get("min_monster_dist", -1.0))
                if min_monster_dist >= 0.0:
                    final_danger = 1.0 / (1.0 + min_monster_dist)
                final_trea_dist = float(reward_info.get(
                    "nearest_treasure_dist", final_trea_dist))

                # Terminal reward / 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    final_total_score = float(total_score)
                    final_step_score = float(env_info.get("step_score", 0.0))
                    final_treasure_score = float(
                        env_info.get("treasure_score", 0.0))
                    final_treasures = float(
                        env_info.get("treasures_collected", 0.0))
                    in_post_phase = float(next_phase[0]) > 0.5

                    if terminated:
                        # 阵亡按前后期区分惩罚，后期惩罚更重。
                        final_reward[0] = (
                            Config.TERMINAL_FAIL_POST if in_post_phase else Config.TERMINAL_FAIL_PRE
                        )
                        result_str = "FAIL"
                        terminated_rate = 1.0
                    else:
                        # 截断区分正常完成与异常截断。
                        max_step = int(env_info.get("max_step", run_conf.get(
                            "env_conf", {}).get("max_step", step)))
                        finished_steps = int(
                            env_info.get("finished_steps", step))
                        if finished_steps >= max_step:
                            final_reward[0] = Config.TERMINAL_COMPLETE
                            result_str = "WIN"
                            completed_rate = 1.0
                        else:
                            final_reward[0] = Config.TERMINAL_ABNORMAL
                            result_str = "ABNORMAL"
                            abnormal_trunc = 1.0

                    # 终局奖励按发生阶段分别累计
                    if in_post_phase:
                        post_terminal += float(final_reward[0])
                    else:
                        pre_terminal += float(final_reward[0])

                    self.logger.info(
                        f"[{mode_str} GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                    # 标记是否在后期阶段阵亡
                    if terminated and float(next_phase[0]) > 0.5:
                        post_terminated = 1.0

                # Build sample frame / 构造样本帧
                # 验证局不入样本池，避免把评估分布写进训练数据
                if not is_val:
                    frame = SampleData(
                        obs=np.array(obs_data.feature, dtype=np.float32),
                        legal_action=np.array(
                            obs_data.legal_action, dtype=np.float32),
                        act=np.array([act_data.action[0]], dtype=np.float32),
                        reward=reward,
                        done=np.array([float(done)], dtype=np.float32),
                        reward_sum=np.zeros(1, dtype=np.float32),
                        value=np.array(
                            act_data.value, dtype=np.float32).flatten()[:1],
                        next_value=np.zeros(1, dtype=np.float32),
                        advantage=np.zeros(1, dtype=np.float32),
                        prob=np.array(act_data.prob, dtype=np.float32),
                    )
                    collector.append(frame)

                # Episode end / 对局结束
                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + \
                            final_reward

                    # Monitor report / 监控上报
                    if self.monitor:
                        episode_metrics = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "total_score": round(final_total_score, 4),
                            "step_score": round(final_step_score, 4),
                            "treasure_score": round(final_treasure_score, 4),
                            "treasures": round(final_treasures, 4),
                            "steps": round(float(step), 4),
                            "speedup_reached": round(speedup_reached, 4),
                            "pre_steps": round(pre_steps, 4),
                            "post_steps": round(post_steps, 4),
                            "pre_total_r": round(pre_total_r, 4),
                            "post_total_r": round(post_total_r, 4),
                            "pre_shaped_r": round(pre_shaped_r, 4),
                            "post_shaped_r": round(post_shaped_r, 4),
                            "pre_step_gain": round(pre_step_gain, 4),
                            "post_step_gain": round(post_step_gain, 4),
                            "pre_trea_gain": round(pre_trea_gain, 4),
                            "post_trea_gain": round(post_trea_gain, 4),
                            "pre_total_gain": round(pre_step_gain + pre_trea_gain, 4),
                            "post_total_gain": round(post_step_gain + post_trea_gain, 4),
                            "pre_terminal": round(pre_terminal, 4),
                            "post_terminal": round(post_terminal, 4),
                            "post_terminated": round(post_terminated, 4),
                            "terminated_rate": round(terminated_rate, 4),
                            "completed_rate": round(completed_rate, 4),
                            "abnormal_trunc": round(abnormal_trunc, 4),
                            "final_danger": round(final_danger, 4),
                            "final_trea_dist": round(final_trea_dist, 4),
                            "flash_count": round(flash_count, 4),
                            "last_flash_used": round(last_flash_used, 4),
                            "last_flash_ready": round(last_flash_ready, 4),
                            "last_flash_legal": round(last_flash_legal, 4),
                            "final_visible_tre": round(final_visible_tre, 4),
                            "dist_shaping_mean": round(sum_dist_shaping / max(step, 1), 4),
                            "treasure_approach_mean": round(sum_treasure_approach / max(step, 1), 4),
                            "danger_penalty_mean": round(sum_danger_penalty / max(step, 1), 4),
                            "flash_reward_mean": round(sum_flash_reward / max(step, 1), 4),
                            "flash_penalty_mean": round(sum_flash_penalty / max(step, 1), 4),
                        }
                        self.monitor.put_data(
                            {os.getpid(): self._prefix_metrics(
                                episode_metrics, mode_prefix)}
                        )

                    if collector and (not is_val):
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
