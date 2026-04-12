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
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

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

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset env / 重置环境
            env_obs = self.env.reset(self.usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            # 每局监控累积器：用于评估新增特征是否被有效利用
            feat_nearest_danger_sum = 0.0
            feat_second_exists_sum = 0.0
            feat_double_pressure_sum = 0.0
            feat_treasure_visible_sum = 0.0
            feat_treasure_dist_sum = 0.0
            feat_treasure_dist_cnt = 0
            feat_buff_visible_sum = 0.0
            feat_buff_dist_sum = 0.0
            feat_buff_dist_cnt = 0
            feat_post_speedup_sum = 0.0
            feat_speedup_eta_sum = 0.0

            flash_cnt = 0
            flash_in_danger_cnt = 0
            flash_waste_cnt = 0

            treasure_approach_cnt = 0
            treasure_approach_den = 0
            buff_approach_cnt = 0
            buff_approach_den = 0

            pre_reward_sum = 0.0
            post_reward_sum = 0.0
            post_steps = 0
            post_terminated = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

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

                # 从当前/下一时刻特征提取监控统计
                curr_nearest = self._feature_group(
                    obs_data.feature, "nearest_monster_dyn")
                curr_second = self._feature_group(
                    obs_data.feature, "second_nearest_monster_dyn")
                curr_treasure = self._feature_group(
                    obs_data.feature, "nearest_treasure_target")
                curr_buff = self._feature_group(
                    obs_data.feature, "nearest_buff_target")
                curr_phase = self._feature_group(
                    obs_data.feature, "phase_hint")
                next_treasure = self._feature_group(
                    _obs_data.feature, "nearest_treasure_target")
                next_buff = self._feature_group(
                    _obs_data.feature, "nearest_buff_target")
                next_phase = self._feature_group(
                    _obs_data.feature, "phase_hint")

                nearest_danger = float(curr_nearest[3] > 0.5)
                second_exists = float(curr_second[0] > 0.5)
                double_pressure = float(curr_second[3] > 0.5)
                tre_visible = float(curr_treasure[0] > 0.5)
                buff_visible = float(curr_buff[0] > 0.5)
                post_speedup = float(curr_phase[0] > 0.5)

                feat_nearest_danger_sum += nearest_danger
                feat_second_exists_sum += second_exists
                feat_double_pressure_sum += double_pressure
                feat_treasure_visible_sum += tre_visible
                feat_buff_visible_sum += buff_visible
                feat_post_speedup_sum += post_speedup
                feat_speedup_eta_sum += float(curr_phase[1])

                if tre_visible > 0.5:
                    feat_treasure_dist_sum += float(curr_treasure[1])
                    feat_treasure_dist_cnt += 1
                    if float(next_treasure[0]) > 0.5:
                        treasure_approach_den += 1
                        if float(next_treasure[1]) < float(curr_treasure[1]):
                            treasure_approach_cnt += 1

                if buff_visible > 0.5:
                    feat_buff_dist_sum += float(curr_buff[1])
                    feat_buff_dist_cnt += 1
                    if float(next_buff[0]) > 0.5:
                        buff_approach_den += 1
                        if float(next_buff[1]) < float(curr_buff[1]):
                            buff_approach_cnt += 1

                act_is_flash = int(act >= 8)
                flash_cnt += act_is_flash
                if act_is_flash and nearest_danger > 0.5:
                    flash_in_danger_cnt += 1
                if act_is_flash and nearest_danger < 0.5 and tre_visible < 0.5 and buff_visible < 0.5:
                    flash_waste_cnt += 1

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get(
                    "reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                # 分阶段奖励统计：观察新增阶段特征是否带来后期收益
                if post_speedup > 0.5:
                    post_reward_sum += float(reward[0])
                    post_steps += 1
                else:
                    pre_reward_sum += float(reward[0])

                # Terminal reward / 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)

                    if terminated:
                        final_reward[0] = -10.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 10.0
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                    # 标记是否在后期阶段阵亡
                    if terminated and float(next_phase[0]) > 0.5:
                        post_terminated = 1.0

                # Build sample frame / 构造样本帧
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
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        step_safe = max(step, 1)
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "feat_nearest_danger_rate": round(feat_nearest_danger_sum / step_safe, 4),
                            "feat_second_exists_rate": round(feat_second_exists_sum / step_safe, 4),
                            "feat_double_pressure_rate": round(feat_double_pressure_sum / step_safe, 4),
                            "feat_treasure_visible_rate": round(feat_treasure_visible_sum / step_safe, 4),
                            "feat_treasure_dist_mean": round(
                                feat_treasure_dist_sum /
                                max(feat_treasure_dist_cnt, 1), 4
                            ),
                            "feat_buff_visible_rate": round(feat_buff_visible_sum / step_safe, 4),
                            "feat_buff_dist_mean": round(
                                feat_buff_dist_sum /
                                max(feat_buff_dist_cnt, 1), 4
                            ),
                            "feat_post_speedup_rate": round(feat_post_speedup_sum / step_safe, 4),
                            "feat_speedup_eta_mean": round(feat_speedup_eta_sum / step_safe, 4),
                            "act_flash_rate": round(float(flash_cnt) / step_safe, 4),
                            "act_flash_in_danger_rate": round(
                                float(flash_in_danger_cnt) /
                                max(flash_cnt, 1), 4
                            ),
                            "act_flash_waste_rate": round(
                                float(flash_waste_cnt) / max(flash_cnt, 1), 4
                            ),
                            "act_treasure_approach_rate": round(
                                float(treasure_approach_cnt) /
                                max(treasure_approach_den, 1), 4
                            ),
                            "act_buff_approach_rate": round(
                                float(buff_approach_cnt) /
                                max(buff_approach_den, 1), 4
                            ),
                            "pre_reward_mean": round(pre_reward_sum / max(step_safe - post_steps, 1), 4),
                            "post_reward_mean": round(post_reward_sum / max(post_steps, 1), 4),
                            "post_steps_mean": round(float(post_steps), 4),
                            "post_terminated_rate": round(post_terminated, 4),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
