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
import random

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


class CurriculumManager:
    """Performance-driven curriculum manager.

    基于验证指标自动晋级课程阶段，避免按固定局数生硬切换。
    """

    def __init__(self):
        self.stage_names = ["easy", "medium", "hard", "expert"]
        self.stage_idx = 0
        self.train_episodes_in_stage = 0
        self.promotion_streak = 0
        self.warmup_episodes = max(
            int(getattr(Config, "CURRICULUM_WARMUP_EPISODES", 100)), 1
        )
        self.promotion_window = max(
            int(getattr(Config, "CURRICULUM_PROMOTION_WINDOW", 5)), 1
        )

        # 每个阶段的环境覆盖配置（tuple 表示 [min, max] 采样）。
        self.stage_overrides = {
            "easy": {
                "treasure_count": (9, 10),
                "buff_count": (2, 2),
                "monster_interval": (2000, 2000),
                "monster_speedup": (2000, 2000),
                "max_step": 600,
            },
            "medium": {
                "treasure_count": (8, 10),
                "buff_count": (1, 2),
                "monster_interval": (300, 500),
                "monster_speedup": (500, 800),
                "max_step": 800,
            },
            "hard": {
                "treasure_count": (7, 10),
                "buff_count": (1, 2),
                "monster_interval": (260, 340),
                "monster_speedup": (450, 550),
                "max_step": 1000,
            },
            "expert": {
                "treasure_count": (6, 10),
                "buff_count": (0, 2),
                "monster_interval": (120, 320),
                "monster_speedup": (140, 420),
                "buff_cooldown": (100, 300),
                "max_step": 1000,
            },
        }

    def current_stage_name(self):
        return self.stage_names[self.stage_idx]

    def note_train_episode_end(self):
        self.train_episodes_in_stage += 1

    def _sample_or_mid(self, value, is_val):
        if isinstance(value, tuple) and len(value) == 2:
            low, high = int(value[0]), int(value[1])
            if low > high:
                low, high = high, low
            if is_val:
                return int((low + high) // 2)
            return random.randint(low, high)
        return int(value)

    def build_env_overrides(self, is_val):
        stage_name = self.current_stage_name()
        cfg = self.stage_overrides[stage_name]
        result = {}
        for k, v in cfg.items():
            result[k] = self._sample_or_mid(v, is_val=is_val)
        return result

    def _promotion_condition(self, stage_name, val_metrics):
        if stage_name == "easy":
            return (
                float(val_metrics.get("completed_rate", 0.0))
                >= float(getattr(Config, "CURRICULUM_EASY_PROMOTE_COMPLETED_RATE", 0.30))
            )

        if stage_name == "medium":
            return (
                float(val_metrics.get("steps", 0.0))
                >= float(getattr(Config, "CURRICULUM_MEDIUM_PROMOTE_STEPS", 400.0))
                and float(val_metrics.get("treasures", 0.0))
                >= float(getattr(Config, "CURRICULUM_MEDIUM_PROMOTE_TREASURES", 0.5))
            )

        if stage_name == "hard":
            return (
                float(val_metrics.get("total_score", 0.0))
                >= float(getattr(Config, "CURRICULUM_HARD_PROMOTE_TOTAL_SCORE", 600.0))
                and float(val_metrics.get("terminated_rate", 1.0))
                <= float(getattr(Config, "CURRICULUM_HARD_PROMOTE_TERMINATED_RATE", 0.5))
            )

        return False

    def try_promote(self, val_metrics):
        if self.stage_idx >= len(self.stage_names) - 1:
            return False, "already_final_stage"

        if self.train_episodes_in_stage < self.warmup_episodes:
            return False, (
                f"warmup {self.train_episodes_in_stage}/{self.warmup_episodes}"
            )

        stage_name = self.current_stage_name()
        cond_ok = self._promotion_condition(stage_name, val_metrics)
        if cond_ok:
            self.promotion_streak += 1
        else:
            self.promotion_streak = 0

        if self.promotion_streak < self.promotion_window:
            return False, (
                f"streak {self.promotion_streak}/{self.promotion_window}"
            )

        prev_stage = stage_name
        self.stage_idx += 1
        self.train_episodes_in_stage = 0
        self.promotion_streak = 0
        return True, f"{prev_stage}->{self.current_stage_name()}"


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
        self.enable_bounded_random = bool(
            getattr(Config, "ENABLE_BOUNDED_DOMAIN_RANDOMIZATION", True)
        )
        self.monster_interval_jitter = max(
            int(getattr(Config, "MONSTER_INTERVAL_JITTER", 50)), 0
        )
        self.monster_speedup_jitter = max(
            int(getattr(Config, "MONSTER_SPEEDUP_JITTER", 50)), 0
        )
        self.enable_treasure_count_randomization = bool(
            getattr(Config, "ENABLE_TREASURE_COUNT_RANDOMIZATION", True)
        )
        self.treasure_count_min = int(
            getattr(Config, "TREASURE_COUNT_MIN", 0)
        )
        self.treasure_count_max = int(
            getattr(Config, "TREASURE_COUNT_MAX", 10)
        )
        self.enable_map_pool_randomization = bool(
            getattr(Config, "ENABLE_MAP_POOL_RANDOMIZATION", True)
        )
        self.map_pool_size_min = int(getattr(Config, "MAP_POOL_SIZE_MIN", 1))
        self.map_pool_size_max = int(getattr(Config, "MAP_POOL_SIZE_MAX", 10))
        self.enable_buff_cooldown_randomization = bool(
            getattr(Config, "ENABLE_BUFF_COOLDOWN_RANDOMIZATION", True)
        )
        self.buff_cooldown_min = int(getattr(Config, "BUFF_COOLDOWN_MIN", 1))
        self.buff_cooldown_max = int(getattr(Config, "BUFF_COOLDOWN_MAX", 500))
        self.enable_monster_speed_randomization = bool(
            getattr(Config, "ENABLE_MONSTER_SPEED_RANDOMIZATION", True)
        )
        self.monster_speed_min = int(getattr(Config, "MONSTER_SPEED_MIN", 1))
        self.monster_speed_max = int(getattr(Config, "MONSTER_SPEED_MAX", 3))
        self._warned_monster_speed_missing = False

        self.enable_curriculum = bool(
            getattr(Config, "ENABLE_CURRICULUM_LEARNING", True)
        )
        self.curriculum_manager = CurriculumManager() if self.enable_curriculum else None
        self.val_eval_window = max(int(getattr(Config, "VAL_EVAL_WINDOW", 5)), 1)
        self.val_eval_buffer = []

        # 基于验证集的最佳模型跟踪
        self.best_val_total_score = float("-inf")
        self.best_val_steps = float("-inf")
        self.best_val_reward = float("-inf")
        self.best_val_episode = -1
        self.best_stage_name = None

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

    def _sample_bounded(self, base_value, jitter, low, high):
        """Sample an integer around base_value within [low, high].

        围绕基础值在合法区间内采样整数，用于受限域随机化。
        """
        base = int(base_value)
        if jitter <= 0:
            return max(low, min(high, base))
        left = max(low, base - jitter)
        right = min(high, base + jitter)
        if left > right:
            return max(low, min(high, base))
        return random.randint(left, right)

    def _build_episode_conf(self, is_val):
        """Build per-episode env config with optional bounded randomization.

        构造每局环境配置：验证局保持固定，训练局可启用小范围随机。
        """
        base_conf = self.val_usr_conf if is_val else self.train_usr_conf
        run_conf = copy.deepcopy(base_conf)
        env_conf = run_conf.setdefault("env_conf", {})

        # 课程学习优先：按阶段直接覆盖环境参数。
        if self.curriculum_manager is not None:
            stage_name = self.curriculum_manager.current_stage_name()
            overrides = self.curriculum_manager.build_env_overrides(is_val=is_val)
            env_conf.update(overrides)

            # 仅在 expert 阶段训练时叠加地图池随机化，增强泛化。
            if (not is_val) and stage_name == "expert":
                map_pool = list(env_conf.get("map", []))
                if self.enable_map_pool_randomization and map_pool:
                    max_pool = min(max(1, self.map_pool_size_max), len(map_pool))
                    min_pool = min(max(1, self.map_pool_size_min), max_pool)
                    sampled_pool_size = random.randint(min_pool, max_pool)
                    env_conf["map"] = random.sample(map_pool, sampled_pool_size)
                    env_conf["map_random"] = True

                if self.enable_monster_speed_randomization and "monster_speed" in env_conf:
                    spd_min = max(1, self.monster_speed_min)
                    spd_max = max(1, self.monster_speed_max)
                    if spd_min > spd_max:
                        spd_min, spd_max = spd_max, spd_min
                    env_conf["monster_speed"] = random.randint(spd_min, spd_max)

            self.logger.info(
                f"[CURRICULUM {('VAL' if is_val else 'TRAIN')}] stage={stage_name} "
                f"treasure_count={env_conf.get('treasure_count', 'NA')} "
                f"buff_count={env_conf.get('buff_count', 'NA')} "
                f"monster_interval={env_conf.get('monster_interval', 'NA')} "
                f"monster_speedup={env_conf.get('monster_speedup', 'NA')} "
                f"max_step={env_conf.get('max_step', 'NA')}"
            )
            return run_conf

        if is_val or (not self.enable_bounded_random):
            return run_conf

        # 地图数量随机化：每局从当前 train maps 中随机采样一个子集
        map_pool = list(env_conf.get("map", []))
        if self.enable_map_pool_randomization and map_pool:
            max_pool = min(max(1, self.map_pool_size_max), len(map_pool))
            min_pool = min(max(1, self.map_pool_size_min), max_pool)
            sampled_pool_size = random.randint(min_pool, max_pool)
            env_conf["map"] = random.sample(map_pool, sampled_pool_size)
            env_conf["map_random"] = True

        # 宝箱数量随机化：控制在协议允许范围 [0, 10]
        sampled_treasure_count = int(env_conf.get("treasure_count", 10))
        if self.enable_treasure_count_randomization:
            trea_min = max(0, min(10, self.treasure_count_min))
            trea_max = max(0, min(10, self.treasure_count_max))
            if trea_min > trea_max:
                trea_min, trea_max = trea_max, trea_min
            sampled_treasure_count = random.randint(trea_min, trea_max)
            env_conf["treasure_count"] = sampled_treasure_count

        # buff 刷新时间随机化：协议范围 [1, 500]
        sampled_buff_cooldown = int(env_conf.get("buff_cooldown", 200))
        if self.enable_buff_cooldown_randomization:
            cool_min = max(1, min(500, self.buff_cooldown_min))
            cool_max = max(1, min(500, self.buff_cooldown_max))
            if cool_min > cool_max:
                cool_min, cool_max = cool_max, cool_min
            sampled_buff_cooldown = random.randint(cool_min, cool_max)
            env_conf["buff_cooldown"] = sampled_buff_cooldown

        base_interval = env_conf.get("monster_interval", 300)
        if int(base_interval) <= 0:
            base_interval = 300
        sampled_interval = self._sample_bounded(
            base_value=base_interval,
            jitter=self.monster_interval_jitter,
            low=11,
            high=2000,
        )

        base_speedup = env_conf.get("monster_speedup", 500)
        if int(base_speedup) <= 0:
            base_speedup = 500
        sampled_speedup = self._sample_bounded(
            base_value=base_speedup,
            jitter=self.monster_speedup_jitter,
            low=1,
            high=2000,
        )

        env_conf["monster_interval"] = sampled_interval
        env_conf["monster_speedup"] = sampled_speedup

        # 怪物移动速度随机化：仅当环境配置支持 monster_speed 字段时生效
        sampled_monster_speed = None
        if self.enable_monster_speed_randomization:
            if "monster_speed" in env_conf:
                spd_min = max(1, self.monster_speed_min)
                spd_max = max(1, self.monster_speed_max)
                if spd_min > spd_max:
                    spd_min, spd_max = spd_max, spd_min
                sampled_monster_speed = random.randint(spd_min, spd_max)
                env_conf["monster_speed"] = sampled_monster_speed
            elif not self._warned_monster_speed_missing:
                self.logger.info(
                    "[TRAIN randomization] monster_speed randomization enabled but env_conf has no monster_speed field; "
                    "fallback to monster_speedup randomization"
                )
                self._warned_monster_speed_missing = True

        sampled_map_cnt = len(env_conf.get("map", []))
        self.logger.info(
            f"[TRAIN randomization] map_cnt={sampled_map_cnt} treasure_count={sampled_treasure_count} "
            f"buff_cooldown={sampled_buff_cooldown} "
            f"monster_interval={sampled_interval} monster_speedup={sampled_speedup} "
            f"monster_speed={sampled_monster_speed if sampled_monster_speed is not None else 'NA'} "
            f"(base_interval={base_interval}, base_speedup={base_speedup})"
        )
        return run_conf

    def _aggregate_val_metrics(self, val_metrics):
        """Aggregate validation metrics with moving-average window.

        使用滑动窗口平均验证指标，降低单局随机方差影响。
        """
        tracked = {
            "reward": float(val_metrics.get("reward", 0.0)),
            "total_score": float(val_metrics.get("total_score", 0.0)),
            "steps": float(val_metrics.get("steps", 0.0)),
            "treasures": float(val_metrics.get("treasures", 0.0)),
            "terminated_rate": float(val_metrics.get("terminated_rate", 1.0)),
            "completed_rate": float(val_metrics.get("completed_rate", 0.0)),
        }
        self.val_eval_buffer.append(tracked)
        if len(self.val_eval_buffer) > self.val_eval_window:
            self.val_eval_buffer = self.val_eval_buffer[-self.val_eval_window:]

        if len(self.val_eval_buffer) < self.val_eval_window:
            self.logger.info(
                f"[VAL EVAL] warmup {len(self.val_eval_buffer)}/{self.val_eval_window}, skip aggregate"
            )
            return None

        avg = {}
        for k in tracked.keys():
            avg[k] = float(np.mean([m[k] for m in self.val_eval_buffer]))
        return avg

    def _try_save_best_val_model(self, val_metrics):
        """Save best-val checkpoint when validation metrics improve.

        当验证指标刷新历史最优时，保存 best_val 检查点。
        """
        stage_name = (
            self.curriculum_manager.current_stage_name()
            if self.curriculum_manager is not None
            else "none"
        )

        # 切换阶段时重置 best 比较基准，避免跨阶段比较失真。
        if self.best_stage_name != stage_name:
            self.best_stage_name = stage_name
            self.best_val_total_score = float("-inf")
            self.best_val_steps = float("-inf")
            self.best_val_reward = float("-inf")
            self.best_val_episode = -1

        val_completed_rate = float(val_metrics.get("completed_rate", 0.0))
        val_terminated_rate = float(val_metrics.get("terminated_rate", 1.0))
        val_treasures = float(val_metrics.get("treasures", 0.0))
        val_total_score = float(val_metrics.get("total_score", 0.0))
        val_steps = float(val_metrics.get("steps", 0.0))
        val_reward = float(val_metrics.get("reward", 0.0))

        if stage_name == "easy":
            candidate = (val_completed_rate, val_steps, val_reward)
            baseline = (self.best_val_total_score, self.best_val_steps, self.best_val_reward)
        elif stage_name == "medium":
            candidate = (val_steps, val_treasures, val_reward)
            baseline = (self.best_val_total_score, self.best_val_steps, self.best_val_reward)
        else:
            candidate = (val_total_score, -val_terminated_rate, val_reward)
            baseline = (self.best_val_total_score, self.best_val_steps, self.best_val_reward)

        improved = candidate > baseline
        if not improved:
            return

        self.best_val_total_score = candidate[0]
        self.best_val_steps = candidate[1]
        self.best_val_reward = candidate[2]
        self.best_val_episode = self.episode_cnt

        try:
            # 保留稳定别名，便于随时加载当前最优。
            self.agent.save_model(id="best_val")
            self.agent.save_model(id="best_model")  # 保存一个最佳前缀供长线调用
            
            # 为了满足业务平台提取 best_model.zip 的需求，我们可以将它额外打一个zip
            import zipfile
            ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ckpt")
            best_model_pkl = os.path.join(ckpt_dir, "best_model.pkl")
            # Navigate to kaiwu/train/backup_model
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            backup_dir = os.path.join(root_dir, "train", "backup_model")
            os.makedirs(backup_dir, exist_ok=True)
            if os.path.exists(best_model_pkl):
                zip_path = os.path.join(backup_dir, "best_model.zip")
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    zf.write(best_model_pkl, arcname="best_model.pkl")
                    
            self.logger.info(
                f"[VAL BEST] episode:{self.episode_cnt} stage:{stage_name} window:{self.val_eval_window} "
                f"avg_score:{val_total_score:.1f} avg_steps:{val_steps:.1f} "
                f"avg_completed:{val_completed_rate:.3f} avg_terminated:{val_terminated_rate:.3f} "
                f"avg_reward:{val_reward:.3f} saved_id:best_val and best_model.zip"
            )
        except (OSError, RuntimeError, ValueError) as err:
            self.logger.error(f"[VAL BEST] save failed: {err}")

    def _try_promote_curriculum(self, avg_val_metrics):
        """Try promote curriculum stage from averaged validation metrics."""
        if self.curriculum_manager is None:
            return

        promoted, detail = self.curriculum_manager.try_promote(avg_val_metrics)
        if not promoted:
            return

        # 切阶段后清空验证窗口，避免不同阶段分布混合。
        self.val_eval_buffer = []

        self.logger.info(
            f"[CURRICULUM PROMOTE] episode:{self.episode_cnt} detail:{detail} "
            f"avg_completed_rate:{avg_val_metrics.get('completed_rate', 0.0):.3f} "
            f"avg_steps:{avg_val_metrics.get('steps', 0.0):.1f} "
            f"avg_treasures:{avg_val_metrics.get('treasures', 0.0):.3f} "
            f"avg_score:{avg_val_metrics.get('total_score', 0.0):.1f}"
        )

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
            stage_name = (
                self.curriculum_manager.current_stage_name()
                if self.curriculum_manager is not None
                else "none"
            )
            run_conf = self._build_episode_conf(is_val)
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
            if hasattr(self.agent, "preprocessor") and hasattr(self.agent.preprocessor, "set_curriculum_stage"):
                self.agent.preprocessor.set_curriculum_stage(stage_name)
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
                f"[{mode_str}] total_episode:{self.episode_cnt} train_episode:{self.train_episode_cnt} "
                f"val_episode:{self.val_episode_cnt} stage:{stage_name} start"
            )

            while not done:
                # Predict action / Agent 推理（随机采样）
                pred_list = self.agent.predict(list_obs_data=[obs_data])
                if not pred_list:
                    self.logger.error(
                        f"[{mode_str}] predict returned empty result, break current episode"
                    )
                    break
                act_data = pred_list[0]
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

                    # Structured model attention diagnostics:
                    # [monster1, monster2, resource, map]
                    attn_mon1 = -1.0
                    attn_mon2 = -1.0
                    attn_resource = -1.0
                    attn_map = -1.0
                    attn_weights = getattr(
                        getattr(self.agent, "model", None),
                        "_last_attn_weights",
                        None,
                    )
                    if attn_weights is not None:
                        try:
                            attn_np = attn_weights.detach().cpu().numpy()
                            if attn_np.ndim == 2 and attn_np.shape[1] == 4:
                                attn_mean = attn_np.mean(axis=0)
                                attn_mon1 = float(attn_mean[0])
                                attn_mon2 = float(attn_mean[1])
                                attn_resource = float(attn_mean[2])
                                attn_map = float(attn_mean[3])
                        except (TypeError, ValueError, RuntimeError):
                            pass

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
                        "attn_mon1": round(attn_mon1, 4),
                        "attn_mon2": round(attn_mon2, 4),
                        "attn_resource": round(attn_resource, 4),
                        "attn_map": round(attn_map, 4),
                        "curriculum_stage_idx": float(
                            self.curriculum_manager.stage_idx
                            if self.curriculum_manager is not None
                            else -1
                        ),
                        "curriculum_train_episodes_in_stage": float(
                            self.curriculum_manager.train_episodes_in_stage
                            if self.curriculum_manager is not None
                            else -1
                        ),
                    }

                    if (not is_val) and self.curriculum_manager is not None:
                        self.curriculum_manager.note_train_episode_end()

                    # Monitor report / 监控上报
                    if self.monitor:
                        self.monitor.put_data(
                            {os.getpid(): self._prefix_metrics(
                                episode_metrics, mode_prefix)}
                        )

                    # 验证局使用滑动均值评估，并驱动课程晋级。
                    if is_val:
                        avg_val_metrics = self._aggregate_val_metrics(episode_metrics)
                        if avg_val_metrics is not None:
                            self._try_save_best_val_model(avg_val_metrics)
                            self._try_promote_curriculum(avg_val_metrics)

                    if collector and (not is_val):
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
