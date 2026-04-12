#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np
from agent_ppo.conf.conf import Config

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Map diagonal distance / 地图对角线长度
MAP_DIAG = MAP_SIZE * 1.41
# Max monster speedup step / 怪物加速步数上限
MAX_MONSTER_SPEEDUP_STEP = 2000.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        # 先声明状态字段，避免静态检查误报
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.prev_monster_raw_dists = [None, None]
        self.prev_step_score = None
        self.prev_treasure_score = None
        self.prev_min_monster_raw_dist = None
        self.prev_nearest_treasure_raw_dist = None
        self.prev_hero_pos = None
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        # 记录上一帧两只怪距离，用于动态特征
        self.prev_monster_raw_dists = [None, None]
        self.prev_step_score = None
        self.prev_treasure_score = None
        self.prev_min_monster_raw_dist = None
        self.prev_nearest_treasure_raw_dist = None
        self.prev_hero_pos = None

    def _as_float(self, v, default=0.0):
        """Safe float conversion.

        安全转换为浮点数。
        """
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _normalize_entity_container(self, data):
        """Normalize different container shapes to list[dict].

        兼容 list / dict(list) / dict-of-dict / single-dict 等实体容器。
        """
        if isinstance(data, list):
            return data

        if isinstance(data, tuple):
            return list(data)

        if isinstance(data, dict):
            # common nested list keys
            for sub_key in ("items", "list", "data", "entities", "objects", "infos"):
                sub_data = data.get(sub_key)
                if isinstance(sub_data, list):
                    return sub_data

            # dict of entities: {id: {...}, id2: {...}}
            dict_values = [v for v in data.values() if isinstance(v, dict)]
            if dict_values and len(dict_values) >= max(1, len(data) // 2):
                return dict_values

            # single entity dict
            if "pos" in data or ("x" in data and ("z" in data or "y" in data)):
                return [data]

        return None

    def _get_entity_list(self, frame_state, candidate_keys):
        """Get first existing entity list by keys.

        按候选键依次读取实体列表。
        """
        for key in candidate_keys:
            data = frame_state.get(key)
            entities = self._normalize_entity_container(data)
            if entities is not None:
                return entities, key
        return [], "none"

    def _split_organs_targets(self, frame_state):
        """Split organs to treasure/buff lists by protocol sub_type.

        按协议从 organs 中拆分目标：sub_type=1 为宝箱，sub_type=2 为加速 buff。
        """
        organs = self._normalize_entity_container(frame_state.get("organs"))
        if organs is None:
            return [], [], 0

        treasures = []
        buffs = []
        for item in organs:
            if not isinstance(item, dict):
                continue

            # 协议: status=1 表示可获取
            if self._as_float(item.get("status", 1), 1) <= 0:
                continue

            sub_type = int(self._as_float(item.get("sub_type", -1), -1))
            if sub_type == 1:
                treasures.append(item)
            elif sub_type == 2:
                buffs.append(item)

        return treasures, buffs, len(organs)

    def _is_target_available(self, entity):
        """Unified availability check for target entities.

        统一目标可用性判定，兼容 is_in_view/is_valid/status 三类字段。
        """
        in_view = self._as_float(entity.get("is_in_view", 1), 1) > 0
        # organs 使用 status=1 表示可获取；旧字段保持兼容
        is_valid = self._as_float(
            entity.get("is_valid", entity.get("status", 1)),
            1,
        ) > 0
        status = self._as_float(entity.get("status", 1), 1) > 0
        return in_view and is_valid and status

    def _extract_pos(self, entity):
        """Extract (x, z) from entity dict.

        从实体中提取坐标（兼容 pos 或扁平字段）。
        """
        if not isinstance(entity, dict):
            return None

        pos = entity.get("pos")
        if isinstance(pos, dict):
            if "x" in pos and ("z" in pos or "y" in pos):
                return float(pos.get("x", 0.0)), float(pos.get("z", pos.get("y", 0.0)))

        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return float(pos[0]), float(pos[1])

        if "x" in entity and ("z" in entity or "y" in entity):
            return float(entity.get("x", 0.0)), float(entity.get("z", entity.get("y", 0.0)))

        return None

    def _nearest_target_feature(self, hero_pos, entities):
        """Build nearest target feature: [exist, dist_norm, dir_cos, dir_sin].

        计算最近目标特征：[是否存在, 距离归一化, 方向cos, 方向sin]。
        """
        best_dist = None
        best_dx = 0.0
        best_dz = 0.0

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            if not self._is_target_available(ent):
                continue

            pos = self._extract_pos(ent)
            if pos is None:
                continue

            dx = float(pos[0] - hero_pos["x"])
            dz = float(pos[1] - hero_pos["z"])
            dist = float(np.sqrt(dx * dx + dz * dz))

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dx = dx
                best_dz = dz

        if best_dist is None:
            return np.zeros(4, dtype=np.float32), None

        inv = 1.0 / (best_dist + 1e-6)
        dir_cos = float(np.clip(best_dx * inv, -1.0, 1.0))
        dir_sin = float(np.clip(best_dz * inv, -1.0, 1.0))
        return np.array([1.0, _norm(best_dist, MAP_DIAG), dir_cos, dir_sin], dtype=np.float32), float(best_dist)

    def _parse_monster_speedup_step(self, env_info):
        """Parse monster speedup step from env_info with fallback.

        解析怪物加速步数，缺失时回退默认值。
        """
        for key in ("monster_speedup", "monster_speedup_step", "monster_accelerate_step"):
            if key in env_info:
                try:
                    value = float(env_info[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass
        return 500.0

    def feature_process(self, env_obs, _last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(
            hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array(
            [hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        current_monster_raw_dists = [None, None]
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt(
                        (hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_DIAG)
                    current_monster_raw_dists[i] = float(raw_dist)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm,
                             m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # 新增特征1：按距离排序的最近怪动态特征
        visible_slots = [(idx, dist) for idx, dist in enumerate(
            current_monster_raw_dists) if dist is not None]
        visible_slots.sort(key=lambda item: item[1])
        nearest_slot = visible_slots[0][0] if visible_slots else None
        second_slot = visible_slots[1][0] if len(visible_slots) > 1 else None

        def _monster_dyn(slot_idx):
            if slot_idx is None:
                return None, 0.0, 1.0

            curr_dist = current_monster_raw_dists[slot_idx]
            prev_dist = self.prev_monster_raw_dists[slot_idx]

            dist_delta = 0.0
            ttc_norm = 1.0
            if prev_dist is not None and curr_dist is not None:
                dist_delta = float(
                    np.clip((prev_dist - curr_dist) / MAP_DIAG, -1.0, 1.0))
                closing_speed = max(prev_dist - curr_dist, 0.0)
                if closing_speed > 1e-6:
                    ttc_norm = _norm(curr_dist / closing_speed, MAP_DIAG)

            return curr_dist, dist_delta, ttc_norm

        nearest_curr_dist, nearest_dist_delta, nearest_ttc_norm = _monster_dyn(
            nearest_slot)
        nearest_danger_flag = float(
            nearest_curr_dist is not None and nearest_curr_dist <= Config.DANGER_DIST_THRESHOLD)
        nearest_monster_dyn_feat = np.array(
            [
                float(nearest_slot == 0) if nearest_slot is not None else 0.0,
                nearest_dist_delta,
                nearest_ttc_norm,
                nearest_danger_flag,
            ],
            dtype=np.float32,
        )

        # 新增特征2：最近第二只怪动态特征（若不存在则 second_exists=0）
        second_curr_dist, second_dist_delta, second_ttc_norm = _monster_dyn(
            second_slot)
        double_pressure_flag = float(
            nearest_curr_dist is not None
            and second_curr_dist is not None
            and nearest_curr_dist <= Config.DOUBLE_PRESSURE_DIST_THRESHOLD
            and second_curr_dist <= Config.DOUBLE_PRESSURE_DIST_THRESHOLD
        )
        second_monster_dyn_feat = np.array(
            [
                float(second_slot is not None),
                second_dist_delta,
                second_ttc_norm,
                double_pressure_flag,
            ],
            dtype=np.float32,
        )

        # Local map features (16D) / 局部地图特征
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # Legal action mask (16D) / 合法动作掩码
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                legal_action = [0] * 16
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # 新增特征3：最近宝箱目标特征
        # 优先使用协议定义的 organs(sub_type=1)，再回退旧字段名兼容。
        organs_treasures, organs_buffs, _ = self._split_organs_targets(
            frame_state)
        if organs_treasures:
            treasures = organs_treasures
        else:
            treasures, _ = self._get_entity_list(
                frame_state,
                [
                    "treasures",
                    "treasure",
                    "treasure_list",
                    "chests",
                    "chest",
                    "chest_list",
                    "boxes",
                    "box",
                ],
            )
        nearest_treasure_feat, nearest_treasure_raw_dist = self._nearest_target_feature(
            hero_pos, treasures)

        # 新增特征4：最近buff目标特征
        # 优先使用协议定义的 organs(sub_type=2)，再回退旧字段名兼容。
        if organs_buffs:
            buffs = organs_buffs
        else:
            buffs, _ = self._get_entity_list(
                frame_state,
                [
                    "buffs",
                    "buff",
                    "buff_list",
                    "speed_buffs",
                    "speed_buff",
                    "speedup_buffs",
                    "speedup_buff",
                    "speedup",
                ],
            )
        nearest_buff_feat, _ = self._nearest_target_feature(hero_pos, buffs)

        # 新增特征5：阶段提示特征（怪物加速前后）
        monster_speedup_step = self._parse_monster_speedup_step(env_info)
        monster_speedup_by_speed = any(
            float(m.get("speed", 1)) >= 2.0 for m in monsters if isinstance(m, dict)
        )
        post_speedup_flag = float(
            monster_speedup_by_speed or self.step_no >= monster_speedup_step)
        speedup_eta = max(monster_speedup_step - self.step_no, 0.0)
        phase_feat = np.array(
            [post_speedup_flag, _norm(speedup_eta, MAX_MONSTER_SPEEDUP_STEP)],
            dtype=np.float32,
        )

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                nearest_monster_dyn_feat,
                second_monster_dyn_feat,
                nearest_treasure_feat,
                nearest_buff_feat,
                phase_feat,
            ]
        )

        # Step reward / 即时奖励（前后期分治 + 直接绑定得分增量）
        step_score = float(env_info.get("step_score", 0.0))
        treasure_score = float(env_info.get("treasure_score", 0.0))

        # 当前最小怪物距离（原始尺度）
        valid_dists = [d for d in current_monster_raw_dists if d is not None]
        cur_min_monster_raw_dist = min(
            valid_dists) if valid_dists else MAP_DIAG
        prev_min_monster_raw_dist = (
            self.prev_min_monster_raw_dist
            if self.prev_min_monster_raw_dist is not None
            else cur_min_monster_raw_dist
        )

        # 当前阶段：前期偏拿资源，后期偏保命
        is_post = bool(post_speedup_flag > 0.5)
        w_step = Config.SCORE_STEP_WEIGHT_POST if is_post else Config.SCORE_STEP_WEIGHT_PRE
        w_treasure = Config.SCORE_TREASURE_WEIGHT_POST if is_post else Config.SCORE_TREASURE_WEIGHT_PRE
        w_survive = Config.SURVIVE_REWARD_POST if is_post else Config.SURVIVE_REWARD_PRE
        w_dist = Config.DIST_SHAPING_WEIGHT_POST if is_post else Config.DIST_SHAPING_WEIGHT_PRE
        w_treasure_approach = (
            Config.TREASURE_APPROACH_WEIGHT_POST if is_post else Config.TREASURE_APPROACH_WEIGHT_PRE
        )

        # 1) 与官方评分直接绑定：步数分增量 + 宝箱分增量
        prev_step_score = self.prev_step_score if self.prev_step_score is not None else step_score
        prev_treasure_score = (
            self.prev_treasure_score if self.prev_treasure_score is not None else treasure_score
        )
        delta_step_score = max(step_score - prev_step_score, 0.0)
        delta_treasure_score = max(treasure_score - prev_treasure_score, 0.0)
        step_gain = delta_step_score / Config.STEP_SCORE_UNIT
        treasure_gain = delta_treasure_score / Config.TREASURE_SCORE_UNIT
        score_reward = w_step * step_gain + w_treasure * treasure_gain

        # 2) 基础生存奖励
        survive_reward = w_survive

        # 3) 安全塑形：远离怪物给正反馈，接近给负反馈
        dist_delta = np.clip(
            cur_min_monster_raw_dist - prev_min_monster_raw_dist,
            -Config.DIST_DELTA_CLIP,
            Config.DIST_DELTA_CLIP,
        )
        dist_shaping = w_dist * float(dist_delta)

        # 4) 资源塑形：仅在可见宝箱时计算接近趋势
        treasure_approach = 0.0
        prev_treasure_raw_dist = self.prev_nearest_treasure_raw_dist
        if nearest_treasure_raw_dist is not None and prev_treasure_raw_dist is not None:
            treasure_delta = np.clip(
                prev_treasure_raw_dist - nearest_treasure_raw_dist,
                -Config.TREASURE_APPROACH_DELTA_CLIP,
                Config.TREASURE_APPROACH_DELTA_CLIP,
            )
            treasure_approach = w_treasure_approach * float(treasure_delta)

        # 5) 风险惩罚：后期更重
        is_high_danger = float(cur_min_monster_raw_dist <=
                               Config.HIGH_DANGER_DIST_THRESHOLD)
        danger_penalty = -(
            Config.HIGH_DANGER_PENALTY_POST if is_post else Config.HIGH_DANGER_PENALTY_PRE
        ) * is_high_danger

        double_pressure = float(
            nearest_curr_dist is not None
            and second_curr_dist is not None
            and nearest_curr_dist <= Config.DOUBLE_PRESSURE_DIST_THRESHOLD
            and second_curr_dist <= Config.DOUBLE_PRESSURE_DIST_THRESHOLD
        )
        double_pressure_penalty = -Config.DOUBLE_PRESSURE_PENALTY_POST * \
            double_pressure if is_post else 0.0

        # 6) 闪现价值奖励：奖励“用得值”，惩罚无效交闪
        flash_reward = 0.0
        flash_penalty = 0.0
        if _last_action is not None and int(_last_action) >= 8:
            prev_danger = prev_min_monster_raw_dist <= Config.DANGER_DIST_THRESHOLD
            danger_improve = cur_min_monster_raw_dist - prev_min_monster_raw_dist
            escaped = prev_danger and danger_improve >= Config.FLASH_ESCAPE_MIN_GAIN
            if escaped:
                flash_reward += (
                    Config.FLASH_ESCAPE_REWARD_POST if is_post else Config.FLASH_ESCAPE_REWARD_PRE
                )
            if treasure_gain > 0.0:
                flash_reward += Config.FLASH_TREASURE_GAIN_REWARD
            if (not escaped) and step_gain <= 0.0 and treasure_gain <= 0.0:
                flash_penalty -= Config.FLASH_WASTE_PENALTY

        # 7) 无效移动惩罚：减少原地抖动/撞墙行为
        invalid_move_penalty = 0.0
        if _last_action is not None and 0 <= int(_last_action) < 8 and self.prev_hero_pos is not None:
            dx = float(hero_pos["x"] - self.prev_hero_pos[0])
            dz = float(hero_pos["z"] - self.prev_hero_pos[1])
            moved = float(np.sqrt(dx * dx + dz * dz))
            if moved < 0.1:
                invalid_move_penalty = -Config.INVALID_MOVE_PENALTY

        total_reward = (
            score_reward
            + survive_reward
            + dist_shaping
            + treasure_approach
            + danger_penalty
            + double_pressure_penalty
            + flash_reward
            + flash_penalty
            + invalid_move_penalty
        )
        total_reward = float(
            np.clip(total_reward, Config.REWARD_CLIP_MIN, Config.REWARD_CLIP_MAX))

        reward_info = {
            # 核心指标：用于监控 reward 是否对齐任务得分。
            "score_reward": float(score_reward),
            "step_gain": float(step_gain),
            "treasure_gain": float(treasure_gain),
            "survive_reward": float(survive_reward),
            "dist_shaping": float(dist_shaping),
            "treasure_approach": float(treasure_approach),
            "danger_penalty": float(danger_penalty),
            "double_pressure_penalty": float(double_pressure_penalty),
            "flash_reward": float(flash_reward),
            "flash_penalty": float(flash_penalty),
            "invalid_move_penalty": float(invalid_move_penalty),
            "is_post": float(is_post),
            "min_monster_dist": float(cur_min_monster_raw_dist),
            "nearest_treasure_dist": float(nearest_treasure_raw_dist)
            if nearest_treasure_raw_dist is not None
            else -1.0,
            "total_reward": float(total_reward),
        }

        # 更新缓存，供下一帧增量奖励计算。
        self.last_min_monster_dist_norm = _norm(
            cur_min_monster_raw_dist, MAP_DIAG)
        self.prev_monster_raw_dists = current_monster_raw_dists
        self.prev_step_score = step_score
        self.prev_treasure_score = treasure_score
        self.prev_min_monster_raw_dist = cur_min_monster_raw_dist
        self.prev_nearest_treasure_raw_dist = nearest_treasure_raw_dist
        self.prev_hero_pos = (float(hero_pos["x"]), float(hero_pos["z"]))

        reward = [total_reward]
        return feature, legal_action, reward, reward_info
