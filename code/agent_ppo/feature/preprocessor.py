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
# Danger threshold / 危险距离阈值
HIGH_DANGER_DIST = 3.0
# Double-pressure threshold / 双怪压迫阈值
DOUBLE_PRESSURE_DIST = 8.0


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
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        # 记录上一帧两只怪距离，用于动态特征
        self.prev_monster_raw_dists = [None, None]

    def _get_entity_list(self, frame_state, candidate_keys):
        """Get first existing entity list by keys.

        按候选键依次读取实体列表。
        """
        for key in candidate_keys:
            data = frame_state.get(key)
            if isinstance(data, list):
                return data
        return []

    def _extract_pos(self, entity):
        """Extract (x, z) from entity dict.

        从实体中提取坐标（兼容 pos 或扁平字段）。
        """
        if not isinstance(entity, dict):
            return None

        pos = entity.get("pos")
        if isinstance(pos, dict):
            if "x" in pos and "z" in pos:
                return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))

        if "x" in entity and "z" in entity:
            return float(entity.get("x", 0.0)), float(entity.get("z", 0.0))

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
            if float(ent.get("is_in_view", 1)) <= 0:
                continue
            if float(ent.get("is_valid", 1)) <= 0:
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
            return np.zeros(4, dtype=np.float32)

        inv = 1.0 / (best_dist + 1e-6)
        dir_cos = float(np.clip(best_dx * inv, -1.0, 1.0))
        dir_sin = float(np.clip(best_dz * inv, -1.0, 1.0))
        return np.array([1.0, _norm(best_dist, MAP_DIAG), dir_cos, dir_sin], dtype=np.float32)

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
            nearest_curr_dist is not None and nearest_curr_dist <= HIGH_DANGER_DIST)
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
            and nearest_curr_dist <= DOUBLE_PRESSURE_DIST
            and second_curr_dist <= DOUBLE_PRESSURE_DIST
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

        # 新增特征3：最近宝箱目标特征（字段名不一致时自动兜底）
        treasures = self._get_entity_list(
            frame_state, ["treasures", "chests", "boxes", "treasure"])
        nearest_treasure_feat = self._nearest_target_feature(
            hero_pos, treasures)

        # 新增特征4：最近buff目标特征（字段名不一致时自动兜底）
        buffs = self._get_entity_list(
            frame_state, ["buffs", "speed_buffs", "speedup_buffs"])
        nearest_buff_feat = self._nearest_target_feature(hero_pos, buffs)

        # 新增特征5：阶段提示特征（怪物加速前后）
        monster_speedup_step = self._parse_monster_speedup_step(env_info)
        post_speedup_flag = float(self.step_no >= monster_speedup_step)
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

        # Step reward / 即时奖励
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        survive_reward = 0.01
        dist_shaping = 0.1 * (cur_min_dist_norm -
                              self.last_min_monster_dist_norm)

        self.last_min_monster_dist_norm = cur_min_dist_norm
        # 更新上一帧怪物距离，供下一帧计算动态特征
        self.prev_monster_raw_dists = current_monster_raw_dists

        reward = [survive_reward + dist_shaping]

        return feature, legal_action, reward
