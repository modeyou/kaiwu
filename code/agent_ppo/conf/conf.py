#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度（共66维）
    FEATURES = [
        4,   # hero_self
        5,   # monster_1
        5,   # monster_2
        16,  # map_local
        16,  # legal_action（16维动作掩码）
        2,   # progress
        4,   # nearest_monster_dyn
        4,   # second_nearest_monster_dyn
        4,   # nearest_treasure_target
        4,   # nearest_buff_target
        2,   # phase_hint
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8个移动方向+8个闪现
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    # Reward design / 奖励设计（前后期分治）
    STEP_SCORE_UNIT = 1.5
    TREASURE_SCORE_UNIT = 100.0

    # 核心得分增量权重：前期偏拿资源，后期偏保命步数
    SCORE_STEP_WEIGHT_PRE = 0.03
    SCORE_STEP_WEIGHT_POST = 0.05
    SCORE_TREASURE_WEIGHT_PRE = 1.0
    SCORE_TREASURE_WEIGHT_POST = 0.7

    # 基础生存奖励
    SURVIVE_REWARD_PRE = 0.006
    SURVIVE_REWARD_POST = 0.010

    # 怪物距离塑形（后期更重）
    DIST_SHAPING_WEIGHT_PRE = 0.05
    DIST_SHAPING_WEIGHT_POST = 0.12
    DIST_DELTA_CLIP = 2.0

    # 宝箱接近塑形（后期弱化，避免贪箱送死）
    TREASURE_APPROACH_WEIGHT_PRE = 0.06
    TREASURE_APPROACH_WEIGHT_POST = 0.02
    TREASURE_APPROACH_DELTA_CLIP = 2.0

    # 风险阈值与惩罚
    DANGER_DIST_THRESHOLD = 3.0
    HIGH_DANGER_DIST_THRESHOLD = 1.5
    DOUBLE_PRESSURE_DIST_THRESHOLD = 8.0
    HIGH_DANGER_PENALTY_PRE = 0.12
    HIGH_DANGER_PENALTY_POST = 0.28
    DOUBLE_PRESSURE_PENALTY_POST = 0.18

    # 闪现价值奖励（奖励“用得值”）
    FLASH_ESCAPE_MIN_GAIN = 1.5
    FLASH_ESCAPE_REWARD_PRE = 0.35
    FLASH_ESCAPE_REWARD_POST = 0.42
    FLASH_TREASURE_GAIN_REWARD = 0.25
    FLASH_WASTE_PENALTY = 0.15

    # 无效位移惩罚（轻惩罚，防止撞墙抖动）
    INVALID_MOVE_PENALTY = 0.04

    # 终局奖励（替代原始 ±10，避免掩盖过程目标）
    TERMINAL_FAIL_PRE = -2.0
    TERMINAL_FAIL_POST = -3.0
    TERMINAL_COMPLETE = 1.2
    TERMINAL_ABNORMAL = -1.0

    # 单步奖励裁剪，增强训练稳定性
    REWARD_CLIP_MIN = -3.0
    REWARD_CLIP_MAX = 3.0

    # 训练/验证分离：每 N 局训练插入 1 局验证
    TRAIN_VAL_INTERVAL = 10
