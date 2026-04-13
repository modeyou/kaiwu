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

    # Feature dimensions / 特征维度（共147维）
    FEATURES = [
        4,   # hero_self
        13,  # monster_1
        13,  # monster_2
        81,  # map_local（9x9）
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

    # Feature slice indices / 特征切片索引（左闭右开）
    IDX_HERO = (0, 4)
    IDX_MON1 = (4, 17)
    IDX_MON2 = (17, 30)
    IDX_MAP = (30, 111)
    IDX_LEGAL = (111, 127)
    IDX_PROGRESS = (127, 129)
    IDX_DYN1 = (129, 133)
    IDX_DYN2 = (133, 137)
    IDX_TREA = (137, 141)
    IDX_BUFF = (141, 145)
    IDX_PHASE = (145, 147)

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
    SURVIVE_REWARD_PRE = 0.03
    SURVIVE_REWARD_POST = 0.05

    # 怪物距离塑形（后期更重）
    DIST_SHAPING_WEIGHT_PRE = 0.15
    DIST_SHAPING_WEIGHT_POST = 0.2
    DIST_DELTA_CLIP = 2.0

    # 宝箱接近塑形（后期弱化，避免贪箱送死）
    TREASURE_APPROACH_WEIGHT_PRE = 0.08
    TREASURE_APPROACH_WEIGHT_POST = 0.04
    TREASURE_APPROACH_DELTA_CLIP = 2.0

    # 风险阈值与惩罚
    DANGER_DIST_THRESHOLD = 3.0
    HIGH_DANGER_DIST_THRESHOLD = 1.5
    DOUBLE_PRESSURE_DIST_THRESHOLD = 8.0
    HIGH_DANGER_PENALTY_PRE = 0.01
    HIGH_DANGER_PENALTY_POST = 0.05
    DOUBLE_PRESSURE_PENALTY_POST = 0.18

    # 闪现价值奖励（奖励“用得值”）
    FLASH_ESCAPE_MIN_GAIN = 1.5
    FLASH_ESCAPE_REWARD_PRE = 0.8
    FLASH_ESCAPE_REWARD_POST = 1.0
    FLASH_TREASURE_GAIN_REWARD = 0.4
    FLASH_WASTE_PENALTY = 0.15

    # 无效位移惩罚（轻惩罚，防止撞墙抖动）
    INVALID_MOVE_PENALTY = 0.01

    # 终局奖励（替代原始 ±10，避免掩盖过程目标）
    TERMINAL_FAIL_PRE = -1.0
    TERMINAL_FAIL_POST = -3.0
    TERMINAL_COMPLETE = 1.2
    TERMINAL_ABNORMAL = -1.0

    # 单步奖励裁剪，增强训练稳定性
    REWARD_CLIP_MIN = -3.0
    REWARD_CLIP_MAX = 3.0

    # 训练/验证分离：每 N 局训练插入 1 局验证
    TRAIN_VAL_INTERVAL = 10

    # 受限域随机化：围绕基础配置做小范围随机抖动（仅训练局启用）
    ENABLE_BOUNDED_DOMAIN_RANDOMIZATION = True
    MONSTER_INTERVAL_JITTER = 100
    MONSTER_SPEEDUP_JITTER = 50

    # 宝箱数量随机化（训练局）
    ENABLE_TREASURE_COUNT_RANDOMIZATION = False
    TREASURE_COUNT_MIN = 7
    TREASURE_COUNT_MAX = 10

    # 地图数量随机化（训练局）：每局从 train maps 里采样一个子集
    ENABLE_MAP_POOL_RANDOMIZATION = False
    MAP_POOL_SIZE_MIN = 6
    MAP_POOL_SIZE_MAX = 8

    # buff 刷新时间随机化（训练局）
    ENABLE_BUFF_COOLDOWN_RANDOMIZATION = False
    BUFF_COOLDOWN_MIN = 120
    BUFF_COOLDOWN_MAX = 260

    # 怪物移动速度随机化（训练局）
    # 仅当 env_conf 中存在 monster_speed 字段时生效；否则仅使用 monster_speedup 随机化。
    ENABLE_MONSTER_SPEED_RANDOMIZATION = False
    MONSTER_SPEED_MIN = 1
    MONSTER_SPEED_MAX = 3
