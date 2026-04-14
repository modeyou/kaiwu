#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.conf.conf import Config


def make_fc(in_dim, out_dim, gain=1.0):
    """Create a linear layer with orthogonal initialization."""
    fc = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(fc.weight, gain=gain)
    nn.init.zeros_(fc.bias)
    return fc


class ResidualBlock(nn.Module):
    """Residual MLP block to stabilize optimization."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            make_fc(dim, dim), nn.LayerNorm(dim), nn.ReLU(),
            make_fc(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class MapEncoder(nn.Module):
    """Encode 9x9 local map to a 32D embedding."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.proj = nn.Sequential(
            make_fc(64 * 3 * 3, 64), nn.ReLU(),
            make_fc(64, 32),
        )

    def forward(self, map_feat):
        batch = map_feat.size(0)
        x = map_feat.view(batch, 1, 9, 9)
        x = self.conv(x).view(batch, -1)
        return self.proj(x)


class EntityAttention(nn.Module):
    """Hero-query attention over [monster1, monster2, resource, map]."""

    def __init__(self, embed_dim=32):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.q_proj = make_fc(embed_dim, embed_dim)
        self.k_proj = make_fc(embed_dim, embed_dim)
        self.v_proj = make_fc(embed_dim, embed_dim)
        self.out_proj = make_fc(embed_dim, embed_dim)

    def forward(self, hero_enc, entity_list):
        entities = torch.stack(entity_list, dim=1)
        query = self.q_proj(hero_enc).unsqueeze(1)
        key = self.k_proj(entities)
        value = self.v_proj(entities)

        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, value).squeeze(1)
        return self.out_proj(context), weights.squeeze(1)


class Model(nn.Module):
    """Structured Actor-Critic with grouped encoders and attention fusion."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_structured"
        self.device = device

        embed = 32

        self.hero_encoder = nn.Sequential(
            make_fc(8, 64), nn.LayerNorm(64), nn.ReLU(),
            make_fc(64, embed), nn.ReLU(),
        )

        self.monster_encoder = nn.Sequential(
            make_fc(17, 64), nn.LayerNorm(64), nn.ReLU(),
            make_fc(64, embed), nn.ReLU(),
        )

        self.resource_encoder = nn.Sequential(
            make_fc(8, 64), nn.ReLU(),
            make_fc(64, embed), nn.ReLU(),
        )

        self.map_encoder = MapEncoder()
        self.attention = EntityAttention(embed_dim=embed)

        self.backbone = nn.Sequential(
            make_fc(embed * 3, 128), nn.LayerNorm(128), nn.ReLU(),
            ResidualBlock(128),
            make_fc(128, 64), nn.LayerNorm(64), nn.ReLU(),
        )

        self.actor_head = make_fc(64, Config.ACTION_NUM, gain=0.01)
        self.critic_feature = nn.Sequential(
            make_fc(64, 64), nn.LayerNorm(64), nn.ReLU(),
            make_fc(64, 32), nn.ReLU(),
        )
        self.critic_head = make_fc(32, Config.VALUE_NUM, gain=1.0)

        self._last_attn_weights = None

    def _split_obs(self, obs):
        cfg = Config
        hero = obs[:, cfg.IDX_HERO[0]:cfg.IDX_HERO[1]]
        mon1 = obs[:, cfg.IDX_MON1[0]:cfg.IDX_MON1[1]]
        mon2 = obs[:, cfg.IDX_MON2[0]:cfg.IDX_MON2[1]]
        map_feat = obs[:, cfg.IDX_MAP[0]:cfg.IDX_MAP[1]]
        progress = obs[:, cfg.IDX_PROGRESS[0]:cfg.IDX_PROGRESS[1]]
        dyn1 = obs[:, cfg.IDX_DYN1[0]:cfg.IDX_DYN1[1]]
        dyn2 = obs[:, cfg.IDX_DYN2[0]:cfg.IDX_DYN2[1]]
        trea = obs[:, cfg.IDX_TREA[0]:cfg.IDX_TREA[1]]
        buff = obs[:, cfg.IDX_BUFF[0]:cfg.IDX_BUFF[1]]
        phase = obs[:, cfg.IDX_PHASE[0]:cfg.IDX_PHASE[1]]

        hero_ctx = torch.cat([hero, progress, phase], dim=1)
        mon1_feat = torch.cat([mon1, dyn1], dim=1)
        mon2_feat = torch.cat([mon2, dyn2], dim=1)
        res_feat = torch.cat([trea, buff], dim=1)
        return hero_ctx, mon1_feat, mon2_feat, res_feat, map_feat

    def forward(self, obs, inference=False):
        hero_ctx, mon1_feat, mon2_feat, res_feat, map_feat = self._split_obs(
            obs)

        hero_enc = self.hero_encoder(hero_ctx)
        mon1_enc = self.monster_encoder(mon1_feat)
        mon2_enc = self.monster_encoder(mon2_feat)
        resource_enc = self.resource_encoder(res_feat)
        map_enc = self.map_encoder(map_feat)

        context, attn_weights = self.attention(
            hero_enc,
            [mon1_enc, mon2_enc, resource_enc, map_enc],
        )
        self._last_attn_weights = attn_weights.detach()

        # Keep a direct path for nearest threat to avoid attention over-smoothing.
        fused = torch.cat([hero_enc, context, mon1_enc], dim=1)
        hidden = self.backbone(fused)
        logits = self.actor_head(hidden)
        value_hidden = self.critic_feature(hidden)
        value = self.critic_head(value_hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
