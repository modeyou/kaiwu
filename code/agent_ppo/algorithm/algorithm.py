#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
峡谷追猎 PPO 算法实现。

损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import os
import time

import torch
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [
            p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.lr_start = float(Config.INIT_LEARNING_RATE_START)
        self.lr_end = float(
            getattr(Config, "INIT_LEARNING_RATE_END", self.lr_start))
        self.enable_lr_decay = bool(getattr(Config, "ENABLE_LR_DECAY", False))
        self.lr_decay_steps = max(int(getattr(Config, "LR_DECAY_STEPS", 1)), 1)

        self.beta_start = float(Config.BETA_START)
        self.beta_end = float(getattr(Config, "BETA_END", self.beta_start))
        self.enable_beta_decay = bool(
            getattr(Config, "ENABLE_BETA_DECAY", False))
        self.beta_decay_steps = max(
            int(getattr(Config, "BETA_DECAY_STEPS", 1)), 1)
        self.var_beta = self.beta_start
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        self.value_clip_param = float(
            getattr(Config, "VALUE_CLIP_PARAM", self.clip_param)
        )
        self.enable_value_target_norm = bool(
            getattr(Config, "ENABLE_VALUE_TARGET_NORM", False)
        )

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """Training entry: PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行 PPO 更新。
        """
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack(
            [f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]
                          ).to(self.device).view(-1, 1)
        old_prob = torch.stack(
            [f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack(
            [f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack(
            [f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack(
            [f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack(
            [f.reward_sum for f in list_sample_data]).to(self.device)

        current_lr, current_beta = self._update_schedules()

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        # 裁剪前梯度范数：用于监控训练是否过激。
        grad_clip_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, Config.GRAD_CLIP_RANGE
        )
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "cum_reward": round(reward.mean().item(), 4),
                "total_loss": round(total_loss.item(), 4),
                "lr": round(current_lr, 8),
                "beta": round(current_beta, 8),
                "value_loss": round(info_list["value_loss"].item(), 4),
                "policy_loss": round(info_list["policy_loss"].item(), 4),
                "entropy_loss": round(info_list["entropy_loss"].item(), 4),
                "grad_clip_norm": round(float(grad_clip_norm), 4),
                "clip_frac": round(info_list["clip_frac"].item(), 4),
                "explained_var": round(info_list["explained_var"].item(), 4),
                "adv_mean": round(advantage.mean().item(), 4),
                "ret_mean": round(reward_sum.mean().item(), 4),
                "ret_std": round(info_list["ret_std"].item(), 4),
            }
            self.logger.info(
                f"[train] cum_reward:{results['cum_reward']} "
                f"[train] total_loss:{results['total_loss']} "
                f"lr:{results['lr']} beta:{results['beta']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']} "
                f"grad_norm:{results['grad_clip_norm']} "
                f"clip_frac:{results['clip_frac']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 损失（策略损失 + 价值损失 + 熵正则化）。
        """
        # Masked softmax / 合法动作掩码 softmax
        prob_dist = self._masked_softmax(logits, legal_action)

        # Policy loss (PPO Clip) / 策略损失
        one_hot = torch.nn.functional.one_hot(
            old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        adv = (adv - adv.mean()) / \
            (adv.std(unbiased=False) + 1e-8)  # 标准 PPO 稳定训练的关键
        clip_mask = (ratio > (1 + self.clip_param)
                     ) | (ratio < (1 - self.clip_param))
        clip_frac = clip_mask.float().mean()
        policy_loss1 = -ratio * adv
        policy_loss2 = - \
            ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # Value loss (Clipped) / 价值损失
        vp = value_pred.view(-1, self.value_num)
        ov = old_value.view_as(vp)
        tdret = reward_sum.view_as(vp)

        if self.enable_value_target_norm:
            ret_mean = tdret.mean(dim=0, keepdim=True)
            ret_std = tdret.std(dim=0, unbiased=False,
                                keepdim=True).clamp_min(1e-6)
            vp_loss = (vp - ret_mean) / ret_std
            ov_loss = (ov - ret_mean) / ret_std
            tdret_loss = (tdret - ret_mean) / ret_std
        else:
            ret_std = tdret.std(dim=0, unbiased=False, keepdim=True)
            vp_loss = vp
            ov_loss = ov
            tdret_loss = tdret

        value_clip = ov_loss + (vp_loss - ov_loss).clamp(
            -self.value_clip_param, self.value_clip_param
        )
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret_loss - vp_loss),
                torch.square(tdret_loss - value_clip),
            ).mean()
        )

        # Entropy loss / 熵损失
        entropy_loss = (-prob_dist *
                        torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        # Total loss / 总损失
        total_loss = self.vf_coef * value_loss + \
            policy_loss - self.var_beta * entropy_loss

        # 解释方差：衡量 value 对 return 的拟合质量。
        var_ret = torch.var(tdret, unbiased=False)
        explained_var = 1.0 - \
            torch.var(tdret - vp, unbiased=False) / (var_ret + 1e-8)

        return total_loss, {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "clip_frac": clip_frac,
            "explained_var": explained_var,
            "ret_std": ret_std.mean(),
        }

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking (suppress illegal actions).

        合法动作掩码下的 softmax（将非法动作概率压为极小值）。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)

    def _linear_decay(self, start, end, step, total_steps):
        """Linear decay from start to end over total_steps.

        在 total_steps 内从 start 线性衰减到 end。
        """
        progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
        return start + (end - start) * progress

    def _update_schedules(self):
        """Update optimizer lr and entropy beta based on current train step.

        基于当前 train_step 更新学习率和熵系数。
        """
        if self.enable_lr_decay:
            current_lr = self._linear_decay(
                start=self.lr_start,
                end=self.lr_end,
                step=self.train_step,
                total_steps=self.lr_decay_steps,
            )
        else:
            current_lr = self.lr_start

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        if self.enable_beta_decay:
            self.var_beta = self._linear_decay(
                start=self.beta_start,
                end=self.beta_end,
                step=self.train_step,
                total_steps=self.beta_decay_steps,
            )
        else:
            self.var_beta = self.beta_start

        return current_lr, self.var_beta
