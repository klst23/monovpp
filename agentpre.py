"""
AgentNavPre: 基于视频预测世界模型的单目 VLN 智能体 (由 agent_nav_mono.py 修改).

与 AgentNavMono 的区别:
    原代码中候选点特征的来源是
        batch_pano_features = self._render_pano_feature(obs)      # 3DGS 渲染全景 + MAE 编码
        edge_tokens = self._forward_candidate_feature(wp_outputs, batch_pano_features)  # PRET OPE
    这里取缔该方案, 改为使用视频预测模块 VideoPredictor (vpp_vln_v10_predictor.py):
        1. 从观测中维护 RGB 历史帧序列 (初始 360° 环视帧 + 每步沿途帧);
        2. 对每个候选路点 (waypoint), 由其相对当前 agent 的角度差/距离差离散化出
           顺序 7 步轨迹 (先旋转对准目标, 再直行; 角度单位 30°, 距离单位 0.25m, 均向下取整;
           轨迹词表: 1=前进, 2=后退, 3=左转, 4=右转, 0=停止);
        3. 将 RGB 历史 + 轨迹 + 指令文本送入 VideoPredictor, 得到该候选点对应的
           未来 latent 预测 (8, 4, 32, 32);
        4. 将 latent 投影到 PRET hidden_dim (768), 作为 edge_tokens 供拓扑图/规划器使用.

保留部分: mobile_sam + Mapper3D 点云建图, 3DGS 俯视渲染 + waypoint_predictor (候选点仍由其预测),
PRET 规划器 (MAM/CCM), 拓扑图逻辑, 三种 rollout (loss/eval/inference).
删除部分: AgentMAE, _render_pano_feature, _get_camera, pano_angle_feature, PRET OPE 路径.

"""
import sys
import math
import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical

import utils
from config import device

from model.PRET import PRET
from model.mapper import Mapper3DFF as Mapper3D

from model.transferableGS import TransferableGS, pipe
from model.feature_3dgs.render import render
from model.feature_3dgs.camera import MiniCam
from model.feature_3dgs.gaussian_model import GaussianModel
from model.mobilesam_ae import MobileSAM512

from .agent_nav import TopoMapper, draw_topdown_map, batch_obs, to_global_xyz

from .agent_waypoint import AgentWaypoint

from model.Video_Former import Video_Former_3D

# 视频预测模块位于项目根目录
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))
from svd import VideoPredictor


class AgentNavPre(nn.Module):
    """
    navigation in continuous environments with monocular camera,
    candidate (edge) features come from a video-prediction world model.
    """

    # 轨迹离散化参数
    ANGLE_UNIT_DEG = 15.0   # 每次旋转 15°
    DIST_UNIT_M = 0.25      # 每步前进 0.25m
    TRAJ_LEN = 7            # 顺序 7 步轨迹

    # 轨迹动作词表, 与训练 SVD 世界模型时完全一致:
    #   0=停止, 1=前进, -1=后退, 2=左转, 3=右转
    # (habitat 中 rel_heading > 0 为逆时针, 即目标在左侧 -> 左转(2);
    #  rel_heading < 0 -> 右转(3); 后退(-1) 在本任务轨迹中不会出现)

    def __init__(self, args, config):
        super().__init__()

        assert config.SIMULATOR.DEPTH_SENSOR['MAX_DEPTH'] == 10
        assert config.SIMULATOR.DEPTH_SENSOR['NORMALIZE_DEPTH'] == False
        assert config.SIMULATOR.RGB_SENSOR.POSITION[1] == config.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        assert config.SIMULATOR.RGB_SENSOR.HEIGHT == config.SIMULATOR.RGB_SENSOR.WIDTH

        self.HFOV = np.deg2rad(config.SIMULATOR.RGB_SENSOR.HFOV)
        self.VFOV = self.HFOV
        self.camera_height = config.SIMULATOR.RGB_SENSOR.POSITION[1]

        self.count = 0
        self.not_train_model = set()

        # settings
        self.max_step = args.max_step
        self.mask_visited = args.mask_visited
        self.localize_tolerance = args.localize_tolerance
        self.use_tryout = args.tryout and (not config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)

        # control point number
        self.max_obs_num = 12

        # visualize
        self.visualize = args.visualize
        self.video_frames = []

        # vision encoder (仍用于点云建图, 供 waypoint 预测的俯视渲染使用)
        self.mobile_sam = MobileSAM512(128)
        for params in self.mobile_sam.parameters():
            params.requires_grad = False
        self.not_train_model.add('mobile_sam')

        # 3dgs (仅用于俯视渲染 -> waypoint 预测, 不再渲染全景)
        self.render_model = TransferableGS()

        # waypoint predictor
        self.waypoint_predictor = AgentWaypoint(args)
        self.waypoint_predictor.load(args.load_waypoint_predictor, iteration=args.load_waypoint_predictor_iter)
        for params in self.waypoint_predictor.parameters():
            params.requires_grad = False
        self.not_train_model.add('waypoint_predictor')

        # video prediction world model (取代 MAE + 全景渲染 + OPE)
        # checkpoint 路径由 policy_models/module/config_v10.py 的 wm_args.ckpt_path 指定
        self.video_predictor = VideoPredictor()
        self.video_predictor.eval()
        for params in self.video_predictor.parameters():
            params.requires_grad = False
        self.not_train_model.add('video_predictor')

        # planner, add the stop embedding in model to save the model more conveniently
        self.model = PRET(args)
        self.model.STOP_embedding = nn.Parameter(torch.zeros(1, 1, self.model.hidden_dim))
        self.model.clip_project = nn.Linear(512, 768)  # unused, kept for checkpoint compatibility

        # 将预测 latent (8, 4, 32, 32) 编码为 edge token (hidden_dim=768):
        # 每帧 latent (4, 32, 32) 按 8x8 patch 划分为 4x4=16 个 token (每个 4*8*8=256 维),
        # 得到 (8, 16, 256) 的视频 token 序列, 送入 Video_Former_3D (Perceiver Resampler
        # + 跨帧时序注意力) 融合为 8 个 query, 平均后即"从当前点走向该候选点"的视频特征. 可训练.
        self.latent_patch_size = 8
        latent_token_dim = 4 * self.latent_patch_size * self.latent_patch_size  # 256
        self.video_former = Video_Former_3D(
            dim=self.model.hidden_dim,      # 768, 直接对齐 PRET
            depth=2,
            condition_dim=latent_token_dim, # 256
            dim_head=64,
            heads=8,
            num_latents=8,                  # 8 帧 x 每帧 1 个 query
            num_frame=8,
            num_time_embeds=8,              # 需 >= 帧数
            ff_mult=4,
            use_temporal=True,              # 开启跨帧时序注意力
        )

        # 可学习的 attention pooling queries (双粒度):
        #   path_query: 关注前 4 帧 (行走路径质量: 路是否通畅、方向是否对)
        #   dest_query: 关注后 4 帧 (目的地特征: 到达后看到什么)
        self.pool_query_path = nn.Parameter(torch.randn(1, 1, self.model.hidden_dim))
        self.pool_query_dest = nn.Parameter(torch.randn(1, 1, self.model.hidden_dim))

    # utils
    def _add_visualize_frames(self, obs_list, metrics):
        print('add frame')
        for i, obs in enumerate(obs_list):
            top_down_map = draw_topdown_map(metrics, self.batch_maps[0])
            self.video_frames.append((obs['rgb'], top_down_map))
        print('---')

    def _imgs_to_tensor(self, batch_image):
        batch_img_torch = []
        for i in range(len(batch_image)):
            img = batch_image[i]
            assert img.shape[0] == img.shape[1]
            img_torch = torch.as_tensor(img, device=device)
            img_torch = img_torch.permute(2, 0, 1).contiguous()
            batch_img_torch.append(img_torch)
        batch_img_torch = torch.stack(batch_img_torch)  # (B, 3, H, W)
        return batch_img_torch

    # mapping
    @torch.no_grad()
    def _render_topdown(self, mapper, position):
        map_size = self.waypoint_predictor.map_size
        cell_size = self.waypoint_predictor.cell_size

        # convert to colmap coordinates
        x, y, z = position
        position = np.array([x, -y, -z])

        points = mapper.points.clone().detach()
        points[:, 1:] *= -1
        gaussians = self.render_model([points], [mapper.colors], [mapper.features])[0]

        # prepare camera matrix
        TOP_DOWN_H = 100  # set a remote value to mimic orthogonal projection
        position[1] -= TOP_DOWN_H
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ])
        T = -R.T.dot(position.reshape(3, 1)).reshape(3)
        map_radius = map_size / 2 * cell_size
        camera = MiniCam(
            map_size, map_size, R, T,
            np.arctan(map_radius/TOP_DOWN_H)*2, np.arctan(map_radius/TOP_DOWN_H)*2)

        # remove some points
        local_points_mask = (gaussians._xyz[:, 1] > -(y+1)) & (gaussians._xyz[:, 1] < -(y-3))
        new_gaussians = GaussianModel(3)
        new_gaussians._xyz = gaussians._xyz[local_points_mask]
        new_gaussians._features_dc = gaussians._features_dc[local_points_mask]
        new_gaussians._features_rest = gaussians._features_rest[local_points_mask]
        new_gaussians._scaling = gaussians._scaling[local_points_mask]
        new_gaussians._rotation = gaussians._rotation[local_points_mask]
        new_gaussians._opacity = gaussians._opacity[local_points_mask]
        new_gaussians._semantic_feature = gaussians._semantic_feature[local_points_mask]

        # render
        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        render_results = render(camera, new_gaussians, pipe, bg_color)
        img = render_results['render']  # (3, H, W)
        depth = render_results['depth'].squeeze(0)  # (H, W)
        mask = (depth > 0.1)
        feature_map = render_results['feature_map']  # (C, H, W)

        return img, feature_map, mask

    @torch.no_grad()
    def _forward_waypoint(self, obs):
        B = len(self.batch_pcd)
        map_size = self.waypoint_predictor.map_size
        cell_size = self.waypoint_predictor.cell_size

        batch_feature_map = []
        for i in range(B):
            x, y, z = obs['globalgps'][i]
            img, feature_map, mask = self._render_topdown(self.batch_pcd[i], (x, y, z))
            batch_feature_map.append(feature_map)
        batch_feature_map = torch.stack(batch_feature_map)  # (B, C, H, W)
        _, _, batch_waypoints = self.waypoint_predictor.forward_inference(batch_feature_map)

        # convert waypoint to heading and distance
        wp_outputs = dict()
        wp_outputs['cand_headings'] = []
        wp_outputs['cand_distances'] = []
        for b in range(B):
            heading = obs['heading'][b, 0]
            waypoints = batch_waypoints[b]

            cand_headings = []
            cand_distances = []
            for k in range(waypoints.shape[0]):
                i, j = waypoints[k]
                i -= (map_size - 1) / 2.  # corresponding z-axis in habitat coordinates
                j -= (map_size - 1) / 2.  # corresponding x-axis in habitat coordinates
                abs_heading = - np.arctan2(j, -i)
                rel_heading = abs_heading - heading  # relative heading
                distance = math.sqrt(i * i + j * j) * cell_size
                cand_headings.append(rel_heading)
                cand_distances.append(distance)
            wp_outputs['cand_headings'].append(cand_headings)
            wp_outputs['cand_distances'].append(cand_distances)

        return wp_outputs

    # ---------------- 视频预测式候选特征 (取代全景渲染 + OPE) ----------------
    def _update_rgb_history(self, obs):
        """
        维护每个环境的 RGB 历史帧: 先补充上一段轨迹的沿途帧, 再追加当前帧.
        列表最后一帧始终为当前观测, 与 VideoPredictor 的约定一致.
        """
        B = obs['rgb'].shape[0]
        if 'obs_list' in obs:
            for i in range(B):
                for o in obs['obs_list'][i]:
                    self.batch_rgb_history[i].append(o['rgb'])
        for i in range(B):
            self.batch_rgb_history[i].append(obs['rgb'][i])

    @staticmethod
    def _wrap_angle(angle):
        """wrap to (-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _waypoint_to_actions(self, rel_heading, distance):
        """
        由候选点相对当前 agent 的角度差和距离差, 生成顺序 7 步轨迹:
        先旋转对准目标方向, 再直走. 不必一定走到; 超过 7 步截断, 不足补 0 (停止).

        动作词表与训练 SVD 世界模型时一致: 0=停止, 1=前进, -1=后退, 2=左转, 3=右转.
        habitat 中 rel_heading > 0 为逆时针 (目标在左侧) -> 左转(2);
        rel_heading < 0 -> 右转(3). 角度按 15° 四舍五入, 距离按 0.25m 四舍五入.
        """
        rel = self._wrap_angle(float(rel_heading))
        deg = np.degrees(rel)
        num_turn = int(abs(deg) / self.ANGLE_UNIT_DEG + 0.5)
        turn_act = 2 if deg > 0 else 3  # 2=左转, 3=右转 (SVD 训练词表)
        num_forward = int(float(distance) / self.DIST_UNIT_M + 0.5)

        traj = [turn_act] * num_turn + [1] * num_forward
        traj = traj[:self.TRAJ_LEN]
        traj = traj + [0] * (self.TRAJ_LEN - len(traj))
        return traj

    def _compute_prev_actions(self, curr_xyz, curr_heading, prev_pose):
        """
        由上一步位姿到当前位姿的相对运动, 离散化出"刚执行过的"7 步动作序列,
        作为 VideoPredictor 的 prev_actions. episode 起始 (prev_pose 为 None) 返回空列表.
        """
        if prev_pose is None:
            return []
        prev_xyz, prev_heading = prev_pose
        dx = float(curr_xyz[0] - prev_xyz[0])
        dz = float(curr_xyz[2] - prev_xyz[2])
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < 1e-3:  # 原地未动 (如被阻挡), 视为全停
            return [0] * self.TRAJ_LEN
        # habitat: x = -d*sin(theta), z = -d*cos(theta)
        abs_heading = np.arctan2(-dx, -dz)
        rel = self._wrap_angle(abs_heading - prev_heading)
        return self._waypoint_to_actions(rel, dist)

    def _project_latent(self, latents):
        """
        (N, 8, 4, 32, 32) 预测 latent -> (N, hidden_dim) edge token.

        8帧 latent 经 Video_Former_3D 融合为 (N, 8, 768) 的帧级特征,
        分成前4帧(路径)和后4帧(目的地)各做 attention pooling,
        再融合为一个 768 维的 edge token, 同时保留时序和空间信息.
        """
        N, F_, C, H, W = latents.shape  # (N, 8, 4, 32, 32)
        p = self.latent_patch_size
        x = latents.reshape(N, F_, C, H // p, p, W // p, p)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)  # (N, F, H/p, W/p, C, p, p)
        x = x.reshape(N, F_, (H // p) * (W // p), C * p * p)  # (N, 8, 16, 256)

        tokens = self.video_former(x)  # (N, 8, 768)

        # 双粒度 attention pooling: 前4帧=路径质量, 后4帧=目的地特征
        path_tokens = tokens[:, :4, :]   # (N, 4, 768)
        dest_tokens = tokens[:, 4:, :]   # (N, 4, 768)
        q_path = self.pool_query_path.expand(N, -1, -1)  # (N, 1, 768)
        q_dest = self.pool_query_dest.expand(N, -1, -1)  # (N, 1, 768)
        path_feat = F.scaled_dot_product_attention(q_path, path_tokens, path_tokens)  # (N, 1, 768)
        dest_feat = F.scaled_dot_product_attention(q_dest, dest_tokens, dest_tokens)  # (N, 1, 768)

        # 融合: 拼接后线性映射回 hidden_dim, 保持下游维度不变
        combined = torch.cat([path_feat, dest_feat], dim=1)  # (N, 2, 768)
        combined = combined.reshape(N, -1)                    # (N, 1536)
        return self.dual_pool_proj(combined)                  # (N, 768)

    def _forward_candidate_feature(self, obs, wp_outputs, num_inference_steps=None):
        """
        对每个候选路点: 离散化 7 步轨迹 -> 视频预测模块 -> 未来 latent -> 投影为 edge token.
        Returns:
            (B, max_candidate_num, hidden_dim)
        """
        B = obs['rgb'].shape[0]
        predictor = self.video_predictor

        feature_list = []
        for i in range(B):
            images = self.batch_rgb_history[i]
            instruction = self.instruction_texts[i]

            # 上一段执行的动作 (SVD 训练词表); 起始步为空
            prev_actions = self._compute_prev_actions(
                obs['globalgps'][i], obs['heading'][i, 0], self.batch_prev_pose[i])
            initial = len(prev_actions) == 0

            # 历史帧 / 当前帧 latent 每个环境只编码一次, 各候选点共享
            with torch.no_grad():
                his_latent_gt = predictor.build_history_latents(images, initial)  # (1, 4, 4, 32, 32)
                current_latent = predictor.img_to_latent(images[-1]).float()      # (1, 4, 32, 32)

            cand_latents = []
            headings_i = wp_outputs['cand_headings'][i]
            distances_i = wp_outputs['cand_distances'][i]
            if len(headings_i) > 0:
                # 所有候选点的动作序列堆成一个 batch, 一次 UNet 前向同时预测
                # (CFG 下 UNet 实际 batch 为 2*chunk, 分块以防显存溢出)
                actions_all = torch.stack([
                    predictor.build_action_sequence(
                        prev_actions, self._waypoint_to_actions(h, d))
                    for h, d in zip(headings_i, distances_i)
                ])  # (N, 12, 7)

                CAND_CHUNK = 5
                with torch.no_grad():
                    for s in range(0, actions_all.shape[0], CAND_CHUNK):
                        chunk = actions_all[s:s + CAND_CHUNK]
                        pred_latents = predictor.gen_latent(
                            current_latent, his_latent_gt, instruction, chunk,
                            num_inference_steps=num_inference_steps)  # (n, 8, 4, 32, 32)
                        cand_latents.append(pred_latents.float())

            if len(cand_latents) > 0:
                cand_latents = torch.cat(cand_latents, dim=0).to(device)  # (N, 8, 4, 32, 32)
                tokens = self._project_latent(cand_latents)               # (N, hidden_dim)
            else:
                tokens = torch.zeros(0, self.model.hidden_dim, device=device)
            feature_list.append(tokens)

        edge_tokens, _ = utils.stack_list_of_tensor(feature_list)  # (B, max_N, hidden_dim)
        return edge_tokens.to(device)

    def _update_pcd(self, obs):
        B = obs['rgb'].shape[0]
        is_initial_step = 'obs_list' not in obs

        if is_initial_step:
            imgs = self._imgs_to_tensor(obs['rgb'])
            feautre_maps = self.mobile_sam(imgs)  # (B, C, H, W)
            for i in range(B):
                x, y, z = obs['globalgps'][i]
                heading = obs['heading'][i, 0]
                pose = (x, y + self.camera_height, z, heading)
                self.batch_pcd[i].update(imgs[i], obs['depth'][i], feautre_maps[i], self.HFOV, self.VFOV, pose)
        else:
            for i in range(B):
                obs_list = obs['obs_list'][i][::-2][::-1]  # sample observations with interval 1
                obs_list = obs_list[-self.max_obs_num:]  # take the final max_obs_num observations
                if len(obs_list) == 0:
                    continue
                imgs = [o['rgb'] for o in obs_list]
                imgs = self._imgs_to_tensor(imgs)  # to tensor
                feature_maps = self.mobile_sam(imgs)
                for k, o in enumerate(obs_list):
                    x, y, z = o['globalgps']
                    heading = o['heading'].item()
                    pose = (x, y + self.camera_height, z, heading)
                    self.batch_pcd[i].update(imgs[k], o['depth'], feature_maps[k], self.HFOV, self.VFOV, pose)

    def _forward_map(self, envs, obs, batch_prev_node, batch_last_choice,
                     num_inference_steps=None):
        """
        Predict waypoints, extract features via video prediction, and update topo map
        """
        self._update_rgb_history(obs)  # 必须在候选特征之前, 保证最后一帧为当前观测
        self._update_pcd(obs)
        wp_outputs = self._forward_waypoint(obs)
        edge_tokens = self._forward_candidate_feature(obs, wp_outputs, num_inference_steps=num_inference_steps)

        # update graph, this is the most complex part
        batch_xyz = obs['globalgps']  # (B, 3) in word frame
        batch_heading = obs['heading'].squeeze(1)  # (B,)
        batch_curr_node = [None] * len(batch_xyz)
        for i, topo_map in enumerate(self.batch_maps):
            candidate_xyz = []
            for cand_heading, cand_dis in zip(wp_outputs['cand_headings'][i], wp_outputs['cand_distances'][i]):
                pos = to_global_xyz(cand_heading, cand_dis, batch_xyz[i], batch_heading[i])
                candidate_xyz.append(pos)

            batch_curr_node[i] = topo_map.update_graph(
                batch_prev_node[i], batch_xyz[i], batch_last_choice[i],
                candidate_xyz, edge_tokens[i])

        # 记录当前位姿, 供下一步推断 prev_actions
        for i in range(len(batch_xyz)):
            self.batch_prev_pose[i] = (np.array(batch_xyz[i], copy=True), float(batch_heading[i]))

        return batch_curr_node

    # planning
    def _get_path_features(self, batch_curr_node):
        path_features = []
        local_features = []
        for i, topo_map in enumerate(self.batch_maps):
            curr_node = batch_curr_node[i]
            path_feature, local_feature = topo_map.get_path_feature(curr_node)
            path_features.append(path_feature)
            local_features.append(local_feature)

        path_features, path_padding_mask = utils.stack_list_of_tensor(path_features)
        local_features, local_padding_mask = utils.stack_list_of_tensor(local_features)
        return path_features, path_padding_mask, local_features, local_padding_mask

    def _forward_global_action(self, batch_curr_node, text_features, text_padding_mask):
        """
        Returns:
            probs: (B, max_frontier_num)
            batch_frontiers: list[list[str]], batch of frontiers, the first frontier is current node
        """
        B = len(batch_curr_node)
        stop_embedding = self.model.STOP_embedding.expand(B, 1, -1)  # (B, 1, C)

        # predict path feature
        path_features, path_padding_mask, local_features, local_padding_mask = self._get_path_features(batch_curr_node)
        local_features = torch.cat([stop_embedding, local_features], dim=1)
        local_padding_mask = F.pad(local_padding_mask, (1, 0), value=False)
        path_tokens, local_tokens = self.model.forward_MAM(
            text_features, text_padding_mask,
            path_features=path_features, path_padding_mask=path_padding_mask,
            local_features=local_features, local_padding_mask=local_padding_mask)

        # update path feature, get frontiers, and predict
        batch_features = []
        batch_frontiers = []
        for i, topo_map in enumerate(self.batch_maps):
            curr_node = batch_curr_node[i]
            topo_map.set_node_feature(curr_node, local_tokens[i])
            frontiers, feature = topo_map.get_frontier_features(curr_node)
            batch_features.append(feature)
            batch_frontiers.append(frontiers)
        batch_features, padding_mask = utils.stack_list_of_tensor(batch_features)

        logits = self.model.forward_CCM(batch_features, padding_mask)
        probs = torch.softmax(logits, dim=1)

        return probs, batch_frontiers

    def _get_habitat_actions(self, batch_curr_node, batch_frontiers, batch_global_action):
        batch_actions = []
        batch_prev_node = [None] * len(self.batch_maps)
        for i, topo_map in enumerate(self.batch_maps):
            curr_node = batch_curr_node[i]
            frontiers = batch_frontiers[i]
            global_action = batch_global_action[i]

            action = dict()
            if global_action == 0:
                action["action"] = {
                    "action": "STOP",
                }
                batch_prev_node[i] = curr_node
            else:
                # back tracking or step forward
                frontier_viewpoint = frontiers[global_action]
                path_nodes, path = topo_map.get_path(curr_node, frontier_viewpoint)
                batch_prev_node[i] = path_nodes[-2]
                action["action"] = {
                    "action": "TRAJECTORY",
                    "action_args": {
                        "trajectory" : path,
                        "mode": "control",
                        "visualize": True,  # 必须为 True, 以便返回沿途 obs_list 维护 RGB 历史
                    }
                }
            batch_actions.append(action)
        return batch_actions, batch_prev_node

    def _get_teacher_actions(self, envs, obs, batch_frontiers):
        teacher_actions = []
        batch_curr_xyz = obs['globalgps']
        batch_goal_xyz = [path[-1] for path in self.groundtruth_paths]

        for i, topo_map in enumerate(self.batch_maps):
            curr_xyz = batch_curr_xyz[i]
            goal_xyz = batch_goal_xyz[i]
            frontiers = batch_frontiers[i]
            distance_to_goal = np.linalg.norm(curr_xyz - goal_xyz)

            if len(frontiers) == 1:  # can only stop, e.g. frontier = ['stop']
                action = -1  # loss will not be computed
            elif distance_to_goal < 1.5:  # if reach the goal, stop
                action = 0  # STOP
            elif self.mask_visited and topo_map.localize_visited(goal_xyz) is not None:  # target is not available
                action = -1  # loss will not be computed
            else:
                action = None  # get the closest node to goal
                min_distance = 1e6
                for j, node in enumerate(frontiers):
                    if j == 0:
                        continue
                    node_xyz = topo_map.graph.nodes[node]['xyz']
                    d = envs.call_at(i, 'point_distance_to_goal', {'pos': node_xyz})
                    if d < min_distance:
                        min_distance = d
                        action = j
                if action is None:  # no valid frontier
                    action = -1
            teacher_actions.append(action)
        return teacher_actions


    # public
    def reset(self, envs):
        """
        Args:
            envs: vector_env
        """
        batch_size = envs.num_envs
        current_episodes = envs.current_episodes()

        texts = [e.instruction.instruction_text for e in current_episodes]
        self.text_features, self.text_padding_mask = self.model.forward_text(texts)
        self.instruction_texts = list(texts)  # 视频预测模块需要原始指令文本

        self.batch_pcd = [
            Mapper3D(self.mobile_sam.feature_map_size, feature_dim=self.mobile_sam.feature_dim, device=device)
            for _ in range(batch_size)]
        self.batch_maps = [TopoMapper(self.localize_tolerance) for _ in range(batch_size)]

        # RGB 历史帧与上一步位姿 (供视频预测模块使用)
        self.batch_rgb_history = [[] for _ in range(batch_size)]
        self.batch_prev_pose = [None] * batch_size

        if self.training:
            self.groundtruth_paths = [e.reference_path for e in current_episodes]

        # rotate 360 at the begininig
        for _ in range(0, 12):
            envs.step([2] * batch_size)  # turn left 15°
            outputs = envs.step([2] * batch_size)  # turn left 15°
            obs_list, _, _, _ = [list(x) for x in zip(*outputs)]

            imgs = [obs_list[i]['rgb'] for i in range(batch_size)]
            imgs = self._imgs_to_tensor(imgs)
            batch_feature_map = self.mobile_sam(imgs)
            for i in range(batch_size):
                obs = obs_list[i]
                x, y, z = obs['globalgps']
                heading = obs['heading'].item()
                pose = (x, y + self.camera_height, z, heading)
                self.batch_pcd[i].update(imgs[i], obs['depth'], batch_feature_map[i], self.HFOV, self.VFOV, pose)
                self.batch_rgb_history[i].append(obs['rgb'])  # 环视帧作为初始历史 (12 帧)

    def _pop_env_state(self, i):
        """移除已结束环境的各类状态 (供 rollout 循环调用)."""
        self.batch_pcd.pop(i)
        self.batch_maps.pop(i)
        self.batch_rgb_history.pop(i)
        self.batch_prev_pose.pop(i)
        self.instruction_texts.pop(i)

    def forward(self, envs, mode):
        if mode == 'loss':
            return self.forward_loss(envs)
        elif mode == 'eval':
            return self.forward_eval(envs)
        elif mode == 'inference':
            return self.forward_inference(envs)
        raise ValueError()

    def forward_loss(self, envs):
        """
        Args:
            envs: vector_env
        Returns:
            loss, and log_dict
        """
        ACTION_STRATEGY = 'teacher' if np.random.rand() < 0.2 else 'sample'

        # reset
        envs.resume_all()
        obs_list = envs.reset()
        obs = batch_obs(obs_list)
        self.reset(envs)  # create map, encode instructions

        # 训练时用较少的去噪步数加速 (5步 vs 评测时15步), 配置在 config_v10.wm_args
        train_steps = getattr(self.video_predictor.args, 'train_num_inference_steps', None)

        # some variables
        loss = 0.0
        total_step_num = 0
        batch_size = envs.num_envs
        batch_prev_node = [None] * batch_size
        batch_last_choice = [None] * batch_size

        active_env_index = list(range(batch_size))

        # rollout
        for step in range(self.max_step):
            # update map (训练用5步加速, 评测用默认15步)
            batch_curr_node = self._forward_map(envs, obs, batch_prev_node, batch_last_choice,
                                                num_inference_steps=train_steps)

            # forward
            text_features = self.text_features[active_env_index, :, :]
            text_padding_mask = self.text_padding_mask[active_env_index, :]
            probs, batch_frontiers = self._forward_global_action(batch_curr_node, text_features, text_padding_mask)

            # loss
            global_teacher_actions = self._get_teacher_actions(envs, obs, batch_frontiers)
            global_teacher_actions = torch.tensor(global_teacher_actions, device=device)
            ignore_mask = (global_teacher_actions == -1)
            loss = loss + utils.cross_entropy_loss(probs, global_teacher_actions, ignore_mask, reduction='none').sum()

            # process action
            if ACTION_STRATEGY == 'teacher':
                global_actions = global_teacher_actions.clone().detach()  # create copy, to avoid inplace error
            elif ACTION_STRATEGY == 'argmax':
                global_actions = probs.argmax(dim=1)
            elif ACTION_STRATEGY == 'sample':
                global_actions = Categorical(probs).sample()
            else:
                raise ValueError()
            global_actions[ignore_mask] = 0  # if no valid teacher action, stop the agent
            global_actions = [0] * len(global_actions) if step == self.max_step - 1 else global_actions.tolist()
            batch_last_choice = [batch_frontiers[i][a] for i, a in enumerate(global_actions)]

            # step
            batch_actions, batch_prev_node = self._get_habitat_actions(batch_curr_node, batch_frontiers, global_actions)
            outputs = envs.step(batch_actions)
            obs_list, _, dones, infos = [list(x) for x in zip(*outputs)]

            # remove stopped env
            for i in reversed(list(range(len(dones)))):  # reversed, otherwise the pop operation will error
                if not dones[i]:
                    continue

                total_step_num += step + 1
                active_env_index.pop(i)
                batch_prev_node.pop(i)
                batch_last_choice.pop(i)
                obs_list.pop(i)
                envs.pause_at(i)
                self._pop_env_state(i)
                self.groundtruth_paths.pop(i)

            if len(active_env_index) == 0:
                break

            # for next step
            obs = batch_obs(obs_list)

        loss = loss / total_step_num
        log_dict = pd.Series(dtype=np.float32)
        log_dict['loss'] = loss.item()
        return loss, log_dict

    @torch.no_grad()
    def forward_eval(self, envs):
        """
        Args:
            envs: vector_env
        Returns:
            dataframe, each row is a sample, each column is a metric
        """
        # reset
        envs.resume_all()
        obs_list = envs.reset()
        obs = batch_obs(obs_list)
        self.reset(envs)  # create map, encode instructions

        # some variables
        batch_size = envs.num_envs
        batch_prev_node = [None] * batch_size
        batch_last_choice = [None] * batch_size
        active_env_index = list(range(batch_size))
        log_dict = pd.DataFrame(dtype=np.float32)

        # rollout
        for step in range(self.max_step):
            self.count += 1
            # update map
            batch_curr_node = self._forward_map(envs, obs, batch_prev_node, batch_last_choice)

            # forward
            text_features = self.text_features[active_env_index, :, :]
            text_padding_mask = self.text_padding_mask[active_env_index, :]
            probs, batch_frontiers = self._forward_global_action(batch_curr_node, text_features, text_padding_mask)

            # process action
            global_actions = probs.argmax(dim=1)  # (B,)
            global_actions = [0] * len(global_actions) if step == self.max_step - 1 else global_actions.tolist()
            batch_last_choice = [batch_frontiers[i][a] for i, a in enumerate(global_actions)]

            # step
            batch_actions, batch_prev_node = self._get_habitat_actions(batch_curr_node, batch_frontiers, global_actions)
            outputs = envs.step(batch_actions)
            obs_list, _, dones, infos = [list(x) for x in zip(*outputs)]

            # record metrics for stopped episode, and remove them
            current_episodes = envs.current_episodes()
            for i in reversed(list(range(len(dones)))):  # reversed, otherwise the pop operation will error
                if not dones[i]:
                    continue

                current_episode = current_episodes[i]
                episode_id = current_episode.episode_id
                for metric_name, value in infos[i].items():
                    if metric_name in ('top_down_map_vlnce', 'position'):
                        continue
                    if metric_name == 'collisions':
                        log_dict.at[episode_id, metric_name] = value['count'] / len(infos[i]['position']['position'])
                        continue
                    log_dict.at[episode_id, metric_name] = value
                log_dict.at[episode_id, 'backtracking_fail'] = 1 if batch_curr_node[i] is None else 0
                log_dict.at[episode_id, 'n_plan'] = step
                log_dict.at[episode_id, 'scan'] = current_episode.scene_id.split('/')[3]

                active_env_index.pop(i)
                batch_prev_node.pop(i)
                batch_last_choice.pop(i)
                obs_list.pop(i)
                envs.pause_at(i)
                self._pop_env_state(i)

            if len(active_env_index) == 0:
                break

            # for next step
            obs = batch_obs(obs_list)

            if self.visualize:
                self._add_visualize_frames(obs['obs_list'][0], envs.get_metrics()[0])
        # end navigation
        if self.visualize:
            self.video_frames = []
            exit()

        # make metric name shorter
        mapper = {
            "spl": "SPL",
            "success": "SR", "oracle_success": "OSR",
            "path_length": "TL", "distance_to_goal": "NE",
        }
        log_dict.rename(columns=mapper, inplace=True)
        return log_dict

    @torch.no_grad()
    def forward_inference(self, envs):
        """
        Args:
            envs: vector_env
        Returns:
            dataframe, each row is a sample, each column is a metric
        """
        # reset
        envs.resume_all()
        obs_list = envs.reset()
        obs = batch_obs(obs_list)
        self.reset(envs)  # create map, encode instructions

        # some variables
        batch_size = envs.num_envs
        batch_prev_node = [None] * batch_size
        batch_last_choice = [None] * batch_size
        active_env_index = list(range(batch_size))
        trajectories = {e.episode_id: list() for e in envs.current_episodes()}

        for i, episode in enumerate(envs.current_episodes()):
            episode_id = episode.episode_id
            trajectories[episode_id].append({
                "position": obs_list[i]['globalgps'].tolist(),
                "heading": obs_list[i]['heading'].item(),
                "stop": False,
            })

        # rollout
        for step in range(self.max_step):
            # update map
            batch_curr_node = self._forward_map(envs, obs, batch_prev_node, batch_last_choice)

            # forward
            text_features = self.text_features[active_env_index, :, :]
            text_padding_mask = self.text_padding_mask[active_env_index, :]
            probs, batch_frontiers = self._forward_global_action(batch_curr_node, text_features, text_padding_mask)

            # process action
            global_actions = probs.argmax(dim=1)  # (B,)
            global_actions = [0] * len(global_actions) if step == self.max_step - 1 else global_actions.tolist()
            batch_last_choice = [batch_frontiers[i][a] for i, a in enumerate(global_actions)]

            # step
            batch_actions, batch_prev_node = self._get_habitat_actions(batch_curr_node, batch_frontiers, global_actions)
            outputs = envs.step(batch_actions)
            obs_list, _, dones, infos = [list(x) for x in zip(*outputs)]

            # record metrics for stopped episode, and remove them
            current_episodes = envs.current_episodes()
            for i in reversed(list(range(len(dones)))):  # reversed, otherwise the pop operation will error
                if not dones[i]:
                    continue

                episode_id = current_episodes[i].episode_id
                trajectories[episode_id].append({
                    "position": obs_list[i]['globalgps'].tolist(),
                    "heading": obs_list[i]['heading'].item(),
                    "stop": True,
                })

                active_env_index.pop(i)
                batch_prev_node.pop(i)
                batch_last_choice.pop(i)
                obs_list.pop(i)
                envs.pause_at(i)
                self._pop_env_state(i)

            if len(active_env_index) == 0:
                break

            # for next step
            obs = batch_obs(obs_list)

            # record trajectory
            for i, episode in enumerate(envs.current_episodes()):
                episode_id = episode.episode_id
                for obs_ in obs['obs_list'][i]:
                    trajectories[episode_id].append({
                        "position": obs_['globalgps'].tolist(),
                        "heading": obs_['heading'].item(),
                        "stop": False,
                    })
        return trajectories


    # others
    def save(self, log_dir):
        for name, model in self.named_children():
            if name in self.not_train_model:
                continue
            path = osp.join(log_dir, f'{name}.pt')
            torch.save(model.state_dict(), path)

    def load(self, log_dir, strict=True):
        load_any = False
        for name, model in self.named_children():
            if name in self.not_train_model:
                continue
            path = osp.join(log_dir, f'{name}.pt')
            if osp.exists(path):
                checkpoint = torch.load(path, map_location='cpu')
                model.load_state_dict(checkpoint, strict=strict)
                print(f'{name}.pt loaded')
                load_any = True
        assert load_any, 'Does not load any model!'
