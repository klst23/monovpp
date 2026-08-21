import os
os.environ["HYDRA_PLUGINS"] = "none"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import numpy as np
np.float = np.float64
np.int = np.int64
import torch
import torch.nn as nn
import torchvision.transforms as T
import einops

from diffusers.models import AutoencoderKLTemporalDecoder
from policy_models.module.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from policy_models.module.svd_v10 import CrtlWorld
from policy_models.module.config_v10 import wm_args
from transformers import AutoTokenizer, CLIPTextModelWithProjection


class VideoPredictor(nn.Module):
    """
    接受 RGB 图片序列和一段动作轨迹，输出预测的未来 latent 的视频预测模块。
    """

    def __init__(
        self,
        text_encoder_path='/mnt/data0/chenghao/vln-vpp/video-prediction-policy/clip-vit-base-patch32',
        vae_path='/mnt/data0/chenghao/vln-vpp/video-prediction-policy/pretrained/svd',
        seed=42,
        device=None,
        use_fp16=True,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.seed = seed
        # fp16 仅在 GPU 上启用; UNet/action_encoder/text_encoder 转半精度,
        # VAE (编码/解码) 保持 fp32 以避免数值不稳定, 其耗时占比很小
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        # SVD / CtrlWorld 世界模型 (checkpoint 路径直接取 config_v10.wm_args.ckpt_path)
        args = wm_args()
        model = CrtlWorld(args)
        model.eval()
        if args.ckpt_path is not None:
            print(f"Loading checkpoint from {args.ckpt_path}!")
            state_dict = torch.load(args.ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
        model.to(self.device)

        self.args = args
        pipeline = model.pipeline

        # 文本编码器
        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(self.device).eval()
        self.action_encoder = model.action_encoder.to(self.device).eval()

        for param in pipeline.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in pipeline.vae.parameters():
            param.requires_grad = False
        for param in pipeline.unet.parameters():
            param.requires_grad = False

        pipeline = pipeline.to(self.device)
        pipeline.unet.eval()
        if self.use_fp16:
            pipeline.unet.half()
            self.text_encoder.half()
            self.action_encoder.half()
        pipeline.set_progress_bar_config(disable=True)  # 去噪循环不打印 tqdm 进度条
        self.svd = pipeline

        # 图像编码用 VAE（原脚本 img_to_latent 中每次调用都重新加载，这里只加载一次）
        self.img_vae = AutoencoderKLTemporalDecoder.from_pretrained(
            vae_path, subfolder="vae").to(self.device).eval()
        for param in self.img_vae.parameters():
            param.requires_grad = False

    # ---------------- 预处理 ----------------
    @torch.no_grad()
    def img_to_latent(self, img: np.ndarray):
        """
        单张 RGB 图 (C, H, W) 或 (H, W, C) -> VAE latent (1, 4, 32, 32), 在 CPU 上。
        """
        if img.ndim == 3 and img.shape[-1] == 3:  # HWC -> CHW
            img = img.transpose(2, 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img))
        if img.max() > 1.0:
            img = img.float() / 255.0
        img = T.functional.resize(img.unsqueeze(0), [256, 256], antialias=True)
        img = img * 2.0 - 1.0
        img = img.to(device=self.device, dtype=torch.float32)
        latent = self.img_vae.encode(img).latent_dist.sample().mul_(
            self.img_vae.config.scaling_factor).cpu()
        return latent

    @staticmethod
    def build_action_sequence(prev_actions, cur_actions):
        """
        原脚本中的 get_act: 由上一段已执行动作(取最后 4 个)和当前 7 步轨迹
        构造 (12, 7) 的动作序列张量。prev_actions 为空表示 episode 起始。
        """
        reverse_act = {0: 0, 1: -1, 2: 3, 3: 2}
        if len(prev_actions) == 0:
            act_sequence = [
                [2, 2, 2, 2, 0, 0, 0],
                [2, 2, 2, 0, 0, 0, 0],
                [2, 2, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        else:
            act1, act2, act3, act4 = prev_actions[-4:]
            a1 = reverse_act[int(act1)]
            a2 = reverse_act[int(act2)]
            a3 = reverse_act[int(act3)]
            a4 = reverse_act[int(act4)]
            act_sequence = [
                [a4, a3, a2, a1, 0, 0, 0],
                [a4, a3, a2, 0, 0, 0, 0],
                [a4, a3, 0, 0, 0, 0, 0],
                [a4, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        cur_actions = np.asarray(cur_actions, dtype=int)
        for k in range(len(cur_actions)):
            current = cur_actions[:k + 1]
            pad_len = 7 - len(current)
            current = np.concatenate([current, np.zeros(pad_len, dtype=int)])
            act_sequence.append(current[:7].tolist())

        return torch.tensor(act_sequence, dtype=torch.float32)

    @torch.no_grad()
    def build_history_latents(self, images, initial):
        """
        由 RGB 图片列表构造历史 latent (1, 4, 4, 32, 32)。

        initial=True  (episode 起始, 对应原脚本 text_id==0 and count==0):
            用除最后一帧外的全部图片作为历史候选。
        initial=False (后续步骤):
            在 [frame_now-8, frame_now-2] 上等间隔取 7 帧, 再加上 frame_now-1。
        两种情况最后都按原脚本取 indices=[-6, -4, -3, -2] 的 4 帧。
        """
        his_latent_gt = []
        if initial:
            for k in range(len(images) - 1):
                his_latent_gt.append(self.img_to_latent(images[k]))
        else:
            frame_now = len(images) - 1
            low, high = frame_now - 8, frame_now - 2
            rgb_id = np.linspace(low, high, 7, dtype=int).tolist() + [frame_now - 1]
            his_latent_gt.extend(self.img_to_latent(images[k]) for k in rgb_id)
        his_latent_gt = torch.stack(his_latent_gt).permute(1, 0, 2, 3, 4)
        indices = [-6, -4, -3, -2]
        his_latent_gt = his_latent_gt[:, indices]
        return his_latent_gt

    # ---------------- 预测 ----------------
    @torch.no_grad()
    def gen_latent(self, current_latent, his_latent_gt, text, actions,
                    num_inference_steps=None):
        """
        SVD 推理: 当前帧 latent + 历史 latent + 指令文本 + 动作序列 -> 未来 latent。

        支持批量: actions 可为 (N, 12, 7), N 为候选轨迹数;
        current_latent (1, 4, 32, 32) 与 his_latent_gt (1, 4, 4, 32, 32)
        会自动扩展到 N 份, 一次 UNet 前向同时预测 N 条轨迹的未来。
        Returns: (N, 8, 4, 32, 32)
        """
        current_latent = current_latent.to(device=self.device, dtype=self.dtype)
        his_latent_gt = his_latent_gt.to(device=self.device, dtype=self.dtype)
        actions = actions.to(device=self.device, dtype=self.dtype)
        args = self.args
        pipeline = self.svd
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        N = actions.shape[0]
        if current_latent.shape[0] == 1 and N > 1:
            current_latent = current_latent.expand(N, -1, -1, -1)
        if his_latent_gt.shape[0] == 1 and N > 1:
            his_latent_gt = his_latent_gt.expand(N, -1, -1, -1, -1)

        action_latent = self.action_encoder(
            actions, text, self.tokenizer, self.text_encoder,
            args.frame_level_cond)  # (N, 12, 1024); 文本嵌入 (1,1,1024) 自动广播

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            generator=generator,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=num_inference_steps if num_inference_steps is not None else args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type='latent',
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )

        pred_latents = einops.rearrange(
            pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=1, n=1)  # (N, 8, 4, 32, 32)
        return pred_latents.float()

    @torch.no_grad()
    def forward(self, images, trajectory, instruction, prev_actions=()):
        """
        Args:
            images: list[np.ndarray], RGB 观测序列 (H, W, 3) uint8,
                最后一帧为当前观测。episode 起始时(即 prev_actions 为空)
                应传入初始环视得到的多帧图片(至少 7 帧)。
            trajectory: 一段轨迹, 长度 >= 7 的动作序列, 取前 7 步。
            instruction: str, (子)指令文本。
            prev_actions: 上一段已执行的动作序列(长度 >= 4); 为空表示 episode 起始。
        Returns:
            pred_latents: torch.Tensor, (8, 4, 32, 32), 预测的未来 8 帧 latent。
        """
        trajectory = np.asarray(trajectory, dtype=int)
        assert len(trajectory) >= 7, f"轨迹长度不足 7: {len(trajectory)}"

        initial = len(prev_actions) == 0

        # 历史帧 latent
        his_latent_gt = self.build_history_latents(images, initial)  # (1, 4, 4, 32, 32)

        # 当前帧 latent
        current_latent = self.img_to_latent(images[-1]).float()  # (1, 4, 32, 32)

        # 动作序列
        actions = self.build_action_sequence(prev_actions, trajectory[:7])
        actions = actions.unsqueeze(0)  # (1, 12, 7)

        pred_latents = self.gen_latent(current_latent, his_latent_gt, instruction, actions)
        return pred_latents.squeeze(0)  # (8, 4, 32, 32)

    # ---------------- 可选: 解码可视化 ----------------
    @torch.no_grad()
    def decode_latents(self, pred_latents):
        """
        将预测 latent 解码为视频帧, 便于可视化调试。
        Args:
            pred_latents: (8, 4, 32, 32) 或 (B, 8, 4, 32, 32)
        Returns:
            np.ndarray, (B, 8, 256, 256, 3) uint8
        """
        if pred_latents.dim() == 4:
            pred_latents = pred_latents.unsqueeze(0)
        pred_latents = pred_latents.to(self.device)
        args = self.args
        pipeline = self.svd

        decoded_video = []
        bsz, frame_num = pred_latents.shape[:2]
        pred_latents = pred_latents.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, pred_latents.shape[0], args.decode_chunk_size):
            chunk = pred_latents[i:i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = videos.to(pipeline.unet.dtype).detach().cpu().numpy(
        ).transpose(0, 1, 3, 4, 2).astype(np.uint8)  # (B, 8, 256, 256, 3)
        return videos
