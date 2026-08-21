from dataclasses import dataclass

@dataclass
class wm_args:
    """
    仅保留 svd.py (VideoPredictor) 推理链路用到的参数:
      - svd.py 直接使用: ckpt_path, width, height, num_frames, num_inference_steps,
        decode_chunk_size, guidance_scale, fps, motion_bucket_id,
        frame_level_cond, his_cond_zero
      - CrtlWorld(args) (svd_v10.py) 使用: svd_model_path, clip_model_path,
        action_dim, num_history, num_frames, text_cond, his_cond_zero,
        motion_bucket_id, fps, frame_level_cond
    """
    # model paths
    svd_model_path = "/mnt/data0/chenghao/vln-vpp/video-prediction-policy/pretrained/svd"
    clip_model_path = "/mnt/data0/chenghao/vln-vpp/video-prediction-policy/clip-vit-base-patch32"
    ckpt_path = "/home/chenghao/VLN-CE-mono-main/checkpoint-95000.pt"

    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 7.5
    # 30
    num_inference_steps = 15
    train_num_inference_steps = 10  # 训练时用更少步数, 评测保持15步
    eval_num_inference_steps = 15  # 评测时保持15步
    decode_chunk_size = 7
    width = 256
    height = 256
    # num history and num future predictions
    num_frames = 8
    num_history = 4
    action_dim = 7
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
