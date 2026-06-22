# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_aloha_3dcmp_cfg import va_aloha_3dcmp_cfg

va_aloha_3dcmp_server_cfg = EasyDict(__name__='Config: VA aloha 3dcmp server')
va_aloha_3dcmp_server_cfg.update(va_aloha_3dcmp_cfg)

va_aloha_3dcmp_server_cfg.transformer_path = '/mi/data2T/lijianwen/Checkpoints/lingbot-vggt-ckpts/no_freeze_norm/checkpoint_step_1000'
va_aloha_3dcmp_server_cfg.infer_mode = 'server'
