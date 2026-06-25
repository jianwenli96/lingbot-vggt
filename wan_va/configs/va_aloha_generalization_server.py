# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_aloha_generalization_cfg import va_aloha_generalization_cfg

va_aloha_generalization_server_cfg = EasyDict(__name__='Config: VA aloha generalization server')
va_aloha_generalization_server_cfg.update(va_aloha_generalization_cfg)

va_aloha_generalization_server_cfg.transformer_path = '/path/to/finetune/transformer'
va_aloha_generalization_server_cfg.infer_mode = 'server'
