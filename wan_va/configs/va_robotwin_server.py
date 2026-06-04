# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_server_cfg = EasyDict(__name__='Config: VA robotwin server')
va_robotwin_server_cfg.update(va_robotwin_cfg)

va_robotwin_server_cfg.transformer_path = '/path/to/finetune/transformer'
va_robotwin_server_cfg.infer_mode = 'server'