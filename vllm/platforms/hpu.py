import torch

from .interface import Platform, PlatformEnum


class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU
