# Copyright (c) OpenMMLab. All rights reserved.
from . import textrecog
from .builder import (BACKBONES, CONVERTORS, DECODERS, DETECTORS, ENCODERS,
                      HEADS, LOSSES, PREPROCESSOR, build_backbone,
                      build_convertor, build_decoder, build_detector,
                      build_encoder, build_loss, build_preprocessor)
from .textrecog import *  # NOQA

__all__ = [
    'BACKBONES', 'DETECTORS', 'LOSSES', 'build_backbone',
    'build_detector', 'build_loss', 'CONVERTORS', 'ENCODERS', 'DECODERS',
    'PREPROCESSOR', 'build_convertor', 'build_encoder', 'build_decoder',
    'build_preprocessor'
]
__all__ +=textrecog.__all__
