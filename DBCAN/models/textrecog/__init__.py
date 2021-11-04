# Copyright (c) OpenMMLab. All rights reserved.
from . import (backbones, convertors, decoders, encoders, losses, recognizer)

from .backbones import *  # NOQA
from .convertors import *  # NOQA
from .decoders import *  # NOQA
from .encoders import *  # NOQA
from .losses import *  # NOQA
from .recognizer import *  # NOQA

__all__ = (
    backbones.__all__ + convertors.__all__ + decoders.__all__ +
    encoders.__all__  + losses.__all__ + recognizer.__all__)
