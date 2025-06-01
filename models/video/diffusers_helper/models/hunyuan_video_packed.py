from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import einops
import torch.nn as nn
import numpy as np

from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import logging
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers_helper.dit_common import LayerNorm
from diffusers_helper.utils import zero_module

# ... existing code ... 