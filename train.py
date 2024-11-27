import math
import time

from torch import nn, optim
from torch.optim import Adam

from models.model.transformer import Transformer



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)