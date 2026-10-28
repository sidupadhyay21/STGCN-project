from .spatial_gcn import SpatialGCN
from .lstm import LSTM

MODELS = {
    "spatial_gcn": SpatialGCN,
    "lstm": LSTM,
}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)