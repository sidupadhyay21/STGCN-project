from .spatial_gcn import SpatialGCN
from .lstm import LSTM
from .temporal_cnn import TemporalCNN
from .st_gcn import STGCN

MODELS = {
    "spatial_gcn": SpatialGCN,
    "lstm": LSTM,
    "temporal_cnn": TemporalCNN,
    "st_gcn": STGCN,
}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)