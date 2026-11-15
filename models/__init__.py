from .spatial_gcn import SpatialGCN
from .lstm import LSTM
from .temporal_cnn import TemporalCNN

MODELS = {
    "spatial_gcn": SpatialGCN,
    "lstm": LSTM,
    "temporal_cnn": TemporalCNN,
}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)