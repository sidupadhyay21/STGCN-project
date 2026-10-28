from .spatial_gcn import SpatialGCN

# Every model takes x [N, T, V, C] (and optionally vis [N, T, V]) and returns logits [N, num_classes].
MODELS = {
    "spatial_gcn": SpatialGCN,
}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)
