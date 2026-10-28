from .lstm import LSTM

# Interface: x [N, T, V, C], optional vis [N, T, V] -> logits [N, num_classes].
MODELS = {
    "lstm": LSTM,
}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)
