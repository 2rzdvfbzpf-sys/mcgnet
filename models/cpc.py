"""Cosine Prototype Classifier (CPC)."""
from torch.nn import functional as F


def cosine_prototype_logits(support, query, eps):
    """Query-to-prototype cosine scores for support [way,shot,224] and query [count,224]."""
    if support.ndim != 3 or query.ndim != 2 or support.shape[-1] != query.shape[-1]:
        raise ValueError('Invalid support/query feature dimensions')
    prototypes = F.normalize(F.normalize(support, dim=-1, eps=eps).mean(dim=1),
                             dim=-1, eps=eps)
    return F.normalize(query, dim=-1, eps=eps) @ prototypes.t()


def episode_scores(model, dt, at, n_way, k_shot, q_query):
    """CPC scores for the query samples of an N-way K-shot episode."""

    if dt.shape[0] != n_way * (k_shot + q_query):
        raise ValueError('Episode batch does not match way/shot/query')
    embeddings = model(dt, at).reshape(n_way, k_shot + q_query, -1)
    support = embeddings[:, :k_shot]
    query = embeddings[:, k_shot:].reshape(n_way * q_query, -1)
    return model.scale * cosine_prototype_logits(support, query, model.config.normalization_eps)
