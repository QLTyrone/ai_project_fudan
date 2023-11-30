import torch

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum(vec):
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))