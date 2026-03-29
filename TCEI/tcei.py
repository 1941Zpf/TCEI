import math
import torch
import torch.nn.functional as F
import operator
import numpy as np

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def logits_process(id_logits):
    loss = softmax_entropy(id_logits)
    prob_map = id_logits.softmax(1)
    pred = int(id_logits.topk(1, 1, True, True)[1].t()[0])
    return loss, prob_map, pred

def get_entropy(loss, num_id):
    max_entropy = math.log2(num_id)
    return float(loss / max_entropy)

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

def compute_cache_logits(image_features, cache, alpha, beta, num_id, id_label_to_id, temperature, neg_mask_thresholds=None, is_wide=False):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        if is_wide == False:
            cache_keys = []
            cache_values = []
            for class_index in sorted(cache.keys()):
                comb_keys = []
                for item in cache[class_index]:
                    comb_keys.append(item[0])
                if neg_mask_thresholds:
                    matches = []
                    for val in item[2]:
                        key = next((k for k, v in id_label_to_id.items() if v == val), -1)
                        matches.append(key)
                    matches = torch.cat([torch.tensor(matches, dtype=torch.int64, device=image_features.device), torch.full((num_id - len(matches),), -1, dtype=torch.int64, device=image_features.device)])
                    cache_values.append(matches)
                else:
                    key = next((k for k, v in id_label_to_id.items() if v == class_index), -1)
                    cache_values.append(key)
                cache_keys.append(comb_keys)
            if neg_mask_thresholds:
                cache_values = torch.stack(cache_values, dim=0).to(device=image_features.device)
                mask = torch.zeros((cache_values.size(0), num_id), dtype=torch.float32, device=image_features.device)
                rows = torch.arange(cache_values.size(0)).unsqueeze(1).to(device=image_features.device)  # 行索引 [[0],[1]]
                valid_mask = cache_values != -1
                valid_mask = valid_mask.to(device=image_features.device)
                mask[rows, cache_values[valid_mask]] = 1.0
                cache_values = mask
            else:
                cache_values = torch.tensor(cache_values, dtype=torch.int64)
                mask = (cache_values == -1)
                cache_values = cache_values.clone()
                cache_values[mask] = 50
                one_hot = F.one_hot(cache_values, num_classes=num_id).float()
                one_hot[mask] = 0.0
                cache_values = one_hot.cuda()
            affinity = []
            for cache_key in cache_keys:
                cache_key = torch.cat(cache_key, dim=0).permute(1, 0)
                similarity = image_features @ cache_key
                similarity_mean = similarity.mean()
                affinity.append(similarity_mean)
            affinity = torch.stack(affinity, dim=0).unsqueeze(0)
            affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min())
            if temperature != 0:
                T = temperature
                affinity = F.softmax(affinity / T, dim=-1)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        elif is_wide == True:
            cache_keys = []
            cache_values = []
            for class_index in sorted(cache.keys()):
                for item in cache[class_index]:
                    cache_keys.append(item[0])
                    if neg_mask_thresholds:
                        cache_values.append(item[2])
                    else:
                        cache_values.append(class_index)
            cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0).to(device=image_features.device)
            if neg_mask_thresholds:
                cache_values = torch.cat(cache_values, dim=0)
                cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().float().to(image_features.device)
            else:
                cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64),num_classes=num_id)).cuda().float().to(image_features.device)
            affinity = image_features @ cache_keys
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

def fisher_inner(u, v, p):
    eps = 1e-8
    inner = 1 - torch.abs(u - v) / (torch.maximum(torch.abs(u), torch.abs(v)) + eps)
    return inner.clamp(0,1)

def fisher_orthogonalize(a, b, p):
    coef = fisher_inner(a, b, p) / (fisher_inner(a, a, p) + 1e-12)
    b_perp = b - coef * a
    return b_perp