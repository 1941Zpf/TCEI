
import copy

import torch
import einops
from scipy.optimize import linear_sum_assignment

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh
from models.misc import get_model

from TCEI.tcei import *


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
            # TCEI
            pos_enabled: bool = True,
            pos_shot_capacity: int = 3,
            pos_alpha: float = 1.0,
            pos_beta: float = 5.0,
            neg_enabled: bool = True,
            neg_shot_capacity: int = 2,
            neg_alpha: float = 0.117,
            neg_beta: float = 1.0,
            neg_entropy_threshold_lower: float = 0.2,
            neg_entropy_threshold_upper: float = 0.5,
            neg_mask_threshold_lower: float = 0.03,
            neg_mask_threshold_upper: float = 1.0,
            pos_temperature: float = 0,
            neg_temperature: float = 0.01,
            # TCEI-WIDE
            w_pos_enabled: bool = True,
            w_pos_shot_capacity: int = 3,
            w_pos_alpha: float = 1.0,
            w_pos_beta: float = 5.0,
            w_neg_enabled: bool = True,
            w_neg_shot_capacity: int = 2,
            w_neg_alpha: float = 0.117,
            w_neg_beta: float = 1.0,
            w_neg_entropy_threshold_lower: float = 0.2,
            w_neg_entropy_threshold_upper: float = 0.5,
            w_neg_mask_threshold_lower: float = 0.03,
            w_neg_mask_threshold_upper: float = 1.0,
            w_pos_temperature: float = 0,
            w_neg_temperature: float = 0.01,
            use_cache: str = "object_and_id",
            lambda_object: float = 1.0,
            lambda_id: float = 1.0,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.num_id_vocabulary = get_model(model).num_id_vocabulary

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, 256), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )
        # self.trajectory_features = torch.zeros(())

        self.current_track_results = {}

        # TCEI
        self.pos_enabled = pos_enabled
        self.pos_shot_capacity = pos_shot_capacity
        self.pos_alpha = pos_alpha
        self.pos_beta = pos_beta
        self.neg_enabled = neg_enabled
        self.neg_shot_capacity = neg_shot_capacity
        self.neg_alpha = neg_alpha
        self.neg_beta = neg_beta
        self.neg_entropy_threshold_lower = neg_entropy_threshold_lower
        self.neg_entropy_threshold_upper = neg_entropy_threshold_upper
        self.neg_mask_threshold_lower = neg_mask_threshold_lower
        self.neg_mask_threshold_upper = neg_mask_threshold_upper
        self.pos_temperature = pos_temperature
        self.neg_temperature = neg_temperature
        # TCEI-WIDE
        self.w_pos_enabled = w_pos_enabled
        self.w_pos_shot_capacity = w_pos_shot_capacity
        self.w_pos_alpha = w_pos_alpha
        self.w_pos_beta = w_pos_beta
        self.w_neg_enabled = w_neg_enabled
        self.w_neg_shot_capacity = w_neg_shot_capacity
        self.w_neg_alpha = w_neg_alpha
        self.w_neg_beta = w_neg_beta
        self.w_neg_entropy_threshold_lower = w_neg_entropy_threshold_lower
        self.w_neg_entropy_threshold_upper = w_neg_entropy_threshold_upper
        self.w_neg_mask_threshold_lower = w_neg_mask_threshold_lower
        self.w_neg_mask_threshold_upper = w_neg_mask_threshold_upper
        self.w_pos_temperature = w_pos_temperature
        self.w_neg_temperature = w_neg_temperature
        self.lambda_object = lambda_object
        self.use_cache = use_cache
        self.lambda_id = lambda_id
        return

    @torch.no_grad()
    # def update(self, image):
    def update(self, image, pos_cache, neg_cache, w_pos_cache, w_neg_cache):
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = self._get_activate_detections(detr_out=detr_out)
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            id_pred_labels, id_logits, after_id_decoder_features = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds, pos_cache=pos_cache, neg_cache=neg_cache, w_pos_cache=w_pos_cache, w_neg_cache=w_neg_cache, scores=scores)
        # Filter out illegal newborn detections:
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]

        # A hack implementation, before assign new id labels, update the id_queue to ensure the uniqueness of id labels:
        n_activate_id_labels = 0
        n_newborn_targets = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn_targets += 1

        # Make sure the length of newborn instances is less than the length of remaining IDs:
        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_keep_idxs = torch.ones(len(newborn_idxs), dtype=torch.bool, device=newborn_idxs.device)
            newborn_keep_idxs[n_remaining_ids:] = False
            keep_idxs[newborn_idxs] = newborn_keep_idxs
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]
        pass

        # Assign new id labels:
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels, id_logits=id_logits, output_embeds=output_embeds, after_id_decoder_features=after_id_decoder_features, pos_cache=pos_cache, neg_cache=neg_cache)

        if len(torch.unique(id_labels)) != len(id_labels):
            print(id_labels, id_labels.shape)
            exit(-1)

        # Update the results:
        self.current_track_results = {
            "score": scores,
            "category": categories,
            # "bbox": boxes * self.bbox_unnorm,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": torch.tensor(
                [self.id_label_to_id[_] for _ in id_labels.tolist()], dtype=torch.int64,
            ),
        }

        # Update id_queue:
        for _ in range(len(id_labels)):
            self.id_queue.add(id_labels[_].item())

        # Update trajectory infos:
        self._update_trajectory_infos(boxes=boxes, output_embeds=output_embeds, id_labels=id_labels)

        # Filter out inactive tracks:
        self._filter_out_inactive_tracks(pos_cache=pos_cache, neg_cache=neg_cache)
        pass
        return

    def get_track_results(self):
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        # Selecting:
        # logits = logits[activate_indices]
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds

    # def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor:
    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor, pos_cache: dict, neg_cache: dict, w_pos_cache, w_neg_cache, scores):
        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device), None, output_embeds.clone()
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            # 3. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _, after_id_decoder_features, after_id_decoder_id_embeds = self.model(seq_info=seq_info, part="id_decoder")
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]

            copy_id_logits = copy.deepcopy(id_logits)


            # For TCEI
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                # case "object-priority": id_labels = self._object_priority_assignment(id_scores=id_scores)
                case _: raise NotImplementedError
            id_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())

            for i in range(id_logits.shape[0]):
                logits = id_logits[i].unsqueeze(0).clone()
                loss, prob_map, _ = logits_process(logits)
                pred = id_labels[i].item()
                features = after_id_decoder_id_embeds[i].unsqueeze(0).clone()
                features /= features.norm(dim=-1, keepdim=True)
                prop_entropy = get_entropy(loss, (self.num_id_vocabulary + 1))

                if self.w_pos_enabled:
                    update_cache(w_pos_cache, pred, [features, loss], self.w_pos_shot_capacity)
                if self.w_neg_enabled and self.w_neg_entropy_threshold_lower < prop_entropy < self.w_neg_entropy_threshold_upper:
                    update_cache(w_neg_cache, pred, [features, loss, prob_map], self.w_neg_shot_capacity, True)

                final_logits_id = logits.clone()
                if self.w_pos_enabled and w_pos_cache:
                    final_logits_id += compute_cache_logits(features, w_pos_cache, self.w_pos_alpha, self.w_pos_beta, (self.num_id_vocabulary + 1), self.id_label_to_id, self.w_pos_temperature, is_wide=True)
                if self.w_neg_enabled and w_neg_cache:
                    final_logits_id -= compute_cache_logits(features, w_neg_cache, self.w_neg_alpha, self.w_neg_beta, (self.num_id_vocabulary + 1), self.id_label_to_id, self.w_neg_temperature,(self.w_neg_mask_threshold_lower, self.w_neg_mask_threshold_upper), is_wide=True)

                logits = id_logits[i].unsqueeze(0).clone()
                loss, prob_map, _ = logits_process(logits)
                pred = id_labels[i].item()
                if pred != self.num_id_vocabulary:
                    pred = self.id_label_to_id[pred]
                    prob_map = prob_map[0][:len(id_labels)] # 这里保留的长度应该是不对的，但是暂时不是很重要所以没处理
                    mask = (prob_map > self.neg_mask_threshold_lower) & (prob_map < self.neg_mask_threshold_upper)
                    indices = torch.where(mask)[0].tolist()
                    prob_map = id_labels[indices]
                    prob_map = prob_map[prob_map != self.num_id_vocabulary]
                    prob_map = [self.id_label_to_id[_] for _ in prob_map.tolist()]
                    features = output_embeds[i].unsqueeze(0).clone()
                    features /= features.norm(dim=-1, keepdim=True)
                    prop_entropy = get_entropy(loss, (self.num_id_vocabulary + 1))

                    if self.pos_enabled:
                        update_cache(pos_cache, pred, [features, loss], self.pos_shot_capacity)
                    if self.neg_enabled and self.neg_entropy_threshold_lower < prop_entropy < self.neg_entropy_threshold_upper:
                        update_cache(neg_cache, pred, [features, loss, prob_map], self.neg_shot_capacity, True)

                    final_logits_oj = logits.clone()
                    if self.pos_enabled and pos_cache:
                        final_logits_oj += compute_cache_logits(features, pos_cache, self.pos_alpha, self.pos_beta, (self.num_id_vocabulary + 1), self.id_label_to_id, self.pos_temperature)
                    if self.neg_enabled and neg_cache:
                        final_logits_oj -= compute_cache_logits(features, neg_cache, self.neg_alpha, self.neg_beta, (self.num_id_vocabulary + 1), self.id_label_to_id, self.neg_temperature, (self.neg_mask_threshold_lower, self.neg_mask_threshold_upper))

                    if self.use_cache == "object_and_id":
                        l_oj = final_logits_oj - logits
                        l_id = final_logits_id - logits
                        p = logits.softmax(dim=-1)
                        l_id_perp = fisher_orthogonalize(l_oj, l_id, p)
                        u = p * (1. - p)
                        l_id_perp *= u
                        logits_new = logits + l_oj + self.lambda_id * l_id_perp
                        id_logits[i] = logits_new.squeeze(0)
                    elif self.use_cache == "id":
                        id_logits[i] = final_logits_id.squeeze(0)
                    elif self.use_cache == "object":
                        id_logits[i] = final_logits_oj.squeeze(0)



            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 5. assign id labels:
            # Different assignment protocols:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                # case "object-priority": id_labels = self._object_priority_assignment(id_scores=id_scores)
                case _: raise NotImplementedError


            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())

            return id_pred_labels, id_logits, after_id_decoder_features

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor, id_logits, output_embeds, after_id_decoder_features, pos_cache, neg_cache):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            newborn_id_labels = torch.tensor(
                list(self.id_queue)[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
            )
            for _ in range(len(newborn_id_labels)):
                if self.trajectory_id_labels.shape[0] > 0:
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                if newborn_id_labels[_].item() in self.id_label_to_id:
                    self.id_label_to_id.pop(newborn_id_labels[_].item())
            # remove from trajectory infos:
            self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
            self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
            self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
            self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
            self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]


            # For TCEI
            idxs = torch.where(pred_id_labels == self.num_id_vocabulary)[0]
            for i in range(len(idxs)):
                idx = idxs[i].item()
                # new_id = newborn_id_labels[i].item()
                new_id = self.next_id + i
                if id_logits is not None:
                    logit = id_logits[idx].unsqueeze(0).clone()
                else:
                    logit = torch.full((1, self.num_id_vocabulary), 1.0 / self.num_id_vocabulary, dtype=torch.float32, device=output_embeds.device)
                loss, _, _ = logits_process(logit)
                pred = new_id
                features = output_embeds[idx].unsqueeze(0).clone()
                features /= features.norm(dim=-1, keepdim=True)
                if self.pos_enabled:
                    update_cache(pos_cache, pred, [features, loss], self.pos_shot_capacity)


            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.next_id += 1

            return pred_id_labels

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor):
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels = _id_labels - already_id_labels
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels) > 0:
            newborn_id_labels = torch.tensor(list(newborn_id_labels), dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0]
            _N = len(newborn_id_labels)
            _id_labels = einops.repeat(newborn_id_labels, 'n -> t n', t=_T)
            _boxes = torch.zeros((_T, _N, 4), dtype=self.dtype, device=distributed_device())
            _times = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N,
            )
            _features = torch.zeros(
                (_T, _N, 256), dtype=self.dtype, device=distributed_device(),
            )
            _masks = torch.ones((_T, _N), dtype=torch.bool, device=distributed_device())
            # 3.1. padding to trajectory infos:
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _id_labels], dim=1)
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, _boxes], dim=1)
            self.trajectory_times = torch.cat([self.trajectory_times, _times], dim=1)
            self.trajectory_features = torch.cat([self.trajectory_features, _features], dim=1)
            self.trajectory_masks = torch.cat([self.trajectory_masks, _masks], dim=1)
        # 4. update trajectory infos:
        _N = self.trajectory_id_labels.shape[1]
        current_id_labels = self.trajectory_id_labels[0] if self.trajectory_id_labels.shape[0] > 0 else id_labels
        current_features = torch.zeros((_N, 256), dtype=self.dtype, device=distributed_device())
        current_boxes = torch.zeros((_N, 4), dtype=self.dtype, device=distributed_device())
        current_times = self.trajectory_id_labels.shape[0] * torch.ones((_N,), dtype=torch.int64, device=distributed_device())
        current_masks = torch.ones((_N,), dtype=torch.bool, device=distributed_device())
        # 4.1. find out the same id labels (matching):
        indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(as_tuple=False)
        current_idxs = indices[:, 0]
        idxs = indices[:, 1]
        # 4.2. fill in the infos:
        current_id_labels[current_idxs] = id_labels[idxs]
        current_features[current_idxs] = output_embeds[idxs]
        current_boxes[current_idxs] = boxes[idxs]
        current_masks[current_idxs] = False
        # 4.3. cat to trajectory infos:
        self.trajectory_features = torch.cat([self.trajectory_features, current_features[None, ...]], dim=0).contiguous()
        self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes[None, ...]], dim=0).contiguous()
        self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels[None, ...]], dim=0).contiguous()
        self.trajectory_times = torch.cat([self.trajectory_times, current_times[None, ...]], dim=0).contiguous()
        self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks[None, ...]], dim=0).contiguous()
        # 4.4. a hack implementation to fix "times":
        self.trajectory_times = einops.repeat(
            torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
            't -> t n', n=self.trajectory_times.shape[1],
        ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self, pos_cache, neg_cache):
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        if len(id_scores) > 1:
            id_scores_newborn_repeat = id_scores[:, -1:].repeat(1, len(id_scores) - 1)
            id_scores = torch.cat((id_scores, id_scores_newborn_repeat), dim=-1)
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())
        match_rows, match_cols = linear_sum_assignment(1 - id_scores.cpu())
        for _ in range(len(match_rows)):
            _id = match_cols[_]
            if _id not in trajectory_id_labels_set:
                id_labels.append(self.num_id_vocabulary)
            elif _id >= self.num_id_vocabulary:
                id_labels.append(self.num_id_vocabulary)
            elif id_scores[match_rows[_], _id] < self.id_thresh:
                id_labels.append(self.num_id_vocabulary)
            else:
                id_labels.append(_id)
        return id_labels

    def _object_max_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        object_max_confs, object_max_id_labels = torch.max(id_scores, dim=-1)   # get the target ID labels and confs
        # Get the max confs of each ID label:
        id_max_confs = dict()
        for conf, id_label in zip(object_max_confs.tolist(), object_max_id_labels.tolist()):
            if id_label not in id_max_confs:
                id_max_confs[id_label] = conf
            else:
                # if conf == id_max_confs[id_label]:  # a very rare case
                #     conf = conf - 0.0001
                id_max_confs[id_label] = max(id_max_confs[id_label], conf)
        if self.num_id_vocabulary in id_max_confs:
            id_max_confs[self.num_id_vocabulary] = 0.0  # special token

        # Assign ID labels:
        for _ in range(len(object_max_id_labels)):
            if object_max_id_labels[_].item() not in trajectory_id_labels_set:         # not in tracked IDs -> newborn
                id_labels.append(self.num_id_vocabulary)
            else:
                _id_label = object_max_id_labels[_].item()
                _conf = object_max_confs[_].item()
                if _conf < self.id_thresh or _conf < id_max_confs[_id_label]:  # low conf or not the max conf -> newborn
                    id_labels.append(self.num_id_vocabulary)
                elif _id_label in id_labels:
                    id_labels.append(self.num_id_vocabulary)
                else:                                                          # normal case
                    id_labels.append(_id_label)

        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = [self.num_id_vocabulary] * len(id_scores)  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        id_max_confs, id_max_obj_idxs = torch.max(id_scores, dim=0)
        # Get the max confs of each object:
        object_max_confs = dict()
        for conf, object_idx in zip(id_max_confs.tolist(), id_max_obj_idxs.tolist()):
            if object_idx not in object_max_confs:
                object_max_confs[object_idx] = conf
            else:
                if conf == object_max_confs[object_idx]:    # a very rare case
                    conf = conf - 0.0001
                object_max_confs[object_idx] = max(object_max_confs[object_idx], conf)

        # Assign ID labels:
        for _ in range(len(id_max_obj_idxs)):
            _obj_idx, _id_label, _conf = id_max_obj_idxs[_].item(), _, id_max_confs[_].item()
            if _conf < self.id_thresh or _conf < object_max_confs[_obj_idx]:
                pass
            elif _id_label not in trajectory_id_labels_set:
                pass
            else:
                id_labels[_obj_idx] = _id_label

        return id_labels
