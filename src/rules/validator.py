"""Lọc luật dựa trên support/confidence, vector hoá trên GPU."""
from typing import Dict, List

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.rules.extractor import RuleExtractor
from src.rules.rule_types import Rule, RuleSet


class GPUFastRuleValidator:
    def __init__(
        self,
        min_supp: float = 0.01,
        min_conf: float = 0.1,
        rule_batch_size: int = 2000,
        data_batch_size: int = 20000,
    ):
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.rule_batch_size = rule_batch_size
        self.data_batch_size = data_batch_size

    @staticmethod
    def _build_rule_tensors(batch_rules: List[Rule], device):
        B = len(batch_rules)
        max_conds = max((len(r.conditions) for r in batch_rules), default=1)
        feat_idx = torch.zeros(B, max_conds, dtype=torch.long, device=device)
        thresholds = torch.zeros(B, max_conds, dtype=torch.float32, device=device)
        ops = torch.zeros(B, max_conds, dtype=torch.bool, device=device)
        valid_m = torch.zeros(B, max_conds, dtype=torch.bool, device=device)
        targets = torch.tensor([r.target_class for r in batch_rules], device=device)
        for j, rule in enumerate(batch_rules):
            for k, cond in enumerate(rule.conditions):
                feat_idx[j, k] = cond.feature_index
                thresholds[j, k] = cond.threshold
                ops[j, k] = cond.operator == ">"
                valid_m[j, k] = True
        return feat_idx, thresholds, ops, valid_m, targets

    def validate(self, rule_set: RuleSet, val_features: torch.Tensor, val_labels: torch.Tensor) -> RuleSet:
        device = val_features.device
        N = val_features.size(0)
        val_labels = val_labels.view(-1)
        filtered: List[Rule] = []

        for i in tqdm(range(0, len(rule_set.rules), self.rule_batch_size), desc="GPU Validation"):
            batch_rules = rule_set.rules[i : i + self.rule_batch_size]
            B = len(batch_rules)
            feat_idx, thresholds, ops, valid_m, targets = self._build_rule_tensors(batch_rules, device)

            total_supp = torch.zeros(B, dtype=torch.long, device=device)
            total_corr = torch.zeros(B, dtype=torch.long, device=device)

            for d_start in range(0, N, self.data_batch_size):
                d_end = min(N, d_start + self.data_batch_size)
                feat_chunk = val_features[d_start:d_end]
                lbl_chunk = val_labels[d_start:d_end]

                sel = feat_chunk[:, feat_idx]
                cond_ok = ((sel <= thresholds) & ~ops) | ((sel > thresholds) & ops)
                cond_ok = cond_ok | ~valid_m
                rule_mask = cond_ok.all(dim=-1)

                total_supp += rule_mask.sum(dim=0)
                correct = rule_mask & (lbl_chunk.unsqueeze(1) == targets.unsqueeze(0))
                total_corr += correct.sum(dim=0)

            supp_ratio = total_supp.float() / N
            confs = torch.zeros(B, device=device)
            valid_mask = total_supp > 0
            confs[valid_mask] = total_corr[valid_mask].float() / total_supp[valid_mask].float()
            keep = (supp_ratio >= self.min_supp) & (confs >= self.min_conf)

            for idx in torch.where(keep)[0].cpu().tolist():
                rule = batch_rules[idx]
                rule.confidence = confs[idx].item()
                filtered.append(rule)

        return RuleSet(rules=filtered)

    def validate_crossval(self, features: torch.Tensor, labels: torch.Tensor, n_folds: int = 5) -> RuleSet:
        device = features.device
        X, y = features.cpu().numpy(), labels.cpu().numpy()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        all_valid: List[Rule] = []
        for train_idx, val_idx in kf.split(X):
            X_val = torch.from_numpy(X[val_idx]).float().to(device)
            y_val = torch.from_numpy(y[val_idx]).long().to(device)
            rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
            rf.fit(X[train_idx], y[train_idx])
            raw_fold = RuleExtractor().extract(rf)
            valid_fold = self.validate(raw_fold, X_val, y_val)
            all_valid.extend(valid_fold.rules)

        return RuleSet(rules=self._deduplicate(all_valid))

    @staticmethod
    def _deduplicate(rules: List[Rule]) -> List[Rule]:
        unique: Dict[tuple, Rule] = {}
        for r in rules:
            key = (
                tuple((c.feature_index, c.operator, round(c.threshold, 6)) for c in r.conditions),
                r.target_class,
            )
            if key not in unique:
                unique[key] = r
        return list(unique.values())
