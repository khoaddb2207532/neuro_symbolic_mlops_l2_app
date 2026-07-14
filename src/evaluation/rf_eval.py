"""Đánh giá Random Forest so với CNN trên cùng tập dữ liệu."""
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
import torch
from scipy.special import softmax


def evaluate_rf(rf_model, X, y_true, cnn_logits=None, class_names=None):
    """
    Tính các chỉ số cho RF và so sánh với CNN nếu có logits.

    Args:
        rf_model: RandomForestClassifier đã train
        X: np.ndarray, features
        y_true: np.ndarray, labels
        cnn_logits: np.ndarray hoặc None, logits từ CNN trên cùng tập
        class_names: list (tuỳ chọn)

    Returns:
        dict: chứa các metric
    """
    y_pred = rf_model.predict(X)
    y_proba = rf_model.predict_proba(X)  # shape (n, n_classes)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    conf_mat = confusion_matrix(y_true, y_pred)

    # Entropy trung bình của RF
    eps = 1e-12
    entropy_rf = -np.sum(y_proba * np.log(y_proba + eps), axis=1).mean()

    results = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_mat.tolist(),
        'avg_entropy_rf': entropy_rf,
        'n_samples': len(y_true),
    }

    if cnn_logits is not None:
        cnn_proba = softmax(cnn_logits, axis=1)
        cnn_preds = np.argmax(cnn_logits, axis=1)
        cnn_acc = accuracy_score(y_true, cnn_preds)
        cnn_entropy = -np.sum(cnn_proba * np.log(cnn_proba + eps), axis=1).mean()

        disagreement = np.mean(y_pred != cnn_preds)
        # Tỷ lệ mẫu mà RF đúng còn CNN sai (có thể dùng để đánh giá bổ sung)
        rf_correct_cnn_wrong = np.mean((y_pred == y_true) & (cnn_preds != y_true))
        cnn_correct_rf_wrong = np.mean((cnn_preds == y_true) & (y_pred != y_true))

        results.update({
            'cnn_accuracy': cnn_acc,
            'cnn_avg_entropy': cnn_entropy,
            'disagreement_rate': disagreement,
            'rf_correct_cnn_wrong': rf_correct_cnn_wrong,
            'cnn_correct_rf_wrong': cnn_correct_rf_wrong,
        })

    return results


def _convert_to_native(obj):
    """Chuyển đổi đệ quy numpy types sang Python native."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, list):
        return [_convert_to_native(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    else:
        return obj

def save_rf_evaluation(results, save_path):
    """Lưu kết quả đánh giá dạng JSON (xử lý numpy types)."""
    converted = _convert_to_native(results)
    with open(save_path, 'w') as f:
        json.dump(converted, f, indent=2)