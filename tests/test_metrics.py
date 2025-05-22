from sklearn.metrics import f1_score
from src.evaluate import evaluate_model

def test_evaluate_model():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    assert abs(evaluate_model(y_pred, y_true)
               - f1_score(y_true, y_pred, average="macro")) < 1e-9

