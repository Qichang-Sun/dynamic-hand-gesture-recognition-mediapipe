# only for test use!!
import numpy as np
from model import KeyPointClassifier, PointHistoryClassifier

# 1) KeyPointClassifier：构造 21*2 = 42 维假数据（例如全零或随机）
kpc = KeyPointClassifier()
dummy_landmarks = np.zeros(42, dtype=np.float32)  # 你的预处理后维度应匹配这里
try:
    idx = kpc(dummy_landmarks)
    print("KeyPointClassifier OK, result index:", idx)
except Exception as e:
    print("KeyPointClassifier ERROR:", e)

# 2) PointHistoryClassifier：构造 16*2 = 32 维假数据（按你的 history_length*2）
phc = PointHistoryClassifier(score_th=0.0)  # 设 0 阈值，便于看到 argmax
dummy_history = np.zeros(32, dtype=np.float32)
try:
    idx = phc(dummy_history)
    print("PointHistoryClassifier OK, result index:", idx)
except Exception as e:
    print("PointHistoryClassifier ERROR:", e)