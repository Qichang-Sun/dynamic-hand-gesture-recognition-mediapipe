import numpy as np
import csv
import os

# Try TF-Lite from TensorFlow first; fallback to tflite-runtime
try:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
except Exception:
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
        Interpreter = tflite.Interpreter  # type: ignore
    except Exception as e:
        raise ImportError(f"Neither tensorflow.lite nor tflite_runtime is available: {e}")

WRIST_IDX = 0
PALM_A = 5
PALM_B = 17
KP_NUM = 21

class FullSequenceClassifier:
    def __init__(self, model_path: str, label_path: str, time_steps: int, dim_per_frame: int):
        self.time_steps = time_steps
        self.dim_per_frame = dim_per_frame
        self.input_len = time_steps * dim_per_frame

        # Load labels
        self.labels = self._load_labels(label_path)

        # Load TFLite model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

        # Load normalization parameters
        mean_path = os.path.join(os.path.dirname(model_path), 'mean.npy')
        std_path = os.path.join(os.path.dirname(model_path), 'std.npy')
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.load(std_path).astype(np.float32)

    def _load_labels(self, path: str):
        labels = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if row:
                    labels.append(row[0])
        return labels

    def preprocess_add_feats(self, X_flat: np.ndarray):
        N = X_flat.shape[0]
        X_seq = X_flat.reshape(N, self.time_steps, KP_NUM, 2)

        wrist = X_seq[:, :, WRIST_IDX:WRIST_IDX+1, :]
        X_rel = X_seq - wrist

        p_a = X_seq[:, :, PALM_A, :]
        p_b = X_seq[:, :, PALM_B, :]
        palm_len = np.linalg.norm(p_a - p_b, axis=-1, keepdims=True)
        palm_len = np.maximum(palm_len, 1e-6)
        X_norm = X_rel / palm_len[:, :, None, :]

        pos = X_norm.reshape(N, self.time_steps, KP_NUM * 2)

        vel = np.zeros_like(pos)
        vel[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]

        acc = np.zeros_like(pos)
        acc[:, 2:, :] = vel[:, 2:, :] - vel[:, 1:-1, :]

        X_full = np.concatenate([pos, vel, acc], axis=-1)
        return X_full.astype(np.float32)

    def infer(self, flat_sample: np.ndarray):
        x_raw = flat_sample.reshape(1, self.time_steps * KP_NUM * 2)
        x_feat = self.preprocess_add_feats(x_raw)
        x_feat = (x_feat - self.mean) / self.std

        self.interpreter.set_tensor(self.input_index, x_feat)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)[0]

        probs = np.exp(y) / np.sum(np.exp(y))  # softmax
        class_id = int(np.argmax(probs))
        score = float(np.max(probs))
        label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
        return score, label