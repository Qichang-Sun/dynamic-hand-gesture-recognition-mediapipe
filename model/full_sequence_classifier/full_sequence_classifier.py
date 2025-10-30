"""
FullSequenceClassifier: TFLite inference wrapper for full hand-sequence classifier.
- Input: flat float32 array of length (TIME_STEPS * DIM_PER_FRAME)
- Output: (class_id, score, label)

Usage:
    clf = FullSequenceClassifier(
        model_path='model/full_sequence_classifier/full_sequence_classifier.tflite',
        label_path='model/full_sequence_classifier/full_sequence_classifier_label.csv',
        time_steps=16,
        dim_per_frame=42,
    )
    class_id, score, label = clf.infer(flat_sample)
"""
from __future__ import annotations
import numpy as np
import csv
# Try TF-Lite from TensorFlow first; fallback to tflite-runtime
try:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
except Exception:  # pragma: no cover
    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
        Interpreter = tflite.Interpreter  # type: ignore
    except Exception as e:
        raise ImportError('Neither tensorflow.lite nor tflite_runtime is available: %r' % e)

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

    def _load_labels(self, path: str):
        labels = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if not row:
                    continue
                labels.append(row[0])
        return labels

    def infer(self, flat_sample: np.ndarray):
        """
        flat_sample: shape (input_len,), dtype float32/float64
        returns: (score, label)
        """
        x = np.asarray(flat_sample, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.input_len:
            raise ValueError(f'Expected input_len={self.input_len}, got {x.shape[1]}')
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)[0]
        class_id = int(np.argmax(y))
        score = float(np.max(y))
        label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
        return score, label