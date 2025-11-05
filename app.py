#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
from collections import Counter, deque
import types
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
import itertools

# ===== 已有的模块 =====
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
from model import FullSequenceClassifier

# ===== 新增：Tasks API =====
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,   # ← 旧代码是 int，改为 float 更合理
                        default=0.5)

    # 可选：自定义 HandLandmarker 模型路径
    parser.add_argument("--hand_task",
                        type=str,
                        default="hand_landmarker.task",
                        help="Path to hand_landmarker.task")
    parser.add_argument("--max_num_hands",
                        type=int,
                        default=1)

    return parser.parse_args()


# --- 小工具：用像素坐标列表计算手部矩形 (x1, y1, x2, y2) ---
def calc_brect_from_pixel_landmarks(img_shape, landmark_list_px):
    h, w = img_shape[:2]
    xs = [pt[0] for pt in landmark_list_px if pt is not None]
    ys = [pt[1] for pt in landmark_list_px if pt is not None]
    if not xs or not ys:
        return (0, 0, 0, 0)
    x1 = max(min(xs), 0)
    y1 = max(min(ys), 0)
    x2 = min(max(xs), w - 1)
    y2 = min(max(ys), h - 1)
    return (x1, y1, x2, y2)


# --- 小工具：把 Tasks 的 handedness 转成旧接口风格（有 .classification[0].label）---
def handedness_to_legacy(handedness_categories):
    """
    handedness_categories: List[Category] (Tasks 的一只手的候选列表)
    返回一个具备 .classification[0].label 的对象，兼容你旧的 draw_info_text(...)
    """
    label = handedness_categories[0].category_name if handedness_categories else "Unknown"
    score = handedness_categories[0].score if handedness_categories else 0.0
    # 构造旧风格：results.multi_handedness[i].classification[0].label / .score
    classification = [types.SimpleNamespace(label=label, score=score)]
    return types.SimpleNamespace(classification=classification)


def fourcc_from_cap(cap):
    v = int(cap.get(cv.CAP_PROP_FOURCC))
    v &= 0xFFFFFFFF  # 转成无符号 32 位
    # OpenCV 按小端序存储 FOURCC：最低字节是第一个字符
    fourcc = "".join(chr((v >> (8*i)) & 0xFF) for i in range(4))
    return fourcc, v


def main():
    # ========= 参数 =========
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    max_num_hands = args.max_num_hands
    hand_task_path = args.hand_task

    use_brect = True

    # ========= 摄像头 =========
    cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # cap.set(cv.CAP_PROP_FPS, 30)

    # ========= MediaPipe Tasks: HandLandmarker 初始化 =========
    running_mode = (mp_vision.RunningMode.IMAGE
                    if use_static_image_mode else mp_vision.RunningMode.VIDEO)

    base_options = mp_python.BaseOptions(model_asset_path=hand_task_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # ========= 训练好的分类器 =========
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    fullseq_classifier = FullSequenceClassifier(
        model_path='model/full_sequence_classifier/full_sequence_classifier_1105_3.tflite',
        label_path='model/full_sequence_classifier/full_sequence_classifier_label.csv',
        time_steps=16,
        dim_per_frame=42
    )


    # ========= 标签文件 =========
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    # ========= FPS / 历史序列 =========
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # ========= 全手点序列采集设置（2D）=========
    fullseq_dim_per_frame = 21 * 2       # 2D: 每帧 21 点 × (x,y) = 42
    fullseq_history_list = deque(maxlen=history_length)
    
    mode = 0
    timestamp_ms = 0  # VIDEO 模式需要单调递增时间戳（毫秒）

    while True:
        fps = cvFpsCalc.get()

        # 键盘控制
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # 读帧
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"Camera stream is not available")
            break
        # frame_bgr = cv.flip(frame_bgr, 1)
        debug_image = copy.deepcopy(frame_bgr)

        # fourcc_str, fourcc_val = fourcc_from_cap(cap)
        # print("FOURCC:", fourcc_str, " hex:", hex(fourcc_val))

        # 转成 MediaPipe Image（SRGB）
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 推理
        if running_mode == mp_vision.RunningMode.IMAGE:
            result = landmarker.detect(mp_image)
        else:
            # 时间戳：你也可以用 time.monotonic()*1000
            timestamp_ms += int(1000 / max(1, cap.get(cv.CAP_PROP_FPS) or 30))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 处理结果
        if result.hand_landmarks:
            # hand_landmarks: List[List[Landmark]] (每只手21点；x/y/z 为归一化坐标)
            # handedness    : List[List[Category]]（每只手的左右手候选）
            for hand_idx in range(len(result.hand_landmarks)):
                landmarks = result.hand_landmarks[hand_idx]
                handedness_categories = result.handedness[hand_idx]

                # 归一化坐标 → 像素坐标（与旧的 landmark_list 结构一致）
                h, w = frame_bgr.shape[:2]
                landmark_list_px = [[int(lm.x * w), int(lm.y * h)] for lm in landmarks]

                # 计算边框（像素）
                brect = calc_brect_from_pixel_landmarks(frame_bgr.shape, landmark_list_px)

                # === 原有的预处理 ===
                pre_processed_landmark_list = pre_process_landmark(landmark_list_px)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                fullseq_history_list.append(pre_processed_landmark_list)

                # 记录数据（如果启用了采集）
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)
                # >>> 记录全手序列（mode=3，按 0~9 落盘到 full_sequence.csv）
                logging_fullseq_csv(number, mode, fullseq_history_list, history_length, fullseq_dim_per_frame)

                # === 手势分类（自定义）===
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # 比如“指点”手势（与原来的逻辑一致）
                    # index finger tip = 8
                    point_history.append(landmark_list_px[8])
                else:
                    point_history.append([0, 0])
                    fullseq_history_list.append([0.0] * fullseq_dim_per_frame)

                # 动作（轨迹）分类
                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # === 全手时序分类（当缓冲满16帧时）===
                if len(fullseq_history_list) == 16:
                    flat = list(itertools.chain.from_iterable(list(fullseq_history_list)))
                    score, label = fullseq_classifier.infer(np.array(flat, dtype=np.float32))
                else:
                    score, label = None, None

                # === 绘制 ===
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list_px)

                # handedness 适配为旧风格对象，避免改 draw_info_text(...)
                legacy_handed = handedness_to_legacy(handedness_categories)

                # legacy draw text for point history
                # debug_image = draw_info_text(
                #     debug_image,
                #     brect,
                #     legacy_handed,
                #     keypoint_classifier_labels[hand_sign_id],
                #     point_history_classifier_labels[most_common_fg_id[0][0]],
                # )
                    
                # 1) buffer 状态
                buf_txt = f'FullSeq buf: {len(fullseq_history_list)}/16'
                cv.putText(debug_image, buf_txt, (10, 140), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv.LINE_AA)
                cv.putText(debug_image, buf_txt, (10, 140), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv.LINE_AA)

                # 2) 有无手
                has_hand = (result.hand_landmarks is not None and len(result.hand_landmarks) > 0)
                cv.putText(debug_image, f'Hand: {"YES" if has_hand else "NO"}', (10, 160),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv.LINE_AA)
                cv.putText(debug_image, f'Hand: {"YES" if has_hand else "NO"}', (10, 160),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if has_hand else (200,200,200), 2, cv.LINE_AA)

                # 3) 预测结果（只有满 16 帧才显示）
                if label is not None:
                    txt = f'FullSeq: {label} ({score:.2f})'
                    cv.putText(debug_image, txt, (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv.LINE_AA)
                    cv.putText(debug_image, txt, (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv.LINE_AA)

        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 102:  # f
        mode = 3
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def logging_fullseq_csv(number, mode, fullseq_sequence, time_steps, dim_per_frame):
    """
    当 mode == 3 且按下 0~9 时，把最近 T 帧 x 每帧维度 的展平向量写入 CSV
    """
    csv_path = 'model/full_sequence_classifier/full_sequence.csv'
    if mode == 3 and (0 <= number <= 9):
        if len(fullseq_sequence) == time_steps:
            # 展平: T × D -> 1D
            flat = list(itertools.chain.from_iterable(list(fullseq_sequence)))
            if len(flat) == time_steps * dim_per_frame:
                with open(csv_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([number, *flat])
                print(f'[FullSeq] Saved class={number}, steps={time_steps}, dim_per_frame={dim_per_frame}')
            else:
                print(f'[FullSeq] length mismatch: {len(flat)} != {time_steps * dim_per_frame}')
        else:
            print(f'[FullSeq] buffer not full: {len(fullseq_sequence)}/{time_steps}')

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (255, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History', 'Logging Full Hand Seq']
    if 1 <= mode <= 3:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
