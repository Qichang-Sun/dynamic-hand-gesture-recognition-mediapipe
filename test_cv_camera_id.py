import cv2
import time
from collections import deque

def test_cameras(max_tested=5, show_seconds=5, target_width=None, target_height=None, backend=cv2.CAP_DSHOW):
    """
    遍历并测试摄像头索引，显示实时画面与 FPS 叠加，并打印统计信息。

    参数:
        max_tested (int): 最大测试的摄像头索引数量（从 0 开始到 max_tested-1）
        show_seconds (int): 每个摄像头窗口显示的秒数；也可按 'q' 提前退出该摄像头
        target_width (int|None): 指定分辨率宽（可选）
        target_height (int|None): 指定分辨率高（可选）
        backend (int|None): OpenCV 后端（Windows 可用 cv2.CAP_DSHOW；Linux 可尝试 cv2.CAP_V4L2；None 为默认）
    """
    print(f"准备测试 0 ~ {max_tested-1} 的摄像头索引...\n"
          f"提示：按 'q' 可提前退出当前摄像头显示。")

    for i in range(max_tested):
        print(f"\n=== 测试 camera index {i} ===")
        # 打开摄像头
        cap = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"[X] camera index {i} 不可用。")
            continue

        # 可选：设置期望分辨率（不同设备/驱动可能忽略）
        if target_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        if target_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

        # 尝试读取设备自报 FPS（可能不准确或为 0）
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        if reported_fps == 0 or reported_fps is None:
            reported_fps_str = "未知/不可靠"
        else:
            reported_fps_str = f"{reported_fps:.2f}"

        print(f"[i] 设备自报 FPS（CAP_PROP_FPS）：{reported_fps_str}")

        # 读取若干帧来估算真实 FPS
        start_time = time.time()
        frame_count = 0
        fps_deque = deque(maxlen=30)  # 用于平滑显示的 FPS

        window_name = f'Camera {i}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # 统计时长
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[!] camera index {i} 已打开但没有读到有效帧。")
                break

            frame_count += 1
            now = time.time()
            elapsed = now - start_time
            # 估算 FPS（用累积帧数/耗时）
            inst_fps = frame_count / elapsed if elapsed > 0 else 0.0
            fps_deque.append(inst_fps)
            # 平滑 FPS（取最近若干帧均值）
            smooth_fps = sum(fps_deque) / len(fps_deque)

            # 叠加文本
            overlay = frame.copy()
            text1 = f"Index: {i}"
            text2 = f"FPS(est): {smooth_fps:.2f}"
            text3 = f"FPS(reported): {reported_fps_str}"
            y0, dy = 28, 28
            for idx, t in enumerate([text1, text2, text3]):
                y = y0 + idx * dy
                cv2.putText(overlay, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow(window_name, overlay)

            # 按 'q' 退出当前摄像头
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[i] 用户按下 'q'，结束当前摄像头测试。")
                break

            # 达到展示时间则退出当前摄像头
            if elapsed >= show_seconds:
                break

        # 最终打印估算 FPS
        total_elapsed = time.time() - start_time
        if total_elapsed > 0 and frame_count > 0:
            est_fps = frame_count / total_elapsed
            print(f"[✓] camera index {i} 可用。估算 FPS：{est_fps:.2f}，自报 FPS：{reported_fps_str}")
            print(f"    分辨率：{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        else:
            print(f"[?] camera index {i} 无法稳定估算 FPS。")

        cap.release()
        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()
    print("\n测试完成。")

if __name__ == "__main__":
    # 示例：测试前 5 个索引，每个显示 5 秒；Windows 用 DirectShow
    test_cameras(max_tested=5, show_seconds=5, target_width=None, target_height=None, backend=cv2.CAP_DSHOW)
    # 若在 Linux，可改为：backend=cv2.CAP_V4L2 或 backend=None