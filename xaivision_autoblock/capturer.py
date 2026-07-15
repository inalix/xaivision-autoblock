import threading
import time
import cv2
import queue

from jimbo.logger import logger


class CaptureBase:
    def __init__(self, path, target_fps, frame_queue):
        self.path = path
        self.target_fps = target_fps
        self.frame_queue = frame_queue


class Video(CaptureBase):
    def start_capture(self):
        self.frame_capture = threading.Thread(target=self.frame_capture_thread)
        self.frame_capture.start()

    def frame_capture_thread(self):
        cap = cv2.VideoCapture(self.path)
        while not cap.isOpened():
            cap.release()
            time.sleep(0.1)
            cap = cv2.VideoCapture(self.path)

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.skip_frames = int(round(self.fps / self.target_fps))
        self.frame_duration = round(1 / self.target_fps, 2) - 0.01
        msg = f'Frame capture started with FPS: {self.fps}, Target FPS: '
        msg += f'{self.target_fps}, SKIP FRAMES {self.skip_frames}, '
        msg += f'frame duration {self.frame_duration}'
        logger.info(msg)

        while True:
            if not cap.isOpened():
                time.sleep(1)
                continue

            ret, frame = cap.read()
            if not ret:
                logger.error('Failed to read frame from video stream! is it ended?')
                time.sleep(5)
                continue

            if self.frame_queue.full():
                self.frame_queue.get()
            # logger.info('Captured')
            self.frame_queue.put(frame)

            if self.skip_frames > 1:
                # Skip frames by grabbing to discard frames
                for _ in range(self.skip_frames - 1):
                    cap.grab()

            time.sleep(self.frame_duration)

        cap.release()
        logger.info('Frame capture thread stopped')


class Camera(CaptureBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dq = queue.Queue()

    def start_capture(self):
        threading.Thread(target=self.frame_capture_thread).start()

    def frame_capture_thread(self):
        cap = cv2.VideoCapture(self.path)
        while not cap.isOpened():
            cap.release()
            time.sleep(0.1)
            cap = cv2.VideoCapture(self.path)

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_duration = 1 / self.target_fps
        msg = f'Camera capture started with FPS: {self.fps}, Target FPS: '
        msg += f'{self.target_fps}, frame duration {self.frame_duration}'
        logger.info(msg)

        threading.Thread(target=self.deliver_frame_thread).start()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if not cap.isOpened():
                logger.error('Camera closed! try reopen it')
                cap.release()
                time.sleep(0.1)
                cap = cv2.VideoCapture(self.path)
                continue

            if not self.dq.empty():
                try:
                    # discard previous (unprocessed) frame
                    self.dq.get_nowait()
                except queue.Empty:
                    pass
            self.dq.put(frame)

    def read_a_frame(self):
        return self.dq.get()

    def deliver_frame_thread(self):
        while True:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put(self.read_a_frame())
            time.sleep(self.frame_duration)
