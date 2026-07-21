
import threading
import time
import os
import queue
import subprocess
import niquests
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from io import BufferedReader, BytesIO

import cv2

import numpy as np

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from shapely.geometry import Polygon

from .utils import (
    settings, get_data_target_url, draw_bbox_and_label
)
from .capturer import Video, Camera
from .logger import logger

utc = ZoneInfo('UTC')

class AutoBlock:
    tracking_ids = [4]
    parked_airplanes = {}
    api_session = None

    def __init__(self):
        self.initial_settings()

    def initial_settings(self):
        parking_pos = []
        for r in settings.PARKING_POSITION.split('|'):
            parking_pos.append([int(rs) for rs in r.split(',')])

        self.parking_position = np.array(parking_pos)
        self.parking_label: str = settings.PARKING_LABEL
        self.parking_label_p = f'BAY {self.parking_label}'
        self.parking_polygon = Polygon(self.parking_position)
        self.show_parking_area = settings.SHOW_PARKING_AREA
        self.show_clock: bool = settings.SHOW_CLOCK

        self.device = select_device('')
        self.model_path: str = settings.MODEL_PATH
        print(f'MODELPATH {self.model_path}')
        self.model = YOLO(self.model_path)
        self.model_res = [int(res) for res in settings.MODEL_RES.split(',')]

        # THRESHOLDS
        self.airplane_stay_still_threshold = settings.AIRPLANE_STAY_STILL_THRESHOLD
        self.nms_confidence_threshold = settings.NMS_CONFIDENCE_THRESHOLD
        self.nms_iou_threshold = settings.NMS_IOU_THRESHOLD

        # YOLO tracker config
        self.yolo_tracker: str = settings.YOLO_TRACKER

        # RTSP output stream
        self.video_output_stream: bool = settings.VIDEO_OUTPUT_STREAM
        self.video_output: str = settings.VIDEO_OUTPUT
        self.video_output_fps: int = settings.VIDEO_OUTPUT_FPS

        self.video_path: str = settings.VIDEO_PATH
        self.target_fps: int = settings.TARGET_FPS
        self.orig_shape = None
        self.store_footage_image: bool = settings.STORE_FOOTAGE_IMAGE
        self.store_footage_video: bool = settings.STORE_FOOTAGE_VIDEO
        self.store_footage_video_secs: int = settings.STORE_FOOTAGE_VIDEO_SECS

        self.local_images_save: bool = settings.LOCAL_IMAGES_SAVE
        self.local_images_directory: str = settings.LOCAL_IMAGES_DIR
        self.local_images_secs: int = settings.LOCAL_IMAGES_SECS

        self.data_target: str = get_data_target_url()

        # block settings
        self.on_block_min_secs_on_bay = settings.ON_BLOCK_MIN_SECS_ON_BAY
        self.off_block_max_overlap = settings.OFF_BLOCK_MAX_OVERLAP
        self.on_bay_min_overlap = settings.ON_BAY_MIN_OVERLAP
        self.out_of_bay_to_start_off_block = settings.OUT_OF_BAY_TO_START_OFF_BLOCK
        self.gap_overlap_considered_move = settings.GAP_OVERLAP_CONSIDERED_MOVE

        # queues
        self.frame_queue = queue.Queue(10)
        self.processed_frame_queue = queue.Queue(5)
        self.image_queue = queue.Queue(3)
        self.stream_frame_queue = queue.Queue(3)
        self.ffmpeg_process = None

        # IMAGES TO STORE QUEUE
        self.before_on_block_queue = queue.Queue(self.target_fps)
        self.before_off_block_queue = queue.Queue(self.target_fps)
        self.before_on_block_vid_queue = queue.Queue(self.target_fps * settings.STORE_FOOTAGE_VIDEO_SECS)
        self.before_off_block_vid_queue = queue.Queue(self.target_fps * settings.STORE_FOOTAGE_VIDEO_SECS)
        self.on_block_image = None
        self.process_image_width = self.model_res[0]

        self.opencv_using_cuda = settings.OPENCV_USING_CUDA

        logger.info(f'DEVICE IS {self.device}')
        logger.info(f'VIDEO_PATH {self.video_path}')
        logger.info(f'PARKING_POSITION {self.parking_position}')
        logger.info(f'MODEL_PATH {self.model_path}')
        logger.info(f'MODEL_RES {self.model_res}')
        logger.info(f'DATA_TARGET_URL {self.data_target["cleaned"]}')

    def to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]

    def preprocess_frame(self, frame, width, height):
        if width <= self.process_image_width:
            return frame
        elif self.opencv_using_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            img = cv2.cuda.resize(gpu_frame, self.model_res, interpolation=cv2.INTER_AREA)
            img = img.download()
            return img
        else:
            return cv2.resize(frame, self.model_res, interpolation=cv2.INTER_AREA)

    def scale_bbox_to_original(self, x1, y1, x2, y2):
        """
        Scale bounding box coordinates dari ukuran model ke ukuran frame asli.
        """
        original_height, original_width = self.orig_shape
        if original_width <= self.process_image_width:
            return x1, y1, x2, y2

        model_width, model_height = self.model_res
        x1 = x1 * original_width / model_width
        x2 = x2 * original_width / model_width
        y1 = y1 * original_height / model_height
        y2 = y2 * original_height / model_height
        return x1, y1, x2, y2

    def get_airplane_bay_overlap(self, bbox):
        x, y, w, h = map(int, bbox)
        bbox_polygon = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
        intersect = self.parking_polygon.intersection(bbox_polygon).area
        overlap_ratio = intersect / self.parking_polygon.area
        # print(f'OVERLAP {overlap_ratio}')
        return overlap_ratio

    def is_still(self, bbox, last_position):
        x, y, w, h = bbox
        last_x, last_y, last_w, last_h = last_position

        # logger.info(f'BBOX {bbox}')
        # logger.info(f'LAST BBOX {last_position}')
        threshold = self.airplane_stay_still_threshold
        gapx = abs((x + w) - (last_x + last_w))
        gapy = abs((y + h) - (last_y + last_h))
        logger.info(f'IS STILL GAP {gapx}, {gapy}')
        if gapx < threshold and gapy < threshold:
            return True

        return False

    def get_before_image(self, block_type, filename):
        if block_type == 'ON':
            bef_queue = self.before_on_block_queue
        else:
            bef_queue = self.before_off_block_queue

        if bef_queue.empty():
            return

        bret, bjpeg = cv2.imencode('.jpeg', bef_queue.get())
        if not bret:
            return

        befimg_bytes = BytesIO(bjpeg.tobytes())
        befimg_bytes.name = f'BEF_{filename}.jpg'
        bef_image = BufferedReader(befimg_bytes)
        return bef_image

    def convert_to_h264(self, input_file, output_file):
        def worker():
            try:
                # Use FFmpeg to convert the video
                subprocess.run([
                    'ffmpeg',
                    '-i', input_file,
                    '-c:v', 'libx264',
                    '-crf', '23', # Quality setting (lower is better, range: 0-51)
                    '-preset', 'fast', # Preset for encoding speed vs. quality
                    '-movflags', '+faststart',  # Optimize for web streaming
                    output_file
                ], check=True)

                # Delete the original file after conversion
                os.remove(input_file)
                print(f'Deleted original file: {input_file}')
                return output_file
            except subprocess.CalledProcessError as e:
                print(f'Error during video conversion: {e}')
                return None
            except FileNotFoundError as e:
                print(f'Error deleting original file: {e}')
                return None

        convert_thread = threading.Thread(target=worker)
        convert_thread.start()
        convert_thread.join()

    def generate_video(self, block_type, filename):
        if block_type == 'ON':
            frame_queue = self.before_on_block_vid_queue
        else:
            frame_queue = self.before_off_block_vid_queue

        if frame_queue.empty():
            logger.info('EMPTY VIDEO FRAME')
            return

        frames = list(frame_queue.queue)
        first_frame = frames[0]
        height, width, layers = first_frame.shape
        writer = cv2.VideoWriter_fourcc(*'mp4v')
        filename = f'/app/store_videos/FOOTAGE_{filename}.mp4'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        out_temp = cv2.VideoWriter(
            filename, cv2.CAP_FFMPEG, writer, self.target_fps, (width, height))

        if not out_temp.isOpened():
            logger.info(f'Failed to open VideoWriter for file: {filename}')
            return None

        logger.info(f'PANJANG FRAME VIDEO {len(frames)} {filename}')
        for frame in frames:
            # print('NULIS FRAME')
            out_temp.write(frame)

        out_temp.release()

        converted_filename = filename.replace('.mp4', '_h264.mp4')
        self.convert_to_h264(filename, converted_filename)
        # h264_file = self.convert_to_h264(filename, converted_filename)
        if os.path.isfile(converted_filename):
            logger.info(f"Video converted to H.264: {converted_filename}")
            return converted_filename

        logger.info(f"Failed to convert video to H.264")
        return None

    def _get_api_session(self):
        if self.api_session is not None:
            return self.api_session

        self.api_session = niquests.Session()
        if self.data_target['auth']:
            self.api_session.auth = self.data_target['auth']
        return self.api_session

    def _post_data_to_target(self, data, files):
        ses = self._get_api_session()

        try:
            resp = ses.post(
                self.data_target['cleaned'], data=data,
                files=files, headers=self.data_target['headers'])
            logger.info(f'RESPONSE {resp.text}')
        except Exception as e:
            logger.error(f'ERROR save to target {e}')

    def post_data_thread(self, jpeg, block_type, block_time):
        if not self.data_target['cleaned']:
            logger.info('Not storing data (NO TARGET URL)')
            return

        # generating all footage files needed
        print('START POST THREAD')
        filename = block_time.strftime('%s%f')
        files = set([])
        if self.store_footage_image:
            bef_image = self.get_before_image(block_type, filename)
            if bef_image:
                files.add(('images', (f'BEF_{filename}.jpg', bef_image, 'image/jpeg')))

            img_bytes = BytesIO(jpeg)
            img_bytes.name = f'{block_type}_{filename}.jpg'
            image = BufferedReader(img_bytes)
            files.add(('images', (f'{block_type}_{filename}.jpg', image, 'image/jpeg')))

        video_file = None
        if self.store_footage_video:
            print('GENERATING VIDEO')
            video_file = self.generate_video(block_type, filename)
            if video_file:
                files.add(('footage_video', (f'VID_{filename}.mp4', open(video_file, 'rb'), 'video/mp4')))

        block_data = {
            'block_type': block_type,
            'block_time': block_time.isoformat(),
            'bay': self.parking_label,
        }
        # logger.info(f'POSTING DATA {block_type} {block_time}')
        self._post_data_to_target(block_data, files)

        if video_file:
            if os.path.exists(video_file):
                os.remove(video_file)

    def generate_capture(self):
        if '://' in self.video_path:
            logger.info('USING CAMERA')
            return Camera(self.video_path, self.target_fps, self.frame_queue)
        logger.info('USING VIDEO')
        return Video(self.video_path, self.target_fps, self.frame_queue)


    def process_on_block_data(self, bbox, track_id, frame):
        if self.parked_airplanes[track_id]['on_block']:
            return False

        if self.before_on_block_queue.full():
            self.before_on_block_queue.get()
        self.before_on_block_queue.put(frame)

        if self.before_on_block_vid_queue.full():
            self.before_on_block_vid_queue.get()
        self.before_on_block_vid_queue.put(frame)

        now = datetime.now().replace(tzinfo=utc)
        overlap_value = self.get_airplane_bay_overlap(bbox)

        if overlap_value >= self.on_bay_min_overlap:
            if not self.parked_airplanes[track_id]['on_bay']:
                self.parked_airplanes[track_id]['on_bay'] = True
                self.parked_airplanes[track_id]['on_bay_last_overlap'] = overlap_value
                self.parked_airplanes[track_id]['on_bay_start'] = now
                self.parked_airplanes[track_id]['on_bay_last_pos'] = bbox
                self.parked_airplanes[track_id]['ever_off_bay'] += 1
                return False
        else:
            self.parked_airplanes[track_id]['on_bay'] = False
            self.parked_airplanes[track_id]['on_bay_last_overlap'] = 0
            self.parked_airplanes[track_id]['on_bay_start'] = None
            self.parked_airplanes[track_id]['on_bay_last_pos'] = None
            self.parked_airplanes[track_id]['ever_off_bay'] += 1

        if not self.parked_airplanes[track_id]['on_bay']:
            self.parked_airplanes[track_id]['ever_off_bay'] += 1
            return False

        diff = (now - self.parked_airplanes[track_id]['on_bay_start']).total_seconds()
        if diff < 1:
            return False

        # counting gap after 1 second
        is_still = self.is_still(bbox, self.parked_airplanes[track_id]['on_bay_last_pos'])
        self.parked_airplanes[track_id]['on_bay_start'] = now
        self.parked_airplanes[track_id]['on_bay_last_pos'] = bbox
        logger.info(f'IS STILL {is_still}')
        if not is_still:
            self.parked_airplanes[track_id]['on_block_still'] = 0
            return False

        self.parked_airplanes[track_id]['on_block_still'] += 1
        # minimum stay still at n seconds consecutive COUNT AS BLOCK ON
        if self.parked_airplanes[track_id]['on_block_still'] < self.on_block_min_secs_on_bay:
            return False

        if self.parked_airplanes[track_id]['ever_off_bay'] < 5:
            # FAKED because ever off bay < 5
            logger.info('FAKED ON BLOCK (ALREADY ON PARKING STAND)')
            self.parked_airplanes[track_id]['on_block_real'] = False
        else:
            self.parked_airplanes[track_id]['on_block_real'] = True

        on_block_time = now.replace(microsecond=0) - timedelta(seconds=self.on_block_min_secs_on_bay + 1)
        self.parked_airplanes[track_id]['start_time'] = on_block_time
        self.parked_airplanes[track_id]['on_block_time'] = on_block_time
        self.parked_airplanes[track_id]['on_block'] = True
        return is_still

    def process_off_block_data(self, bbox, track_id, frame):
        send_data_off_block = False

        if not self.parked_airplanes[track_id]['on_block']:
            return False

        if self.parked_airplanes[track_id]['off_block']:
            return False

        if self.before_off_block_vid_queue.full():
            self.before_off_block_vid_queue.get()
        self.before_off_block_vid_queue.put(frame)

        if self.before_off_block_queue.full():
            self.before_off_block_queue.get()
        self.before_off_block_queue.put(frame)

        now = datetime.now().replace(tzinfo=utc)
        overlap_ratio = self.get_airplane_bay_overlap(bbox)

        if overlap_ratio <= self.off_block_max_overlap:
            if not self.parked_airplanes[track_id]['out_of_bay']:
                self.parked_airplanes[track_id]['out_of_bay'] = now

            diff = (now - self.parked_airplanes[track_id]['out_of_bay']).total_seconds()
            if diff < 5:
                return False

            self.parked_airplanes[track_id]['off_block'] = True
            if self.parked_airplanes[track_id]['off_block_start']:
                off_block_time = self.parked_airplanes[track_id]['off_block_start'].replace(microsecond=0)
                off_block_time -= timedelta(seconds=3)
                logger.info(f'USING OFF BLOK START {off_block_time}')
            else:
                off_block_time = self.parked_airplanes[track_id]['out_of_bay'].replace(microsecond=0)
                off_block_time -= timedelta(seconds=self.out_of_bay_to_start_off_block)
                logger.info(f'USING OUT OF BAY {off_block_time}')
            self.parked_airplanes[track_id]['off_block_time'] = off_block_time
            return True
        else:
            self.parked_airplanes[track_id]['out_of_bay'] = None

        if not self.parked_airplanes[track_id]['off_block_counting_start']:
            self.parked_airplanes[track_id]['off_block_counting_start'] = now
            self.parked_airplanes[track_id]['off_block_last_overlap'] = overlap_ratio

        diff = (now - self.parked_airplanes[track_id]['off_block_counting_start']).total_seconds()
        if diff < 2:
            return False

        # counting gap after 2 seconds
        gap = self.parked_airplanes[track_id]['off_block_last_overlap'] - overlap_ratio
        self.parked_airplanes[track_id]['off_block_counting_start'] = now
        self.parked_airplanes[track_id]['off_block_last_overlap'] = overlap_ratio
        # print(f'GAP {gap}, MIN {self.gap_overlap_considered_move}')
        if gap >= self.gap_overlap_considered_move:
            if not self.parked_airplanes[track_id]['off_block_start']:
                self.parked_airplanes[track_id]['off_block_start'] = now - timedelta(seconds=2)
        else:
            self.parked_airplanes[track_id]['off_block_start'] = None

        return False

    def draw_parking_area(self, frame):
        cv2.polylines(frame, [self.parking_position], isClosed=True, color=(0, 255, 0),
                        thickness=2)
        px, py = self.parking_position[0][0], self.parking_position[0][1]
        (text_width, text_height), baseline = cv2.getTextSize(
            self.parking_label_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (px, py), (px + text_width, py + text_height + 2),
                        (0, 255, 0), -1)
        cv2.putText(frame, self.parking_label_p,
                    (px, py + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def draw_clock(self, frame, frame_height):
        clock = datetime.now().replace(tzinfo=utc).strftime('%Y/%m/%d %H:%M:%S UTC')
        draw_bbox_and_label(
            frame, (5, frame_height - 16, 100, frame_height - 16), None, 4,
            label=clock, label_color=(255, 255, 255))
        draw_bbox_and_label(
            frame, (5, frame_height - 16, 100, frame_height - 16), None, 2,
            label=clock)

    # ------------------------------------------------------------------ #
    #  Helpers untuk frame_processing_thread                              #
    # ------------------------------------------------------------------ #

    def _run_yolo_tracking(self, frame):
        """jalankan YOLO .track(). Return results."""
        return self.model.track(
            frame,
            conf=self.nms_confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.model_res[0],
            classes=self.tracking_ids,
            persist=True,
            verbose=False,
            tracker=self.yolo_tracker,
        )

    def _register_airplane(self, track_id, bbox):
        """Inisialisasi state pesawat baru jika belum terdaftar."""
        if track_id in self.parked_airplanes:
            return
        self.parked_airplanes[track_id] = {
            'start_time': None,
            'on_block': False,
            'on_block_time': None,
            'on_block_real': True,
            'on_block_still': 0,
            'ever_off_bay': 0,
            'on_bay': False,
            'on_bay_last_overlap': 0,
            'on_bay_start': None,
            'on_bay_last_pos': None,
            'off_block': False,
            'off_block_time': None,
            'last_position': bbox,
            'off_block_last_overlap': 0,
            'off_block_counting_start': None,
            'off_block_start': None,
            'out_of_bay': None,
        }

    def _build_label(self, track_id, conf):
        """Susun string label untuk bbox dari state pesawat."""
        airplane = self.parked_airplanes[track_id]
        label = f'AIRCRAFT: {track_id} ({int(conf * 100)}%)'
        if airplane['on_block_time']:
            on_block_str = (airplane['on_block_time'].isoformat()
                            if airplane['on_block_real'] else 'NOT RECORDED')
            label += f'\nON BLOCK: {on_block_str}'
        if airplane['off_block_time']:
            label += f'\nOFF BLOCK: {airplane["off_block_time"].isoformat()}'
        return label

    def _process_tracked_object(self, bbox, track_id, conf, frame):
        """
        Proses satu objek yang ditrack: daftarkan jika baru, update state
        on/off block, gambar bbox ke frame. Return data_block jika ada event.
        """
        self._register_airplane(track_id, bbox)

        airplane = self.parked_airplanes[track_id]
        send_on = self.process_on_block_data(bbox, track_id, frame)
        send_off = self.process_off_block_data(bbox, track_id, frame)

        data_block = {}
        if send_on and airplane['on_block_real']:
            data_block = {'type': 'ON', 'time': airplane['on_block_time']}
        if send_off:
            data_block = {'type': 'OFF', 'time': airplane['off_block_time']}

        draw_bbox_and_label(
            frame, [int(c) for c in bbox], (255, 0, 0), 1,
            label=self._build_label(track_id, conf),
            label_color=(255, 255, 255), bbox_type='xywh')

        airplane['last_position'] = bbox
        return data_block

    def _process_result_boxes(self, result, frame):
        """
        Ekstrak boxes dari satu YOLO result dan proses tiap objek yang ditrack.
        Return data_block event pertama yang ditemukan, atau {} jika tidak ada.
        Karena hanya 1 parking stand, paling banyak 1 event terjadi per frame.
        """
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return {}

        xyxys = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        data_block = {}
        for xyxy, track_id, conf in zip(xyxys, track_ids, confs):
            x1, y1, x2, y2 = xyxy
            bbox = [x1, y1, x2 - x1, y2 - y1]
            event = self._process_tracked_object(bbox, track_id, conf, frame)
            # Ambil event pertama saja — 1 parking stand, maks 1 event per frame
            if event and not data_block:
                data_block = event

        return data_block

    # ------------------------------------------------------------------ #
    #  Main processing thread                                             #
    # ------------------------------------------------------------------ #

    def frame_processing_thread(self):
        while True:
            if self.frame_queue.empty():
                time.sleep(0.1)
                continue

            frame = self.frame_queue.get()
            if self.orig_shape is None:
                self.orig_shape = frame.shape[:2]
                logger.info(
                    f'VIDEO SHAPE Width: {self.orig_shape[1]}, Height: {self.orig_shape[0]}')

            results = self._run_yolo_tracking(frame)

            frame_height, _ = self.orig_shape
            if self.show_parking_area:
                self.draw_parking_area(frame)
            if self.show_clock:
                self.draw_clock(frame, frame_height)

            data_block = {}
            for result in results:
                event = self._process_result_boxes(result, frame)
                if event:
                    data_block = event

            if self.processed_frame_queue.full():
                self.processed_frame_queue.get()
            self.processed_frame_queue.put((frame, data_block))

            if self.video_output_stream:
                if self.stream_frame_queue.full():
                    self.stream_frame_queue.get()
                self.stream_frame_queue.put(frame)


    def clean_local_images_dir(self):
        os.makedirs(self.local_images_directory, exist_ok=True)
        files = list(Path(self.local_images_directory).glob('*'))

        # Sort files by filename
        files.sort(key=lambda x: x.name)
        delete_count = len(files) - (self.local_images_secs * self.target_fps)

        # Delete the oldest files if there are more than 'seconds * target_fps' files
        if delete_count > 0:
            for file in files[:delete_count]:
                try:
                    os.remove(file)
                    # print(f"Deleted {file}")
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")

    def image_processing_thread(self):
        last_data = None
        while True:
            if self.processed_frame_queue.empty():
                time.sleep(0.05)
                continue

            frame, data = self.processed_frame_queue.get()
            ret, jpeg = cv2.imencode('.jpeg', frame)
            if data:
                last_data = data

            if not ret:
                continue

            jpeg = jpeg.tobytes()
            # print(f'DATA {last_data}')
            if last_data:
                logger.info(f'POST WOE {last_data}')
                post_data = threading.Thread(
                    target=self.post_data_thread,
                    args=(jpeg, last_data['type'], last_data['time']))
                post_data.start()
                last_data = None

            if self.local_images_save:
                os.makedirs(self.local_images_directory, exist_ok=True)
                imagename = f'{int(time.time() * 1_000_000)}.jpg'
                imagename = os.path.join(self.local_images_directory, imagename)
                with open(imagename, 'wb') as f:
                    f.write(jpeg)

                self.clean_local_images_dir()

            x = (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
            self.image_queue.put(x)

    def _start_ffmpeg_process(self, width, height):
        """Spawn subprocess FFmpeg dan simpan ke self.ffmpeg_process."""
        command = [
            'ffmpeg',
            '-y',
            '-loglevel', settings.FFMPEG_LOG_LEVEL,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(self.video_output_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-rtsp_transport', 'tcp',
            '-f', 'rtsp',
            self.video_output,
        ]
        self.ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f'FFmpeg RTSP stream started -> {self.video_output}')

    def stream_output_thread(self):
        """
        Ambil frame dari stream_frame_queue dan kirim ke FFmpeg
        untuk di-encode sebagai RTSP stream. FFmpeg di-restart otomatis
        jika prosesnya mati.
        """
        while True:
            if self.stream_frame_queue.empty():
                time.sleep(0.05)
                continue

            frame = self.stream_frame_queue.get()
            height, width = self.orig_shape

            # Restart FFmpeg jika belum jalan atau sudah mati
            if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                if self.ffmpeg_process is not None:
                    logger.warning('FFmpeg stream process died, restarting...')
                self._start_ffmpeg_process(width, height)

            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                logger.error('FFmpeg stdin broken pipe, will restart on next frame')
                self.ffmpeg_process = None
            except Exception as e:
                logger.error(f'FFmpeg write error: {e}')
                self.ffmpeg_process = None

    def start(self):
        vid = self.generate_capture()
        vid.start_capture()
        frame_processing = threading.Thread(
            target=self.frame_processing_thread, daemon=True)
        image_processing = threading.Thread(
            target=self.image_processing_thread, daemon=True)
        frame_processing.start()
        image_processing.start()
        if self.video_output_stream:
            stream_output = threading.Thread(
                target=self.stream_output_thread, daemon=True)
            stream_output.start()
            logger.info(f'RTSP stream output enabled -> {self.video_output}')