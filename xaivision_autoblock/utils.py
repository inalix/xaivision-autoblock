import os
from pathlib import Path
from urllib import parse

import cv2
import numpy as np
from dotenv import load_dotenv


# Load .env file if it is not PRODUCTION
# Karena settings.py ada di dalam subfolder, kita mundur satu langkah ke root
base_path = Path(__file__).resolve().parent.parent
env_path = base_path / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

SETTINGS_LIST = (
    ('WEB_STREAM', bool, False),
    ('WEB_STREAM_HOST', str, '0.0.0.0'),
    ('WEB_STREAM_PORT', int, 8080),

    ('VIDEO_PATH', str, '/app/test_videos/1.mp4'),
    ('PARKING_LABEL', str, '001'),
    ('PARKING_POSITION', str, '0,0,200,200'),
    ('DATA_TARGET_URL', str, ''),
    ('MODEL_PATH', str, '/app/models/production.pt'),
    ('MODEL_RES', str, '640,640'),
    ('TARGET_FPS', int, 3),
    ('STORE_FOOTAGE_IMAGE', bool, False),
    ('STORE_FOOTAGE_VIDEO', bool, False),
    ('STORE_FOOTAGE_VIDEO_SECS', int, 10),
    ('LOCAL_IMAGES_SAVE', bool, False),
    ('LOCAL_IMAGES_DIR', str, '/app/store_images'),
    ('LOCAL_IMAGES_SECS', int, 1),
    ('SHOW_PARKING_AREA', bool, True),
    ('SHOW_CLOCK', bool, True),
    ('ON_BLOCK_MIN_SECS_ON_BAY', int, 5),
    ('OFF_BLOCK_MAX_OVERLAP', float, 0.05),
    ('ON_BAY_MIN_OVERLAP', float, 0.15),
    # ini jika pakai threshold by "BBOX"
    # ('AIRPLANE_STAY_STILL_THRESHOLD', float, 3.5),
    # ini jika pakai threshold by "OVERLAP"
    ('AIRPLANE_STAY_STILL_THRESHOLD', float, 0.05),
    # "OVERLAP" untuk overlap bbox dengan bay, "BBOX" untuk overlap bbox dengan latest bbox
    ('AIRPLANE_STAY_STILL_BY', str, 'OVERLAP'),
    ('AIRPLANE_STAY_STILL_IS_SEQUENTIAL', bool, False),
    ('OUT_OF_BAY_TO_START_OFF_BLOCK', int, 25),
    ('GAP_OVERLAP_CONSIDERED_MOVE', float, 0.09),

    # SYSTEM TRACKING RELATED CONFIG
    ('YOLO_TRACKER', str, 'bytetrack.yaml'),

    # RTSP OUTPUT STREAM FFMPEG
    ('VIDEO_OUTPUT_STREAM', bool, False),
    ('VIDEO_OUTPUT', str, 'rtsp://mediamtx:8554/001'),
    ('VIDEO_OUTPUT_FPS', int, 10),
    ('FFMPEG_LOG_LEVEL', str, 'error'),

    ('NMS_CONFIDENCE_THRESHOLD', float, 0.55),
    ('NMS_IOU_THRESHOLD', float, 0.5),

    # KHUSUS INTRUDERS
    ('USING_SAHI', bool, True),
    ('ENABLE_TRACKING', bool, False),
    ('RESTRICTED_AREA', str, ''),
    ('SHOW_RESTRICTED_AREA', bool, True),
    ('AREA_LABEL', str, 'AREA 1'),
    ('DATA_TARGET_URL', str, ''),

    ('OPENCV_USING_CUDA', bool, False),
)


class Settings():
    default = None
    settings = {}

    def __init__(self, default=None):
        self.default = default
        self.initial_settings()

    def initial_settings(self):
        for s in SETTINGS_LIST:
            value = os.getenv(s[0], s[2])
            if (type(value) == str):
                value = value.strip()
            setattr(self, s[0], s[1](value))

settings = Settings()


def get_data_target_url(varname='DATA_TARGET_URL'):
    dtu = {
        'cleaned': None,
        'auth': None,
        'headers': {},
    }
    value = getattr(settings, varname, None)
    if value:
        dtp = parse.urlparse(settings.DATA_TARGET_URL)
        is_okay = False
        if dtp.username:
            dtu['auth'] = (dtp.username, dtp.password)
            is_okay = True
        elif dtp.query:
            dtqs = parse.parse_qs(dtp.query)
            if 'api_key' in dtqs and 'api_key_value' in dtqs:
                dtu['headers'][dtqs['api_key'][0]] = dtqs['api_key_value'][0]
                is_okay = True

        if is_okay:
            dtu['cleaned'] = f'{dtp.scheme}://{dtp.hostname}'
            if dtp.port:
                dtu['cleaned'] += f':{dtp.port}'
            dtu['cleaned'] += f'{dtp.path}'
    return dtu


def draw_bbox_and_label(frame, bbox, bg_color, size,
                        label=None, label_color=(0, 0, 0),
                        bbox_type='xyxy'):

    x1, y1, x2, y2 = bbox
    if bbox_type == 'xywh':
        x2 = x1 + x2
        y2 = y1 + y2

    if bg_color:
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, size)

    if label:
        labels = label.split('\n')
        max_width = 0
        max_height = 0
        text_height = 0
        for l in labels:
            (text_width, text_height), baseline = cv2.getTextSize(
                l, cv2.FONT_HERSHEY_SIMPLEX, 0.5, size)
            max_width = max(max_width, text_width)
            max_height += text_height

        if bg_color:
            cv2.rectangle(frame, (x1, y1), (x1 + max_width, y1 + max_height + size), bg_color, -1)

        for i, l in enumerate(labels, 1):
            cv2.putText(frame, l, (x1, y1 + (text_height * i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, size)


# Function to create mask
def create_mask(frame, points):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    return mask
