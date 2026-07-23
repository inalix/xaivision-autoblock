import time

from robyn import Robyn, StreamingResponse, Headers

from xaivision_autoblock.utils import settings
from xaivision_autoblock.logger import logger
from xaivision_autoblock.autoblock import AutoBlock


app = None
if settings.WEB_STREAM:
    app = Robyn(__name__)
    app.autoblock = None
    @app.get('/stream')
    def stream():
        if not app.autoblock:
            return "No Images for you!"

        def generate():
            while True:
                if app.autoblock.image_queue.empty():
                    time.sleep(0.1)
                    continue
                yield app.autoblock.image_queue.get()

        headers = Headers({"Content-Type": "multipart/x-mixed-replace; boundary=frame"})
        return StreamingResponse(generate(), headers=headers, media_type="multipart/x-mixed-replace; boundary=frame")


    @app.get('/hello')
    def hello():
        return "OK"

def main():
    autoblock = AutoBlock()
    autoblock.start()
    if settings.WEB_STREAM:
        app.autoblock = autoblock
    logger.info('STARTED CUY')

if __name__ == '__main__':
    main()
    if settings.WEB_STREAM:
        app.start(host=settings.WEB_STREAM_HOST, port=settings.WEB_STREAM_PORT)
