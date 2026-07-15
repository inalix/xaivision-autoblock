import time

from robyn import Robyn, Response

from xaivision_autoblock.utils import settings
from xaivision_autoblock.logger import logger
from xaivision_autoblock.autoblock import AutoBlock


app = None
if settings.WEB_STREAM:
    app = Robyn(__name__)
    app.autoblock = None
    @app.route('/video_feed')
    def video_feed():
        if not app.autoblock:
            return Response('No Images for you!')

        def generate():
            while True:
                if app.autoblock.image_queue.empty():
                    time.sleep(0.1)
                    continue
                yield app.autoblock.image_queue.get()

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/hello')
    def hello():
        return Response('OK')

def main():
    autoblock = AutoBlock()
    autoblock.start()
    if settings.WEB_STREAM:
        app.autoblock = autoblock
    logger.info('STARTED CUY')

if __name__ == '__main__':
    main()
    if settings.WEB_STREAM:
        app.run(host=settings.WEB_STREAM_HOST, port=settings.WEB_STREAM_PORT)