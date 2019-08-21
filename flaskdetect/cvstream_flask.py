from flask_opencv_streamer.streamer import Streamer
import cv2
import imutils
from imutils.video import VideoStream
import time

port = 3030
require_login = False
streamer = Streamer(port, require_login)

try:
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()

        streamer.update_frame(frame)

        if not streamer.is_streaming:
            streamer.start_streaming()

        key = cv2.waitKey(1)
finally:    
    print("[INFO] quitting...")
    # do a bit of cleanup
    vs.stop()
