#!/usr/bin/env python3
# encoding: utf-8

## ThreadVideoCapture from https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

from threading import Thread, Lock
import time
from queue import Queue, Empty

import cv2 as cv


class ThreadVideoBase:
    def __init__(self, path, queueSize):
        self.video_path = path

        # initialize the queue used to store frames to write
        self.Q = Queue(maxsize=queueSize)
        self.stopped = False

        self.lock = Lock()
        self.thread = None

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"[{self.__class__.__name__}::start] Started thread")
        return self

    def update(self):
        raise NotImplementedError(f"This method has to be reimplemented in child {type(self).__name__} class")

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        with self.lock:
            self.stopped = True

    def isStopped(self):
        with self.lock:
            return self.stopped

    def release(self):
        self.stop()
        if self.thread is not None:
            self.thread.join()
        self.stream.release()


class ThreadVideoWriter(ThreadVideoBase):
    def __init__(self, path, format, fps, size, queueSize=300):
        super().__init__(path, queueSize)

        self.stream = cv.VideoWriter(path, format, fps, size)

        print(f"[ThreadVideoWriter] Ready to write data")

    def update(self):
        # Keep writing until stop is requested AND the queue has been drained, so
        # no frame still buffered is lost on release()
        while not (self.isStopped() and self.Q.empty()):
            try:
                frame = self.Q.get(timeout=0.1)
            except Empty:
                continue
            self.stream.write(frame)

    def write(self, frame):
        self.Q.put(frame)


## preprocessFunctionCallback -> function to process each frame in capture thread
class ThreadVideoCapture(ThreadVideoBase):
    def __init__(self, path, initFrame = 0, frameSkip = 1, queueSize=300, preprocessFunctionCallback=None):
        super().__init__(path, queueSize)
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv.VideoCapture(path)
        self.frameSkip = frameSkip
        self.initFrame = initFrame
        self.preprocessFunctionCallback = preprocessFunctionCallback

        if not self.stream.isOpened():
            print(f"Error opening video {path}")
            return

        print(f"[ThreadVideoCapture] Ready to capture")

    def update(self):
        total_frames = int(self.stream.get(cv.CAP_PROP_FRAME_COUNT))

        if self.initFrame > 0:
            # Single seek to the start point; from here on decoding is sequential
            self.stream.set(cv.CAP_PROP_POS_FRAMES, self.initFrame)

        for frame_index in range(self.initFrame, total_frames, self.frameSkip):
            # if the thread indicator variable is set, stop the thread
            if self.isStopped():
                return

            # ensure the queue has room in it
            while self.Q.full():
                time.sleep(0.1)

            # read the next frame from the file
            (grabbed, frame) = self.stream.read()
            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                self.stop()
                return
            # add the frame to the queue
            if self.preprocessFunctionCallback is not None:
                frame = self.preprocessFunctionCallback(frame)
            self.Q.put(frame)

            # Skip intermediate frames with grab(): decodes but skips the costly
            # retrieve/convert step (seeking per frame forces a keyframe seek)
            for _ in range(self.frameSkip - 1):
                if not self.stream.grab():
                    self.stop()
                    return

        self.stop()

    def read(self):
        # return next frame in the queue
        while self.Q.empty() and not self.isStopped():
            time.sleep(0.01)

        if self.Q.empty() and self.isStopped():
            return None
        else:
            return self.Q.get()
