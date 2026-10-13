"""Count upward movements once, then wait for a completed return before rearming."""
from collections import deque
from statistics import median
import math

class LiftCounter:
    def __init__(self, fps, threshold=20.0):
        self.fps = fps
        self.threshold = threshold
        self.window = deque(maxlen=max(3, round(fps * .15)))
        self.count = 0
        self.rest = None
        self.low = None
        self.peak = None
        self.locked = False
        self.return_frames = 0
        self.rising_frames = 0
        self.previous = None
        self.last_frame = None

    def update(self, value, frame):
        if not math.isfinite(value):
            self.window.clear()
            self.rising_frames = self.return_frames = 0
            return self.count
        if self.last_frame is not None and frame-self.last_frame > self.fps*.5:
            # Never turn a tracking gap into a new upward movement.
            self.window.clear()
            self.low = None
            self.rising_frames = self.return_frames = 0
        self.last_frame = frame
        self.window.append(value)
        if len(self.window) < self.window.maxlen:
            return self.count
        v = median(self.window)
        if self.rest is None:
            self.rest = v
        if self.low is None:
            self.low = v
            self.previous = v
        if not self.locked:
            self.low = min(self.low, v)
            self.rising_frames = self.rising_frames+1 if v > self.previous+.1 else 0
            if v-self.low >= self.threshold and self.rising_frames >= max(2,round(self.fps*.1)):
                self.count += 1
                self.locked = True
                self.peak = v
                self.return_frames = 0
        else:
            self.peak = max(self.peak, v)
            returned = self.peak-v >= self.threshold and v <= max(self.rest,self.low)+10
            self.return_frames = self.return_frames+1 if returned else 0
            if self.return_frames >= max(3,round(self.fps*.25)):
                self.locked = False
                self.low = v
                self.rising_frames = 0
        self.previous = v
        return self.count
