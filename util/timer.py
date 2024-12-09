"""
Simple timer from https://github.com/jannerm/diffuser/blob/main/diffuser/utils/timer.py
"""

import time


class Timer:

    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff

def sec2HMS(seconds):
    """ 
    translate duration in seconds to hour, minute, second format.
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    minutes = int(minutes)
    seconds=int(seconds)
    if hours > 0:
        return f"{int(hours):02d}h:{int(minutes):02d}m:{seconds:02d}s"
    else:
        return f"{int(minutes):02d}m:{seconds:02d}s"