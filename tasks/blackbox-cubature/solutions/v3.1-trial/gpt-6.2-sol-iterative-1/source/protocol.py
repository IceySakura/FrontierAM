"""JSON Lines transport for a single cubature instance."""

import json
import math
import signal
import sys
import time

import numpy as np


class Deadline(ArithmeticError):
    pass


class Session:
    def __init__(self):
        self.start = time.monotonic()
        init = json.loads(sys.stdin.buffer.readline())
        if init["type"] != "init":
            raise ValueError("expected init")
        self.d = int(init["dimension"])
        self.budget = int(init["budget"])
        self.max_batch = int(init["max_batch"])
        self.seed = int(init["seed"])
        self.limit = float(init.get("time_limit_s", 30))
        self.used = 0
        self.in_exchange = False
        self.expired = False

    def arm(self):
        def expire(signum,frame):
            self.expired = True
            if not self.in_exchange:
                raise Deadline("wall-time reserve reached")
        signal.signal(signal.SIGALRM,expire)
        signal.setitimer(signal.ITIMER_REAL,max(1.,self.limit-4.))

    @staticmethod
    def disarm():
        signal.setitimer(signal.ITIMER_REAL,0.)

    @property
    def remaining(self):
        return self.budget - self.used

    @property
    def elapsed(self):
        return time.monotonic() - self.start

    def query(self, points):
        z = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, self.d)
        if len(z) > self.remaining or not np.isfinite(z).all():
            raise ValueError("invalid query")
        z = np.clip(z, -40.0, 40.0)
        y = np.empty(len(z))
        for i in range(0, len(z), self.max_batch):
            x = z[i:i + self.max_batch]
            if self.expired:
                raise Deadline("wall-time reserve reached")
            msg = json.dumps(x.tolist(), separators=(",", ":"), allow_nan=False)
            self.in_exchange = True
            sys.stdout.write('{"type":"query","points":' +
                             msg + '}\n')
            sys.stdout.flush()
            data = json.loads(sys.stdin.buffer.readline())
            if data["type"] != "values" or len(data["values"]) != len(x):
                self.in_exchange = False
                raise ValueError("invalid oracle reply")
            self.used += len(x)
            y[i:i + len(x)] = data["values"]
            self.in_exchange = False
            if self.expired:
                raise Deadline("wall-time reserve reached")
        return y

    @staticmethod
    def result(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite estimate")
        print(json.dumps({"type": "result", "estimate": value}, allow_nan=False), flush=True)
