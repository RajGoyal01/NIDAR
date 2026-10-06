"""A bounded rolling rate counter. Caller owns synchronization."""
from collections import deque
import time


class FPSCounter:
    def __init__(self, window: float = 2.0) -> None:
        self.window = window
        self.times: deque[float] = deque(maxlen=512)

    def tick(self, now: float | None = None) -> None:
        self.times.append(time.monotonic() if now is None else now)

    def value(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        while self.times and self.times[0] < now - self.window:
            self.times.popleft()
        if len(self.times) < 2:
            return 0.0
        elapsed = max(now, self.times[-1]) - self.times[0]
        return (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        self.times.clear()
