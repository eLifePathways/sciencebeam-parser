from time import perf_counter
from typing import Optional


class StopWatch:
    def __init__(self):
        self.start = perf_counter()

    def get_elapsed_seconds(self, reset=False):
        end = perf_counter()
        elapsed = end - self.start
        if reset:
            self.start = end
        return elapsed


class StopWatchRecorder:
    def __init__(self):
        self.stop_watch = StopWatch()
        self.recorded_timings = []
        self.started = None

    def stop(self):
        self.start(None)

    def start(self, name: Optional[str]):
        elapsed = self.stop_watch.get_elapsed_seconds(reset=True)
        if self.started:
            self.recorded_timings.append((self.started, elapsed))
        self.started = name

    def __str__(self):
        total = ('total', sum(elapsed for _, elapsed in self.recorded_timings))
        return ', '.join(
            '%s: %.6fs' % (name, elapsed)
            for name, elapsed in self.recorded_timings + [total]
        )
