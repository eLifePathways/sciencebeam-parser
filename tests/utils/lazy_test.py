import threading
from typing import List

from sciencebeam_parser.utils.lazy import LazyLoaded


class TestLazyLoaded:
    def test_should_return_the_value_the_factory_built(self):
        lazy = LazyLoaded[str](lambda: 'the value')
        assert lazy.get() == 'the value'

    def test_should_not_build_until_asked(self):
        calls: List[int] = []
        lazy = LazyLoaded[int](lambda: calls.append(1) or len(calls))
        assert not lazy.is_loaded
        lazy.get()
        assert lazy.is_loaded
        assert calls == [1]

    def test_should_build_once_for_repeated_calls(self):
        calls: List[int] = []

        def _factory() -> int:
            calls.append(1)
            return len(calls)

        lazy = LazyLoaded[int](_factory)
        assert lazy.get() == lazy.get() == 1
        assert calls == [1]

    def test_should_build_once_under_concurrent_first_use(self):
        """Two requests for the same cold model must not both load it.

        A model is a download and a conversion, and with a profile per request
        first use is an ordinary event rather than a once-per-process one.
        """
        calls: List[int] = []
        lock = threading.Lock()
        started = threading.Barrier(8)

        def _factory() -> object:
            with lock:
                calls.append(1)
            threading.Event().wait(0.02)
            return object()

        lazy = LazyLoaded[object](_factory)
        results: List[object] = []

        def _run() -> None:
            started.wait()
            results.append(lazy.get())

        threads = [threading.Thread(target=_run) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert calls == [1]
        assert len(results) == 8
        assert all(result is results[0] for result in results)
