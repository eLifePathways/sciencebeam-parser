import threading
from abc import abstractmethod
from typing import Callable, Generic, Optional, TypeVar


T = TypeVar('T')


class Preloadable:
    @abstractmethod
    def preload(self):
        raise NotImplementedError()


class LazyLoaded(Generic[T]):
    """A value built at most once, however many callers ask for it at once.

    The lock is per instance rather than shared: a model artefact can take
    minutes to download and convert, and one lock over every lazy value in the
    process would make that wait everyone else's.
    """

    def __init__(self, factory: Callable[[], T]):
        super().__init__()
        self.factory = factory
        self._lock = threading.Lock()
        self._value: Optional[T] = None

    def __repr__(self) -> str:
        return '%s(_value=%r)' % (
            type(self).__name__, self._value
        )

    @property
    def is_loaded(self) -> bool:
        return self._value is not None

    def get(self) -> T:
        value = self._value
        if value is None:
            with self._lock:
                value = self._value
                if value is None:
                    value = self.factory()
                    assert value is not None
                    self._value = value
        return value
