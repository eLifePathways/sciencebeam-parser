import logging
import inspect
from functools import wraps
from typing import Callable, TypeVar, Union


LOGGER = logging.getLogger(__name__)


T_WrappedCallable = TypeVar('T_WrappedCallable', bound=Callable)
T_WrappedCallableOrType = Union[T_WrappedCallable, type]


def wrap_class_methods(cls, wrapper):
    for key, value in cls.__dict__.items():
        if callable(value):
            setattr(cls, key, wrapper(value))
    return cls


def log_on_exception(f: T_WrappedCallableOrType) -> T_WrappedCallableOrType:
    """
    Wraps function to log error on exception.
    That is useful for tests that log a lot of things,
    and pytest displaying the test failure at the top of the method.
    (there doesn't seem to be an option to change that)
    """
    if inspect.isclass(f):
        return wrap_class_methods(f, log_on_exception)

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            # returned, since wrapping a class wraps its helpers as well as its tests
            return f(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception('failed due to %s', repr(e))
            raise
    return wrapper
