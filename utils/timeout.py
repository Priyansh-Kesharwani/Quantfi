"""Centralized timeout utility for running functions with a deadline."""

import threading


class TimeoutError(Exception):
    pass


def run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

    if error[0] is not None:
        raise error[0]

    return result[0]
