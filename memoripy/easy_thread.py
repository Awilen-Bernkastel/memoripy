#!/usr/bin/env python3

from threading import Thread

def thread(function, *args, **kwargs) -> Thread:
    def func_wrapper(*args, **kwargs):
        function(*args, **kwargs)

    _thread = Thread(target=func_wrapper, args=args, kwargs=kwargs)
    _thread.start()
    return _thread

if __name__ == "__main__":
    def print_thread(arg1, arg2):
        print(arg1, arg2)

    obj = thread(print_thread, "This is", arg2="a test.")
    obj.join()
