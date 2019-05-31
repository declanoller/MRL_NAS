import functools
import time
import matplotlib.pyplot as plt




def timer(func):
    # Print the runtime of the decorated function
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f'\n\nFinished function {func.__name__!r} in {run_time:.2f} secs\n')
        return(value)
    return(wrapper_timer)











#
