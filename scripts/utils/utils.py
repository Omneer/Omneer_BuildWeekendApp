import time
from functools import wraps

def timeit(func):
    """
    Decorator to time a function.

    :param func: The function to time.
    :return: The wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # get the current time
        result = func(*args, **kwargs)  # call the original function
        end_time = time.time()  # get the current time again after function execution
        elapsed_time = end_time - start_time  # calculate the difference in time

        # print the function name and the time it took to execute
        print(f"Function '{func.__name__}' took {elapsed_time} seconds to complete.")
        return result

    return wrapper
