
from concurrent.futures import ProcessPoolExecutor


# Example usage
def worker(x):
    """Worker function that must be defined at module level."""
    import os
    import time

    pid = os.getpid()
    time.sleep(1)  # Simulate work
    return f"Process {pid} processed {x}"


def simple_parallel_test():
    """Simple test to verify multiprocessing is working."""
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(8)))
    for result in results:
        print(result)


if __name__ == "__main__":
    simple_parallel_test()
