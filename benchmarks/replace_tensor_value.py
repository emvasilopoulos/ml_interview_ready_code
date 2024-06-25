import torch
import time


def benchmark1():
    # Original tensor
    tensor = torch.tensor([-1, 1, -1, -1])

    # Timing the in-place modification method
    start_time = time.time()
    tensor[tensor == -1] = 0
    end_time = time.time()

    in_place_duration = end_time - start_time
    print(f"In-place modification duration: {in_place_duration:.10f} seconds")


def benchmark2():
    # Original tensor
    tensor = torch.tensor([-1, 1, -1, -1])

    # Timing the torch.where method
    start_time = time.time()
    modified_tensor = torch.where(tensor == -1, torch.tensor(0), tensor)
    end_time = time.time()

    where_duration = end_time - start_time
    print(f"torch.where duration: {where_duration:.10f} seconds")


def benchmark3():
    # Create a larger tensor for more meaningful timing
    large_tensor = torch.randint(-1, 2, (1000000,))

    # Timing the torch.where method
    start_time = time.time()
    large_modified_tensor = torch.where(
        large_tensor == -1, torch.tensor(0), large_tensor
    )
    end_time = time.time()
    where_duration = end_time - start_time

    # Create a new large tensor for in-place modification
    large_tensor = torch.randint(-1, 2, (1000000,))

    # Timing the in-place modification method
    start_time = time.time()
    large_tensor[large_tensor == -1] = 0
    end_time = time.time()
    in_place_duration = end_time - start_time

    print(f"torch.where duration: {where_duration:.10f} seconds")
    print(f"In-place modification duration: {in_place_duration:.10f} seconds")


if __name__ == "__main__":
    for i in range(10):
        print("=============")
        benchmark2()
        benchmark1()
        print("-------------")
        benchmark3()
        print()
