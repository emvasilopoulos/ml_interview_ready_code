import time
from typing import Any, Callable, Tuple
import torch
import torch.nn

FLOATING_POINT_PRECISIONS = {"float32": torch.float32, "float16": torch.float16}


def _apply_to_model(
    model: torch.nn.Module,
    func: Callable[[torch.nn.Module, Tuple[Any]], None],
    args: Tuple[Any],
):
    """
    Apply a function to all submodules of a PyTorch model.

    Args:
        model (torch.nn.Module): model
        func (Callable[[torch.nn.Module], None]): a callable function that takes a torch.nn.Module as input
    """
    func(model, args)
    for child_module in model.children():
        if isinstance(child_module, torch.nn.Module):
            _apply_to_model(child_module, func, args)


def convert_model_to_dtype(model: torch.nn.Module, dtype: torch.dtype):
    """Convert PyTorch model to dtype."""

    def convert_to_dtype(module: torch.nn.Module, args: Tuple[Any]):
        module.to(dtype=dtype)

    _apply_to_model(model, convert_to_dtype, (None,))


def check_model_dtype(model: torch.nn.Module, dtype: torch.dtype):
    """Check if all parameters of a PyTorch model are of a given dtype."""
    parameters_counter = 0

    class ParameterCounter:
        value = 0

    parameters_counter = ParameterCounter()

    def check_dtype(module: torch.nn.Module, args: Tuple[ParameterCounter]):
        params_counter = args[0]
        for param in module.parameters():
            params_counter.value += 1
            assert (
                param.dtype == dtype
            ), f"Parameter {param} has dtype {param.dtype}, expected {dtype}"

    _apply_to_model(model, check_dtype, (parameters_counter,))
    print(f"Checked {parameters_counter.value} parameters")


def check_model_dtype2(model: torch.nn.Module, dtype: torch.dtype):
    params_counter = 0
    correct_params_counter = 0
    wrong_params_counter = 0
    for param in model.parameters(recurse=True):
        if param.requires_grad:
            params_counter += 1
            if param.dtype == dtype:
                correct_params_counter += 1
            else:
                wrong_params_counter += 1
    print(f"Correct: {correct_params_counter}, Wrong: {wrong_params_counter}")
    print(f"Checked {params_counter} parameters")
    assert correct_params_counter == params_counter, "Some parameters have wrong dtype"


def initialize_weights(tensor_shape: torch.Size):
    return torch.rand(tensor_shape)


def inference_test(image: torch.Tensor, model: torch.nn.Module):
    t0 = time.time()
    output = model(image)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    print("Time taken:", t1 - t0, "seconds")
    print("Model's output shape:", output.shape)
    traced_model = torch.jit.trace(model.forward, image, check_trace=True, strict=True)
    return traced_model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModuleTimer:
    message = ""

    def set_message(self, message: str):
        self.message = message

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end = time.time()
        if self.message:
            print(self.message, end=" | ")
        print(f"Time taken: {self.end - self.start:.4f} seconds")
        return False
