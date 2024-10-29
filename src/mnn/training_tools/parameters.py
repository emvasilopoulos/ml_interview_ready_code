# Copied from ultralytics
import torch


def get_params_grouped(model: torch.nn.Module):
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    parameters_grouped = [[], [], []]
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                parameters_grouped[2].append(param)
            elif isinstance(module, bn):  # weight (no decay)
                parameters_grouped[1].append(param)
            else:  # weight (with decay)
                parameters_grouped[0].append(param)
    return parameters_grouped
