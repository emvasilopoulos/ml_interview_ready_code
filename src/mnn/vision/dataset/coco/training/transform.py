import abc

import torch


class BaseIOTransform:

    @abc.abstractmethod
    def transform_input(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def transform_output(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def update_transform_configuration(self) -> None:
        pass
