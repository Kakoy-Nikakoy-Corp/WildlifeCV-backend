from typing_extensions import overload, override
from
from pathlib import Path
from typing import Protocol, TypeAlias


import numpy as np


OpenCVFrame: TypeAlias = np.ndarray


class ModelInterface(Protocol):
    """Интерфейс класса модели."""
    def predict(self, frames: list[OpenCVFrame]) -> list[object]:
        ...

    @overload


class ModelDouble:
    """Заглушка модели."""
    def predict(self, frame: OpenCVFrame) -> object:
        ...
