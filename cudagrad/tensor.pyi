from __future__ import annotations

from typing import Iterable, List, Union

class _DataProxy:
    def __call__(self) -> List[float]: ...
    def __getitem__(self, index: Union[Iterable[int], slice]) -> Tensor: ...
    def __setitem__(self, index: Union[Iterable[int], slice], value: float | Tensor) -> None: ...

class _GradProxy:
    def __call__(self) -> Tensor: ...
    def __getitem__(self, index: Union[Iterable[int], slice]) -> Tensor: ...
    def __setitem__(self, index: Union[Iterable[int], slice], value: float | Tensor) -> None: ...

class Tensor:
    def __init__(self, sizes: Iterable[int], values: Iterable[float]): ...
    def backward(self) -> None: ...
    def zero_grad(self) -> None: ...
    def graph(self) -> None: ...
    def get_shared(self) -> Tensor: ...
    def item(self) -> float: ...
    def sum(self) -> Tensor: ...
    def relu(self) -> Tensor: ...
    def sigmoid(self) -> Tensor: ...
    @property
    def data(self) -> _DataProxy: ...
    @property
    def grad(self) -> _GradProxy: ...
    @property
    def size(self) -> List[int]: ...
    @staticmethod
    def zeros(sizes: Iterable[int]) -> Tensor: ...
    @staticmethod
    def ones(sizes: Iterable[int]) -> Tensor: ...
    @staticmethod
    def rand(sizes: Iterable[int]) -> Tensor: ...
    @staticmethod
    def explode(sizes: Iterable[int], value: float) -> Tensor: ...
    def __eq__(self, other: Tensor) -> Tensor: ...  # type: ignore [override]
    def __ne__(self, other: Tensor) -> Tensor: ...  # type: ignore [override]
    def __add__(self, other: Tensor) -> Tensor: ...
    def __sub__(self, other: Tensor) -> Tensor: ...
    def __mul__(self, other: Tensor) -> Tensor: ...
    def __truediv__(self, other: Tensor) -> Tensor: ...
    def __matmul__(self, other: Tensor) -> Tensor: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

def hello() -> str: ...