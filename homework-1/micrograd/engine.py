"""
This module's type annotations assume you use python3.7+: https://www.python.org/dev/peps/pep-0563
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Set, Union, Tuple


class Value:
    """ Stores a single scalar value and its gradient """

    data: float
    grad: float

    # Internal variables used for autograd graph construction
    _backward: Callable
    _prev: Set
    _op: str

    def __init__(self, data: float, children: Tuple = (), op: str = ""):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(children)
        self._op = op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other: Union[int, float, Value]) -> Value:
        # Cast other to Value
        other = other if isinstance(other, Value) else Value(other)

        # Add Values' datas
        out = Value(data=self.data + other.data, children=(self, other), op="+")

        # Backward of addition is addition
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Union[int, float, Value]) -> Value:
        # Cast other to Value
        other = other if isinstance(other, Value) else Value(other)

        # Multiply Values' datas
        out = ...

        # Backward of multiplication is ...
        def _backward():
            self.grad += ...
            other.grad += ...

        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> Value:
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"

        # Operate on Values' datas
        out = ...

        # Backward of pow is ...
        def _backward():
            self.grad += ...

        out._backward = _backward

        return out

    def exp(self) -> Value:
        out = ...

        def _backward():
            self.grad += ...

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = ...

        def _backward():
            self.grad += ...

        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            # YOUR CODE GOES HERE
            ...

    def __neg__(self) -> Value:  # -self
        return self * -1

    def __radd__(self, other) -> Value:  # other + self
        return self + other

    def __sub__(self, other) -> Value:  # self - other
        return self + (-other)

    def __rsub__(self, other) -> Value:  # other - self
        return other + (-self)

    def __rmul__(self, other) -> Value:  # other * self
        return self * other

    def __truediv__(self, other) -> Value:  # self / other
        return self * other ** -1

    def __rtruediv__(self, other) -> Value:  # other / self
        return other * self ** -1

    def __le__(self, other) -> bool:
        if isinstance(other, Value):
            return self.data <= other.data
        return self.data <= other

    def __lt__(self, other) -> bool:
        if isinstance(other, Value):
            return self.data < other.data
        return self.data < other

    def __gt__(self, other) -> bool:
        if isinstance(other, Value):
            return self.data > other.data
        return self.data > other

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"


class Tensor:
    """
    Tensor is a kinda array with expanded functianality.

    Tensor is very convenient when it comes to matrix multiplication,
    for example in Linear layers.
    """

    # Tensor holds an array of Values
    data: np.ndarray[Value]

    def __init__(self, data: List[Value]):
        self.data = np.array(data)

    def __add__(self, other: Union[Tensor, List[Value]]) -> Tensor:
        if isinstance(other, Tensor):
            assert (
                self.shape() == other.shape()
            ), f"self.shape={self.shape}, other.shape={other.shape}"
            return Tensor(np.add(self.data, other.data))

        return Tensor(self.data + other)

    def __mul__(self, other: Union[Tensor, List[Value]]) -> Tensor:
        return ...

    def __truediv__(self, other: Union[Tensor, List[Value]]) -> Tensor:
        return ...

    def __floordiv__(self, other: Union[Tensor, List[Value]]) -> Tensor:
        return ...

    def __radd__(self, other: Union[Tensor, List[Value]]) -> Tensor:
        return ...

    def __rmull__(self, other: Union[Tensor, List[Value]]) -> Tensor:
        return ...

    def exp(self) -> Tensor:
        return ...

    def dot(self, other: Union[Tensor, List[Value]]) -> Tensor:
        if isinstance(other, Tensor):
            return ...
        return ...

    def shape(self) -> np.ndarray:
        return self.data.shape

    def argmax(self, dim: int = None) -> Tensor:
        return ...

    def max(self, dim: int = None) -> Tensor:
        return ...

    def reshape(self, *args, **kwargs) -> Tensor:
        self.data = ...
        return self

    def backward(self):
        for value in self.data.flatten():
            value.backward()

    def parameters(self) -> List[Value]:
        return list(self.data.flatten())

    def __repr__(self) -> str:
        return "Tensor\n" + str(self.data)

    def __getitem__(self, item) -> Value:
        return self.data[item]

    def item(self) -> Value:
        return self.data.flatten()[0].data
