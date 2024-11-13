from typing import Tuple


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    @classmethod
    def from_tuple(cls, tuple_: Tuple[float, float]):
        return cls(tuple_[0], tuple_[1])

    def to_complex(self):
        return complex(self.x, self.y)
