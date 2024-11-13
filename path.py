import numpy as np
from point import Point
from typing import List


class Path:
    def __init__(self, points: List[Point]):
        self.points = points

    @classmethod
    def from_tuple_list(cls, list_: list):
        points = []
        for item in list_:
            points.append(Point.from_tuple(item))
        return cls(points)

    def to_complex_vector(self):
        return np.array([point.to_complex() for point in self.points], dtype=complex)
