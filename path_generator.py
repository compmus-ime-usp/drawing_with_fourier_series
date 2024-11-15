from abc import ABC
from path import Path
from typing import List
from point import Point
import numpy as np


class PathGenerator(ABC):
    def __init__(self):
        pass

    def generate(self, points: List[Point]) -> Path:
        pass


class TravellingSalesmanPathGenerator(PathGenerator):
    """
    This will only be a theoretical implementation, cause for the number of points we normally have
    its impossible to run it in a feasible time.
    """
    def __init__(self):
        super().__init__()

    def generate(self, points: List[Point]) -> Path:
        pass


class GreedyCloserPointHeuristicPathGenerator(PathGenerator):
    """
    This is a simple heuristic that uses a greedy approach to solve the problem in a suboptimal way.
    At least it will run faster!
    Time complexity: O(n^2), where n is the number of points.
    """
    def __init__(self):
        super().__init__()

    def generate(self, points: List[Point]) -> Path:
        distance_matrix = self.__create_distance_matrix(points)
        ordered_points = []
        current_index = 0
        while True:
            ordered_points.append(points[current_index])
            distance_matrix[:, current_index] = np.inf
            min_distance_index = np.argmin(distance_matrix[current_index])

            if distance_matrix[current_index, min_distance_index] == np.inf:
                break

            current_index = min_distance_index

        return Path(points=ordered_points)

    @staticmethod
    def __create_distance_matrix(points: List[Point]) -> np.ndarray:
        distance_matrix = np.zeros(shape=(len(points), len(points)))
        for i, first_point in enumerate(points):
            for j, second_point in enumerate(points):
                distance_matrix[i, j] = GreedyCloserPointHeuristicPathGenerator.__euclidian_distance(first_point,
                                                                                                     second_point)

        return distance_matrix

    @staticmethod
    def __euclidian_distance(first_point: Point, second_point: Point) -> float:
        return ((first_point.x - second_point.x) ** 2 + (first_point.y - second_point.y) ** 2) ** 0.5
