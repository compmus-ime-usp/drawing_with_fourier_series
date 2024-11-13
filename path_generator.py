from abc import ABC
from path import Path
from typing import List
from point import Point


class PathGenerator(ABC):
    def __init__(self):
        pass

    def generate(self, points: List[Point]) -> Path:
        pass
