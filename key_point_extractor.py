from abc import ABC
from point import Point
from typing import List
import numpy as np


class KeyPointExtractor(ABC):
    def __init__(self):
        pass

    def extract(self, image: np.ndarray) -> List[Point]:
        pass
