from abc import ABC
from point import Point
from typing import List
import numpy as np
import cv2


class KeyPointExtractor(ABC):
    def __init__(self):
        pass

    def extract(self, image: np.ndarray) -> List[Point]:
        pass


class CannyPointExtractor(KeyPointExtractor):
    def __init__(self):
        super().__init__()

    def extract(self, image: np.ndarray) -> List[Point]:
        edges = cv2.Canny(image, threshold1=100, threshold2=200)
        indices = np.argwhere(edges == 255)
        points_list = [Point.from_tuple(tuple((idx[1], idx[0]))) for idx in indices]

        return points_list
