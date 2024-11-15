from drawer import MatplotlibAnimationDrawer, OpenCVAnimationDrawer
from key_point_extractor import CannyPointExtractor
from path_generator import GreedyCloserPointHeuristicPathGenerator
import cv2
import numpy as np


def main():
    """
    example_draw_path = Path.from_tuple_list([(3, 0.0), (2, 0.75), (2.3, 2), (3, 1.4), (3.7, 2), (4, 0.75)])
    """
    image = cv2.imread("./data/compmusos_logo.jpg", cv2.IMREAD_GRAYSCALE)
    points = CannyPointExtractor().extract(image)
    points = np.random.choice(points, size=4000, replace=False)
    draw_path = GreedyCloserPointHeuristicPathGenerator().generate(points)

    drawer = OpenCVAnimationDrawer("./test.mp4")
    drawer.draw_faster(draw_path)
    # from path import Path
    # drawer.draw(Path(draw_path.points[0:10]))


if __name__ == "__main__":
    main()
