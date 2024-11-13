from drawer import MatplotlibAnimationDrawer
from path import Path


def main():
    drawer = MatplotlibAnimationDrawer("./test.mp4")

    draw_path = Path.from_tuple_list([(3, 0.0), (2, 0.75), (2.3, 2), (3, 1.4), (3.7, 2), (4, 0.75)])
    drawer.draw(draw_path)


if __name__ == "__main__":
    main()
