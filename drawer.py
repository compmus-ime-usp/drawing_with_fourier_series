from abc import ABC
from path import Path
from fourier_interpolator import FourierInterpolator1D
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation, FFMpegWriter


class Drawer(ABC):
    def __init__(self):
        pass

    def draw(self, path: Path) -> None:
        pass


class MatplotlibAnimationDrawer(Drawer):
    def __init__(self, output_file: str, display: bool = False, save: bool = True,
                 margin: tuple = (1, 1)):
        super().__init__()
        self.output_file = output_file
        self.display = display
        self.save = save
        self.margin = margin

    def draw(self, path: Path, ritmo: int = 100, fps: int = 60, bitrate: int = 1800, artist: str = None,
             circles: bool = False) -> None:
        fourier_interpolator = FourierInterpolator1D()
        component_sinusoid_x, component_sinusoid_y = fourier_interpolator.interpolate(path.to_complex_vector())

        x_min = np.min(np.sum(np.array(component_sinusoid_x), axis=0))
        x_max = np.max(np.sum(np.array(component_sinusoid_x), axis=0))
        y_min = np.min(np.sum(np.array(component_sinusoid_y), axis=0))
        y_max = np.max(np.sum(np.array(component_sinusoid_y), axis=0))

        fig, ax = plt.subplots(figsize=(np.ceil(x_max - x_min) + self.margin[0],
                                        np.ceil(y_max - y_min) + self.margin[1]))

        ax.set_xlim(x_min - self.margin[0],
                    x_max + self.margin[0])

        ax.set_ylim(y_min - self.margin[1],
                    y_max + self.margin[1])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Fourier Series Approximation Animation')
        ax.grid(True)

        line,  = ax.plot([], [], lw=2, color='red', label='Fourier Series Approximation')
        ax.legend()

        def animate(i):
            x_sum = np.sum(np.array(component_sinusoid_x)[:, 0:i + 1], axis=0)
            y_sum = np.sum(np.array(component_sinusoid_y)[:, 0:i + 1], axis=0)

            line.set_data(x_sum, y_sum)
            return line,

        ani = FuncAnimation(fig, animate, frames=len(component_sinusoid_x[0]), interval=100 // ritmo, blit=True)

        if self.display:
            plt.show()

        if self.save:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist=artist), bitrate=bitrate)
            ani.save(self.output_file, writer=writer)


class OpenCVAnimationDrawer(Drawer):
    def __init__(self, output_file: str, display: bool = False, save: bool = True,
                 margin: tuple = (50, 50), frame_size: tuple = (600, 600)):
        super().__init__()
        self.output_file = output_file
        self.display = display
        self.save = save
        self.margin = margin
        self.frame_size = frame_size

    def draw(self, path: Path, ritmo: int = 100, fps: int = 120) -> None:
        fourier_interpolator = FourierInterpolator1D()
        component_sinusoid_x, component_sinusoid_y = fourier_interpolator.interpolate(path.to_complex_vector())

        x_min = np.min(np.sum(np.array(component_sinusoid_x), axis=0))
        x_max = np.max(np.sum(np.array(component_sinusoid_x), axis=0))
        y_min = np.min(np.sum(np.array(component_sinusoid_y), axis=0))
        y_max = np.max(np.sum(np.array(component_sinusoid_y), axis=0))

        scale_x = (self.frame_size[0] - 2 * self.margin[0]) / (x_max - x_min)
        scale_y = (self.frame_size[1] - 2 * self.margin[1]) / (y_max - y_min)

        writer = None
        if self.save:
            writer = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, self.frame_size)

        for i in range(len(component_sinusoid_x[0])):
            frame = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255

            # Scale and offset values
            x_sum = np.sum(np.array(component_sinusoid_x)[:, 0:i + 1], axis=0)
            y_sum = np.sum(np.array(component_sinusoid_y)[:, 0:i + 1], axis=0)
            x_points = ((x_sum - x_min) * scale_x + self.margin[0]).astype(int)
            y_points = ((y_sum - y_min) * scale_y + self.margin[1]).astype(int)

            # Draw Fourier Series Circles
            first_point = [0, 0]
            lines_to_draw = []
            for j in range(np.array(component_sinusoid_x).shape[0]):
                second_point = (first_point[0] + np.array(component_sinusoid_x)[j, i],
                                first_point[1] + np.array(component_sinusoid_y)[j, i])
                lines_to_draw.append([first_point, second_point])

                # Calculate scaled Euclidean distance as radius
                dx = (second_point[0] - first_point[0]) * scale_x
                dy = (second_point[1] - first_point[1]) * scale_y
                radius = int(np.sqrt(dx ** 2 + dy ** 2))

                # Scale and draw the circle
                center = (int((first_point[0] - x_min) * scale_x + self.margin[0]),
                          int((first_point[1] - y_min) * scale_y + self.margin[1]))
                cv2.circle(frame, center, radius, (0, 255, 0), 1)

                first_point = second_point

            # Draw each line in lines_to_draw with scaling and margin adjustments
            for line_start, line_end in lines_to_draw:
                start = (int((line_start[0] - x_min) * scale_x + self.margin[0]),
                         int((line_start[1] - y_min) * scale_y + self.margin[1]))
                end = (int((line_end[0] - x_min) * scale_x + self.margin[0]),
                       int((line_end[1] - y_min) * scale_y + self.margin[1]))
                cv2.line(frame, start, end, (0, 0, 0), 2)

            # Draw the Fourier approximation line up to the current frame
            for j in range(1, len(x_points)):
                cv2.line(frame, (x_points[j - 1], y_points[j - 1]), (x_points[j], y_points[j]), (0, 0, 255), 2)

            if self.display:
                cv2.imshow('Fourier Series Approximation Animation', cv2.flip(frame, 0))
                if cv2.waitKey(100 // ritmo) & 0xFF == ord('q'):
                    break

            if self.save:
                writer.write(cv2.flip(frame, 0))

        if self.display:
            cv2.destroyAllWindows()

        if self.save:
            writer.release()
