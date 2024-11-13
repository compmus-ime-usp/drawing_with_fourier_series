from abc import ABC
from path import Path
from fourier_interpolator import FourierInterpolator1D
import numpy as np
import matplotlib.pyplot as plt
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

    def draw(self, path: Path, ritmo: int = 40, fps: int = 60, bitrate: int = 1800, artist: str = None) -> None:
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
