import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

points = [(3, 0.0), (2, 0.75), (2.3, 2), (3, 1.4), (3.7, 2), (4, 0.75)]
sample_rate = len(points)

y_max = 1.5
y_min = -1.5
x_max = 1.5
x_min = -1.5
tolerance = 0.05
number_of_points = len(points)


def points_to_complex_vector(points):
    complex_data = np.zeros(len(points), dtype=np.complex64)
    for i in range(len(points)):
        complex_data[i] = points[i][0] + 1j * points[i][1]
    return complex_data


complex_vector = points_to_complex_vector(points)
frequencies = np.fft.fftfreq(len(complex_vector), d=1 / sample_rate)
complex_vector_fft = np.fft.fft(complex_vector)

time_samples = np.linspace(0, 1, 1000)
component_sinusoid_x = []
component_sinusoid_y = []

for i in range(len(points)):
    component = complex_vector_fft[i]
    component_magnitude = np.abs(component)
    component_starting_phase = np.angle(component)
    component_frequency = frequencies[i]

    # Compute x and y components for each frequency component
    component_x = component_magnitude * np.cos(
        2 * np.pi * component_frequency * time_samples + component_starting_phase) / len(points)
    component_y = component_magnitude * np.sin(
        2 * np.pi * component_frequency * time_samples + component_starting_phase) / len(points)

    component_sinusoid_x.append(component_x)
    component_sinusoid_y.append(component_y)

# Coordinates of original points
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(0, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Fourier Series Approximation Animation')
ax.grid(True)

# Plot points and initialize the Fourier series line
# ax.scatter(x_values, y_values, color='blue', label='Points')
line, = ax.plot([], [], lw=2, color='red', label='Fourier Series Approximation')
ax.legend()


# Animation function
def animate(i):
    # Sum up components up to the ith frequency
    x_sum = np.sum(np.array(component_sinusoid_x)[:, 0:i + 1], axis=0)
    y_sum = np.sum(np.array(component_sinusoid_y)[:, 0:i + 1], axis=0)

    # Update the line data
    line.set_data(x_sum, y_sum)
    return line,


# Create the animation
print(len(component_sinusoid_x[0]))
ani = FuncAnimation(fig, animate, frames=len(component_sinusoid_x[0]), interval=0, blit=True)

plt.show()
