import numpy as np


class FourierInterpolator:
    def __init__(self):
        pass

    def interpolate(self, complex_vector: np.ndarray):
        pass


class FourierInterpolator1D(FourierInterpolator):
    def __init__(self):
        super().__init__()

    def interpolate(self, complex_vector: np.ndarray, steps: int = 500) -> (np.ndarray, np.ndarray):
        sample_rate = len(complex_vector)
        frequencies = np.fft.fftfreq(len(complex_vector), d=1 / sample_rate)
        complex_vector_fft = np.fft.fft(complex_vector)

        time_samples = np.linspace(0, 1, steps)
        component_sinusoid_x = []
        component_sinusoid_y = []

        for i in range(len(complex_vector)):
            component = complex_vector_fft[i]
            component_magnitude = np.abs(component)
            component_starting_phase = np.angle(component)
            component_frequency = frequencies[i]

            component_x = component_magnitude * np.cos(
                2 * np.pi * component_frequency * time_samples + component_starting_phase) / len(complex_vector)
            component_y = component_magnitude * np.sin(
                2 * np.pi * component_frequency * time_samples + component_starting_phase) / len(complex_vector)

            component_sinusoid_x.append(component_x)
            component_sinusoid_y.append(component_y)

        return component_sinusoid_x, component_sinusoid_y
