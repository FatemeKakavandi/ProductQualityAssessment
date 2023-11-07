import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Set random seed for reproducibility
np.random.seed(0)


n_normal_samples = 100  # Number of  normal samples
n_abnormal_samples = 10 # Number of abnormal samples
n_features = 100  # Length of time series data


# Function to generate normal sine wave data
def generate_normal_sine_data(n_samples, n_features, frequency, amplitude, noise_std_dev):
    time = np.linspace(0, 10, n_features)  # Time points
    normal_data = []
    for _ in range(n_samples):
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, noise_std_dev, n_features)
        normal_data.append(sine_wave)
    return np.array(normal_data)


# Function to generate abnormal sine wave data
def generate_abnormal_sine_data(n_samples, n_features, frequency, amplitude, noise_std_dev):
    time = np.linspace(0, 10, n_features)  # Time points
    abnormal_data = []
    for _ in range(n_samples):
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, noise_std_dev, n_features)

        # Introduce an abnormal pattern
        abnormal_data.append(sine_wave)

    abnormal_data = np.array(abnormal_data)
    abnormal_data[:, :n_features // 10] += 2.0  # Example: Add an abnormal spike in the first 10% of data

    return abnormal_data


## Replace your data here
base_data = generate_normal_sine_data(n_normal_samples, n_features, frequency=1.0, amplitude=1.0, noise_std_dev=0.1)
faulty_data = generate_abnormal_sine_data(n_abnormal_samples, n_features, frequency=1.0, amplitude=1.0, noise_std_dev=0.1)

y_nomral = [1]*n_normal_samples

#Plotting Data:
data = np.vstack((base_data, faulty_data))

plt.figure(figsize=(10, 6))
for sample in data:
    plt.plot(sample, color='blue' if sample in base_data else 'red', alpha=0.5)
plt.axvline(n_features // 10, color='gray', linestyle='--', label='Abnormal Pattern Start')
plt.title('Stacked Sine Wave Data (Samples on Rows, Features on Columns)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('./figures/data.png')

plt.show()
