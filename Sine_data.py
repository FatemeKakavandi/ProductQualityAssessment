import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Set random seed for reproducibility
np.random.seed(0)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from src_code import OCSVM, OCSVM_train, OCSVM_test, model_results
'''
# Define the parameters for the data
n_samples = 1000  # Number of samples
n_features = 1  # Number of features (1 for a single sine wave)


# Function to generate normal sine wave data
def generate_normal_sine_data(n_samples, frequency, amplitude, noise_std_dev):
    time = np.linspace(0, 10, n_samples)  # Time points
    normal_data = amplitude * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, noise_std_dev, n_samples)
    return normal_data


# Function to generate abnormal sine wave data
def generate_abnormal_sine_data(n_samples, frequency, amplitude, noise_std_dev):
    time = np.linspace(0, 10, n_samples)  # Time points
    abnormal_data = amplitude * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, noise_std_dev, n_samples)

    # Introduce an abnormal pattern
    abnormal_data[:n_samples // 10] += 2.0  # Example: Add an abnormal spike in the first 10% of data

    return abnormal_data


# Generate normal and abnormal sine wave data
normal_data = generate_normal_sine_data(n_samples, frequency=1.0, amplitude=1.0, noise_std_dev=0.1)
abnormal_data = generate_abnormal_sine_data(n_samples, frequency=1.0, amplitude=1.0, noise_std_dev=0.1)

# Plot the normal and abnormal data
plt.figure(figsize=(10, 6))
plt.plot(normal_data, label='Normal Data', color='blue')
plt.plot(abnormal_data, label='Abnormal Data', color='red')
plt.axvline(n_samples // 10, color='gray', linestyle='--', label='Abnormal Pattern Start')
plt.legend()
plt.title('Normal vs. Abnormal Sine Wave Data')
plt.xlabel('Time')
plt.ylabel('Value')
#plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Define the parameters for the data
n_samples = 100  # Number of samples
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


# Generate normal and abnormal sine wave data
normal_data = generate_normal_sine_data(n_samples, n_features, frequency=1.0, amplitude=1.0, noise_std_dev=0.1)
abnormal_data = generate_abnormal_sine_data(n_samples, n_features, frequency=1.0, amplitude=1.0, noise_std_dev=0.1)

# Combine the normal and abnormal data
data = np.vstack((normal_data, abnormal_data))
label = [0]*len(normal_data) + [1]*len(abnormal_data)

# Shuffle the data
X,y = shuffle(data,label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = OCSVM(nu=0.04)
model = OCSVM_train(X_train,model)
y_hat = OCSVM_test(X_test,model)
results = model_results(y_test, y_hat)
print(results)
'''
# Plot the data (optional for visualization)
plt.figure(figsize=(10, 6))
for sample in data:
    plt.plot(sample, color='blue' if sample in normal_data else 'red', alpha=0.5)
plt.axvline(n_features // 10, color='gray', linestyle='--', label='Abnormal Pattern Start')
plt.title('Stacked Sine Wave Data (Samples on Rows, Features on Columns)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
'''
