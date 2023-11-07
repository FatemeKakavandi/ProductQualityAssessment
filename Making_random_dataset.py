import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use 'TkAgg' as an example backend, but you can choose another backend.
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from src_code import OCSVM, OCSVM_train, OCSVM_test, model_results
'''
# Function to generate normal time series data
def generate_normal_data(num_points, mean, std_dev):
    return np.random.normal(mean, std_dev, num_points)

# Function to generate abnormal time series data
def generate_abnormal_data(num_points, abnormal_period, mean_normal, std_dev_normal, mean_abnormal, std_dev_abnormal):
    data = []
    for i in range(num_points):
        if i < abnormal_period:
            data.append(np.random.normal(mean_normal, std_dev_normal))
        else:
            data.append(np.random.normal(mean_abnormal, std_dev_abnormal))
    return data

# Parameters for the data
num_points = 100
normal_mean = 0
normal_std_dev = 1
abnormal_mean = 10
abnormal_std_dev = 2
abnormal_period = 30  # Number of points before abnormal behavior starts

# Generate normal and abnormal data
normal_data = generate_normal_data(num_points, normal_mean, normal_std_dev)
abnormal_data = generate_abnormal_data(num_points, abnormal_period, normal_mean, normal_std_dev, abnormal_mean, abnormal_std_dev)

# Plot the normal and abnormal data
plt.figure(figsize=(10, 6))
plt.plot(normal_data, label='Normal Data', color='blue')
plt.plot(abnormal_data, label='Abnormal Data', color='red')
plt.axvline(abnormal_period, color='gray', linestyle='--', label='Abnormal Start')
plt.legend()
plt.title('Normal vs. Abnormal Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

from sklearn.datasets import make_blobs
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Define the parameters for the baseline dataset
n_samples_baseline = 1000
n_features = 10  # Number of features
n_informative = 5  # Number of informative features
baseline_data, _ = make_blobs(n_samples=n_samples_baseline, n_features=n_features, centers=1, cluster_std=1.0, random_state=0, n_informative=n_informative)

# Define the parameters for the abnormal dataset
n_samples_abnormal = 100
abnormal_data, _ = make_blobs(n_samples=n_samples_abnormal, n_features=n_features, centers=1, cluster_std=3.0, random_state=42, n_informative=n_informative)

# Combine the datasets
data = np.vstack((baseline_data, abnormal_data))

# Shuffle the data
np.random.shuffle(data)

# Separate features and labels (in this case, labels are not needed for training OCSVM, but can be used for testing)
X = data  # Features
y = np.zeros(data.shape[0])  # Labels (0 for baseline, 1 for abnormal)

# Now you can use X and y for training your OCSVM model
'''
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Define the dimensions of the dataset
n_samples_baseline = 100

n_samples_abnormal = 10
n_features = 1000  # Number of features

# Create a baseline dataset with random values for each feature
baseline_data = np.random.normal(0, 1, (n_samples_baseline, n_features))

# Create an abnormal dataset with different characteristics
abnormal_data = np.random.normal(10, 3, (n_samples_abnormal, n_features))

# Combine the datasets
data = np.vstack((baseline_data, abnormal_data))
label = [0]*len(baseline_data) + [1]*len(abnormal_data)

# Shuffle the data
X,y = shuffle(data,label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = OCSVM(nu=0.04)
model = OCSVM_train(X_train,model)
y_hat = OCSVM_test(X_test,model)
results = model_results(y_test, y_hat)
print(results)

# Separate features and labels (labels are not needed for training OCSVM but can be used for testing)
#X = data  # Features
#y = np.zeros(data.shape[0])  # Labels (0 for baseline, 1 for abnormal)

# Now you can use X and y for training your OCSVM model
