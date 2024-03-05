import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample sensor data
fire_sensor_data = [75.2, 80.1, 82.5, 79.3, 85.6]  # Example fire sensor readings
water_sensor_data = [0.3, 0.5, 0.2, 0.4, 0.6]  # Example water sensor readings
pressure_sensor_data = [100, 105, 98, 102, 108]  # Example pressure sensor readings

# Combine sensor data into a feature matrix
sensor_data = np.array([fire_sensor_data, water_sensor_data, pressure_sensor_data]).T

# Standardize the sensor data
scaler = StandardScaler()
standardized_sensor_data = scaler.fit_transform(sensor_data)

# Now you can use standardized_sensor_data for machine learning tasks
print("Standardized Sensor Data:")
print(standardized_sensor_data)
