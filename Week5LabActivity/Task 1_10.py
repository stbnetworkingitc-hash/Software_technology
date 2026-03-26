#your answer
import numpy as np

# Given data
growth_factors = np.array([0.7, 0.2, 0.5])

rainfall = np.array([
    [40, 50, 60],
    [20, 35, 25],
    [30, 40, 55]
])

# Calculate height increase using broadcasting
height_increase = growth_factors[:, np.newaxis] * rainfall

# Print result
print(height_increase)