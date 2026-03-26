#your answer
#task1_3.py
import numpy as np
arr = np.array([[10, 20, 30],
[40, 50, 60],
[70, 80, 90]])
print("Second row:", arr[1])
print("Third column:", arr[:, 2])
print("Submatrix (first 2 rows and cols):")
print(arr[:2, :2])