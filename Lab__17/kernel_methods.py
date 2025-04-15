import numpy as np
import pandas as pd

def transform(X):
    x1_sq = X['x1']**2
    x2_cross = np.sqrt(2) * X['x1'] * X['x2']
    x2_sq = X['x2']**2

    return np.column_stack((x1_sq, x2_cross, x2_sq))

# Step 1: Create the data
data = pd.DataFrame({
    'x1': [3, 6],
    'x2': [10, 10]
})

# Step 2: Transform to higher dimension
transformed = transform(data)

# Step 3: Compute dot product
dot_product = np.dot(transformed[0], transformed[1])

# Step 4: Print results
print("Transformed vectors:")
print("Vector 1:", transformed[0])
print("Vector 2:", transformed[1])
print("\nDot Product in higher dimension:", dot_product)



# Given vectors
x1 = [3, 10]
x2 = [6, 10]

# Polynomial kernel function
def polynomial_kernel(a, b):
    term1 = a[0]**2 * b[0]**2
    term2 = 2 * a[0] * a[1] * b[0] * b[1]
    term3 = a[1]**2 * b[1]**2
    return term1 + term2 + term3

# Apply kernel
kernel_result = polynomial_kernel(x1, x2)
print("Polynomial Kernel result:", kernel_result)


