# Importing numpy for matrix operations
import numpy as np

# Function to perform matrix operations
def matrix_ops(A):
    b = A.T  # Transpose of the matrix A
    print(np.dot(A, b))  # Perform matrix multiplication between A and its transpose

# Main function to create a sample matrix and call matrix_ops
def main():
    a = np.array([[1, 2, 3], [4, 5, 6]])  # Creating a 2x3 matrix
    print(matrix_ops(a))  # Call matrix_ops with matrix 'a'

# Entry point of the program
if __name__ == '__main__':
    main()

# Importing matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Function to plot a linear function y = 2x + 3
def plot_map1(x):
    y = (2 * x) + 3  # Linear equation y = 2x + 3
    for i in x:  # Iterating over x values (unnecessary, as plt.plot already works with arrays)
        plt.plot(x, y)  # Plot the function y against x
    plt.show()  # Display the plot

# Main function to generate x values and call plot_map1
def main():
    b = np.linspace(-100, 100, 100)  # Create 100 evenly spaced values between -100 and 100
    print(plot_map1(b))  # Call plot_map1 with b

# Entry point of the program
if __name__ == '__main__':
    main()

# Function to plot a quadratic function y = 2x^2 + 3x + 4
def plot_map2(x):
    y = (2 * (x ** 2)) + 3 * x + 4  # Quadratic equation y = 2x^2 + 3x + 4
    for i in x:  # Iterating over x values (unnecessary, as plt.plot already works with arrays)
        plt.plot(x, y)  # Plot the function y against x
    plt.show()  # Display the plot

# Main function to generate x values and call plot_map2
def main():
    b = np.linspace(-10, 10, 100)  # Create 100 evenly spaced values between -10 and 10
    print(plot_map2(b))  # Call plot_map2 with b

# Entry point of the program
if __name__ == '__main__':
    main()

# Function to plot a Gaussian (normal distribution) function
def plot_map2(x):
    # Gaussian function with mean=0 and standard deviation=15
    y = (1 / (15 * np.sqrt(2 * np.pi))) * (2.71828) ** (-0.5 * ((x - 0) / 15) ** 2)
    for i in x:  # Iterating over x values (unnecessary, as plt.plot already works with arrays)
        plt.plot(x, y)  # Plot the Gaussian function
    plt.show()  # Display the plot

# Main function to generate x values and call plot_map2
def main():
    b = np.linspace(-100, 100, 100)  # Create 100 evenly spaced values between -100 and 100
    print(plot_map2(b))  # Call plot_map2 with b

# Entry point of the program
if __name__ == '__main__':
    main()

# Function to plot both a quadratic and linear function on the same graph
def plot_map3(x):
    y = x ** 2  # y = x^2 (quadratic function)
    z = 2 * x  # z = 2x (linear function)
    for i in x:  # Iterating over x values (unnecessary, as plt.plot already works with arrays)
        plt.plot(x, y)  # Plot the quadratic function
        plt.plot(x, z, marker='*')  # Plot the linear function with a '*' marker
    plt.show()  # Display the plot

# Main function to generate x values and call plot_map3
def main():
    b = np.linspace(-100, 100, 100)  # Create 100 evenly spaced values between -100 and 100
    print(plot_map3(b))  # Call plot_map3 with b

# Entry point of the program
if __name__ == '__main__':
    main()

# Function to plot both a quadratic and linear function with additional styling
def plot_map3(x):
    y = x ** 2  # y = x^2 (quadratic function)
    z = 2 * x  # z = 2x (linear function)
    for i in x:  # Iterating over x values (unnecessary, as plt.plot already works with arrays)
        plt.plot(x, y)  # Plot the quadratic function
        plt.plot(x, z, marker='*', ms=8, mfc='red')  # Plot the linear function with a '*' marker and red fill
    plt.show()  # Display the plot

# Main function to generate x values and call plot_map3
def main():
    b = np.linspace(-10, 10, 10)  # Create 10 evenly spaced values between -10 and 10
    print(plot_map3(b))  # Call plot_map3 with b

# Entry point of the program
if __name__ == '__main__':
    main()
