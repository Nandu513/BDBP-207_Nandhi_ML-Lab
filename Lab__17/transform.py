import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def transform(X):
    x1 = X['x1'] ** 2
    x2 = np.sqrt(2) * X['x1'] * X['x2']
    x2 = X['x2'] ** 2

    transformed_df = pd.DataFrame({
        'x1^2': x1,
        'sqrt2_x1x2': x2,
        'x2^2': x2,
        'Label': X['Label']
    })

    return transformed_df


def main():
    data = {
        'x1': [1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16],
        'x2': [13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3],
        'Label': ['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Red', 'Red', 'Red', 'Red', 'Red',
                  'Red', 'Red', 'Red']
    }

    df = pd.DataFrame(data)

    X_3d=transform(df)
    print(X_3d)

    # Plot in 2D
    plt.figure(figsize=(6, 6))
    for label, color in zip(['Blue', 'Red'], ['blue', 'red']):
        subset = df[df['Label'] == label]
        plt.scatter(subset['x1'], subset['x2'], color=color, label=label)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Original 2D Points')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Transform and plot in 3D
    transformed = transform(df)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot transformed points
    for label, color in zip(['Blue', 'Red'], ['blue', 'red']):
        subset = transformed[transformed['Label'] == label]
        ax.scatter(
            subset['x1^2'],
            subset['sqrt2_x1x2'],
            subset['x2^2'],
            color=color, label=label, s=50
        )

    ax.set_xlabel('x1^2')
    ax.set_ylabel('√2 * x1 * x2')
    ax.set_zlabel('x2^2')
    ax.set_title('Transformed 3D Points')
    ax.view_init(elev=20, azim=100)  # Adjust view to better see separation
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()


