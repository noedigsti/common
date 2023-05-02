import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Define the function and its derivative
def f(x):
    return x ** 4 - 4 * x ** 3 + 4 * x - 2

def df_dx(x):
    x_var = tf.Variable(x)
    with tf.GradientTape() as tape:
        y = f(x_var)
    return tape.gradient(y, x_var)

# Generate x values
x_values = np.linspace(-2, 4, num=100)

# Evaluate the function and its derivative at each x value
y_values = [f(x) for x in x_values]
dy_dx_values = [df_dx(x).numpy() for x in x_values]

# Find the indices where the sign of the derivative changes
sign_changes = np.where(np.diff(np.sign(dy_dx_values)))[0]

# Calculate the x values where the derivative crosses y=0
crossing_x_values = [0.5 * (x_values[i] + x_values[i+1]) for i in sign_changes]
crossing_y_values = [0 for _ in crossing_x_values]

# Plot the function and its derivative
plt.plot(x_values, y_values, label='f(x)')
plt.plot(x_values, dy_dx_values, label="f'(x)")

# Add a horizontal line for y=0
plt.axhline(y=0, color='k', linestyle='--')

# Plot the points where f'(x) cuts y=0
plt.scatter(crossing_x_values, crossing_y_values, color='red', zorder=3)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Function f(x) and its derivative f'(x) with y=0 crossings")
plt.show()
