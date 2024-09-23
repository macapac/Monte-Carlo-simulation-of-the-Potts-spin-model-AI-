import numpy as np
import matplotlib.pyplot as plt

# Create a common range for the X-axis
x1 = np.linspace(0, 10, 100)
x2 = np.linspace(0, 2*np.pi, 100)

# Define the two functions
y1 = np.sin(x1)  # Sine function for the first range
y2 = np.cos(x2)  # Cosine function for the second range

# Create the plot
plt.plot(x1, y1, 'b-', label="sin(x1)")
plt.plot(x2, y2, 'r-', label="cos(x2)")

# Labels and title
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Sine and Cosine on a Common X-Axis')

# Add a legend
plt.legend()

# Display the plot
plt.show()
