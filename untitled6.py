import numpy as np
import matplotlib.pyplot as plt

# Generate x values (must be ≥ 0 for sqrt)
x = np.linspace(0, 10, 200)
y = np.sqrt(x)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x, y, label=r'$y = \sqrt{x}$', color='teal', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Square Root Dependence')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
