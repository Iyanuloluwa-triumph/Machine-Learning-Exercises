import matplotlib.pyplot as plt
import numpy as np

# Create idealized dome shape
theta = np.linspace(0, np.pi, 100)
v_left = 0.1 - 0.09 * np.cos(theta)
v_right = 0.1 + 0.09 * np.cos(theta)
T_dome = 150 + 100 * np.sin(theta)  # dome spans roughly from 150°C to 250°C

# Process lines
v1 = 0.22  # superheated point to the right of the dome
v2 = 0.1  # saturated vapor point on right edge of dome
T1 = 400  # starting temperature
T2 = 180  # saturated temperature (horizontal line ends here)
T3 = 150  # final temp after constant volume cooling

# Create figure
plt.figure(figsize=(10, 6))

# Plot the dome
plt.plot(v_left, T_dome, 'b')
plt.plot(v_right, T_dome, 'r')
plt.fill_betweenx(T_dome, v_left, v_right, color='lightblue', alpha=0.3, label="2-Phase Region")

# Process 1–2: Constant pressure (horizontal line)
plt.plot([v1, v2], [T2, T2], 'k--', lw=2)
plt.text((v1 + v2) / 2, T2 + 5, "Process 1–2 (P = const)", ha='center')

# Process 2–3: Constant volume (vertical line)
plt.plot([v2, v2], [T2, T3], 'k--', lw=2)
plt.text(v2 + 0.005, (T2 + T3) / 2, "Process 2–3 (v = const)", va='center')

# Extensions of lines for clarity
plt.plot([v1, v1], [T2, T1], 'k:', lw=1)
plt.plot([v2, v2], [T3, 120], 'k:', lw=1)

# Labels
plt.scatter([v1], [T1], color='black')
plt.text(v1 + 0.005, T1, "State 1", fontsize=9)
plt.scatter([v2], [T2], color='black')
plt.text(v2 + 0.005, T2, "State 2", fontsize=9)
plt.scatter([v2], [T3], color='black')
plt.text(v2 + 0.005, T3, "State 3", fontsize=9)

# Formatting
plt.xlabel("Specific Volume (m³/kg)")
plt.ylabel("Temperature (°C)")
plt.title("Idealized T–v Diagram")
plt.grid(True)
plt.xlim(0.05, 0.25)
plt.ylim(100, 420)
plt.legend()
plt.tight_layout()
plt.show()
