import numpy as np
import matplotlib.pyplot as plt

# Erstelle einen gemeinsamen Bereich für die X-Achse
x1 = np.linspace(0, 10, 100)
x2 = np.linspace(0, 2*np.pi, 100)

# Definiere die beiden Funktionen
y1 = np.sin(x1)  # Sinus-Funktion für den ersten Bereich
y2 = np.cos(x2)  # Kosinus-Funktion für den zweiten Bereich

# Erstelle das Plot
plt.plot(x1, y1, 'b-', label="sin(x1)")
plt.plot(x2, y2, 'r-', label="cos(x2)")

# Beschriftungen und Titel
plt.xlabel('X Achse')
plt.ylabel('Y Achse')
plt.title('Sinus und Kosinus auf gemeinsamer X-Achse')

# Legende hinzufügen
plt.legend()

# Anzeigen des Plots
plt.show()
