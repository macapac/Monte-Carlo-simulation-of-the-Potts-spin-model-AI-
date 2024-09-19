import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

T = 5
q = 10
dim = 50
iter = 10**6

def create_random_array(dim, q):

    return np.random.choice(np.arange(1.0,  q+1, 1.0), size=(dim, dim))

A = create_random_array(dim, q) #hot start
#A = np.ones((dim,dim)) #cold start?

# Function to show the heat map
'''plt.imshow(A, cmap='autumn')

# Adding details to the plot
plt.title("2-D Heat Map")
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Adding a color bar to the plot
plt.colorbar()

# Displaying the plot
plt.show()'''

'''def sum_over_neighbors(A, row, col):

    sum = 0.0
    if row == 0:
        sum += A[row + A.shape[0] - 1, col] + A[row + 1, col]
    elif row == A.shape[0] - 1:
        sum += A[row - 1, col] + A[row - A.shape[0] + 1, col]
    else:
        sum += A[row - 1, col] + A[row + 1, col]

    if col == 0:
        sum += A[row, col + A.shape[0] - 1] + A[row, col + 1]
    elif col == A.shape[0] - 1:
        sum += A[row, col - 1] + A[row, col - A.shape[0] + 1]
    else:
        sum += A[row, col - 1] + A[row, col + 1]

    return sum'''

'''def deltaE (A,i,j):

    return 2.0 * A[i,j] * sum_over_neighbors(A, i, j)'''


def calc_dE_Enew(A, row, col, q):

    sum = 0.0
    if row == 0:
        if q == A[row + A.shape[0] - 1, col]:
            sum += 1
        if q == A[row + 1, col]:
            sum += 1
    elif row == A.shape[0] - 1:
        if q== A[row - 1, col]:
            sum += 1
        if q == A[row - A.shape[0] + 1, col]:
            sum += 1
    else:
        if q == A[row - 1, col]:
            sum += 1
        if q == A[row + 1, col]:
            sum += 1

    if col == 0:
        if q == A[row, col + A.shape[0] - 1]:
            sum += 1
        if q == A[row, col + 1]:
            sum += 1
    elif col == A.shape[0] - 1:
        if q == A[row, col - 1]:
            sum += 1
        if q == A[row, col - A.shape[0] + 1]:
            sum += 1
    else:
        if q == A[row, col - 1]:
            sum += 1
        if q == A[row, col + 1]:
            sum += 1

    return -sum


def calc_dE_Eold(A, row, col):
    sum = 0.0
    q = A[row, col]
    if row == 0:
        if q == A[row + A.shape[0] - 1, col]:
            sum += 1
        if q == A[row + 1, col]:
            sum += 1
    elif row == A.shape[0] - 1:
        if q == A[row - 1, col]:
            sum += 1
        if q == A[row - A.shape[0] + 1, col]:
            sum += 1
    else:
        if q == A[row - 1, col]:
            sum += 1
        if q == A[row + 1, col]:
            sum += 1

    if col == 0:
        if q == A[row, col + A.shape[0] - 1]:
            sum += 1
        if q == A[row, col + 1]:
            sum += 1
    elif col == A.shape[0] - 1:
        if q == A[row, col - 1]:
            sum += 1
        if q == A[row, col - A.shape[0] + 1]:
            sum += 1
    else:
        if q == A[row, col - 1]:
            sum += 1
        if q == A[row, col + 1]:
            sum += 1

    return -sum

def calc_dE(A, row, col, q):

    return calc_dE_Enew(A, row, col, q) - calc_dE_Eold(A, row, col)

def calc_E(A):

    A_up = np.roll(A, 1, axis=0)
    A_down = np.roll(A, -1, axis=0)
    A_left = np.roll(A, 1, axis=1)
    A_right = np.roll(A, -1, axis=1)

    return -np.sum((A_up == A) + (A_down == A) + (A_left == A) + (A_right == A))

'''def get_total_energy(self):
    # Nearest neighbor interactions using np.roll
    nn_up = np.roll(self.spin_array, 1, axis=0)
    nn_down = np.roll(self.spin_array, -1, axis=0)
    nn_left = np.roll(self.spin_array, 1, axis=1)
    nn_right = np.roll(self.spin_array, -1, axis=1)

    # Compute energy by checking if spins are the same (delta function)
    energy = -np.sum((self.spin_array == nn_up) +
                     (self.spin_array == nn_down) +
                     (self.spin_array == nn_left) +
                     (self.spin_array == nn_right))

    return energy'''

def prop_to_accept_flip(dE, T):

    return min(1, np.exp((-1) * dE / T))

def random_spin(dim, q):

    return int(dim*random.random()), int(dim*random.random()), 1+int(q*random.random())

def accept_new_state(p):

    return np.random.random() < p

from tqdm import tqdm
'''for i in tqdm(range(1000000)):
    row, col, s_new = random_spin(A.shape[0], q)
    dE = calc_dE(A, row, col, s_new)
    p = prop_to_accept_flip(dE, T)
    k = 0
    if accept_new_state(p):
        A[row, col] = s_new
        k += 1

    if i%10000 == 0:
        print(k)
        print()
        # Function to show the heat map
        plt.imshow(A, cmap='autumn')

        # Adding details to the plot
        plt.title("2-D Heat Map")
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')

        # Adding a color bar to the plot
        plt.colorbar()
        plt.show()'''

'''print(k)
# Function to show the heat map
plt.imshow(A, cmap='autumn')

# Adding details to the plot
plt.title("2-D Heat Map")
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Adding a color bar to the plot
plt.colorbar()

# Displaying the plot
plt.show()'''

def gen_plot(A, dim, T, q, iter):
    # Erstelle das Plot-Fenster
    plt.ion()  # Schalte interaktiven Modus ein
    fig, ax = plt.subplots()

    heatmap = ax.imshow(A, cmap='autumn')  # Erstes Bild anzeigen
    plt.title("2-D Heat Map")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.colorbar(heatmap)

    k_total = 0  # Zähler für akzeptierte Zustände

    for i in (range(iter)):
        row, col, s_new = random_spin(A.shape[0], q)
        dE = calc_dE(A, row, col, s_new)
        p = prop_to_accept_flip(dE, T)

        if accept_new_state(p):
            A[row, col] = s_new
            k_total += 1

        # Aktualisiere alle 10.000 Schritte den Plot
        if i % 10000 == 0:
        # print(f"Iteration: {i}, Akzeptierte Zustände: {k_total}")
            E = calc_E(A)
            avr_E = E/(dim**2)
            plt.title(f"Batch Iteration: {i/10000}, accepted flips: {k_total}, avr energy: {avr_E} ")
            heatmap.set_array(A)  # Aktualisiere das Array in der Heatmap
            plt.draw()  # Zeichne das Bild neu
            plt.pause(0.01)  # Kurze Pause, um das Bild zu aktualisieren

    plt.ioff()  # Schalte den interaktiven Modus wieder aus
    plt.show()

# gen_plot(A, dim, T, q)

def calc_dn(A, row, col):
    B = A
    if row==0:
        B = np.roll(B, 1, axis=0)
        row += 1
    elif row==A.shape[0]-1:
        B = np.roll(B, -1, axis=0)
        row -= 1
    if col==0:
        B = np.roll(B, 1, axis=1)
        col += 1
    elif col==A.shape[1]-1:
        B = np.roll(B, -1, axis=1)
        col -= 1

    return (-2)*(B[row-1, col] + B[row+1, col] + B[row, col-1] + B[row, col+1])+12

def prop_heat_bath(T, deltaN):

    beta = 1/T
    return np.exp(beta*deltaN/2)/(np.exp(beta*deltaN/2)+np.exp(-beta*deltaN/2))

def accept_flip_heat_bath(p):
    return np.random.random() < p

def heat_bath_algorithm(A, T, q, iter):

    assert q == 2, "q has to be 2 for heat-bath algorithm"

    plt.ion()  # Schalte interaktiven Modus ein
    fig, ax = plt.subplots()

    heatmap = ax.imshow(A, cmap='autumn')  # Erstes Bild anzeigen
    plt.title("Heat-Bath")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.colorbar(heatmap)

    k_total = 0  # Zähler für akzeptierte Zustände

    for i in (range(iter)):
        row, col, s_new = random_spin(A.shape[0], q)
        #dE = calc_dE(A, row, col, s_new)
        dN = calc_dn(A, row, col)
        #p = prop_to_accept_flip(dE, T)
        p = prop_heat_bath(T, dN)


        if accept_new_state(p):
            A[row, col] = s_new
            k_total += 1

        # Aktualisiere alle 10.000 Schritte den Plot
        if i % 10000 == 0:
            # print(f"Iteration: {i}, Akzeptierte Zustände: {k_total}")
            E = calc_E(A)
            avr_E = E / (dim ** 2)
            plt.title(f"Batch Iteration: {i / 10000}, accepted flips: {k_total}, avr energy: {avr_E} ")
            heatmap.set_array(A)  # Aktualisiere das Array in der Heatmap
            plt.draw()  # Zeichne das Bild neu
            plt.pause(0.01)  # Kurze Pause, um das Bild zu aktualisieren

    plt.ioff()  # Schalte den interaktiven Modus wieder aus
    plt.show()

# heat_bath_algorithm(A, T, q, iter)
gen_plot(A, dim, T, q, iter)

import h5py

def add_matrix_to_hdf5(file_name, array, metadata, array_name):
    with h5py.File(file_name, 'a') as hf:  # 'a' bedeutet: Anhängen/Öffnen ohne zu überschreiben
        hf.create_dataset(array_name, data=array)  # Speichere das neue Array
        meta_grp = hf.create_group(f'{array_name}_metadata')  # Gruppe für Metadaten erstellen
        for key, value in metadata.items():
            meta_grp.attrs[key] = value  # Speichere Metadaten als Attribute

array_name = 'testarray'
file_name = 'test.h5'
metadata = {"Temp": T, "dim": dim, "iterations": iter, "q": q}

add_matrix_to_hdf5(file_name, A, metadata, array_name)

with h5py.File(file_name, 'r') as hf:
    loaded_array = hf['testarray'][:]  # Zugriff auf 'array2'
    loaded_metadata = dict(hf['testarray_metadata'].attrs)  # Zugriff auf Metadaten von 'array2'

    print("Array 2:")
    print(loaded_array)
    print("Metadata for Array 2:")
    print(loaded_metadata)