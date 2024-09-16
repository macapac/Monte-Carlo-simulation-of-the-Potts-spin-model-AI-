import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_random_array(dim, q):

    return np.random.choice(np.arange(1.0,  q+1, 1.0), size=(dim, dim))

# A = create_random_array(dim, q)

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


def prop_to_accept_flip(dE, T):

    return min(1, np.exp((-1) * dE / T))

def random_spin(dim, q):

    return np.random.randint(0, dim), np.random.randint(0, dim), np.random.randint(1,q+1)

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
    Energy_over_time = np.zeros(int(iter/10000))
    j = 0

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
            Energy_over_time[j] = E/(dim**2)
            j += 1

    plt.ioff()  # Schalte den interaktiven Modus wieder aus
    plt.show()

    fig2 = plt.figure()
    plt.plot(Energy_over_time)
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
# gen_plot(A, dim, T, q, iter)

def Energy_over_Temp_plot_gen():

    q = 10
    iter = 10**7
    dim = 500
    x_axis = np.array([0.02, 0.1, 0.3, 0.5, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75, 0.8, 1.0, 1.5, 2.0, 2.5])
    from tqdm import tqdm
    Energy_over_temp = np.zeros(15)
    j = 0

    for T in (x_axis):

        A = create_random_array(dim, q)
        E_old = calc_E(A)/ (dim**2)
        k = 0
        for i in tqdm(range(iter)):
            row, col, s_new = random_spin(A.shape[0], q)
            dE = calc_dE(A, row, col, s_new)
            p = prop_to_accept_flip(dE, T)

            if accept_new_state(p):
                A[row, col] = s_new
            if i % 10000 == 0 and i > 10**5:
                E_new = calc_E(A)/(dim**2)
                abs = np.abs(E_new - E_old)
                eps = 10**(-4)
                if abs < eps:
                    k += 1
                    if k == 5:
                        break
                else:
                    k = 0
                E_old = E_new

        E = calc_E(A)
        Energy_over_temp[j] = E / (dim ** 2)
        j += 1

    plt.plot(x_axis, Energy_over_temp)
    plt.title(f"q = {q}, {iter} iterations")
    plt.xlabel("Temperatur")
    plt.ylabel("Energy")
    plt.show()

def Energy_over_time_plot_gen(T):

    q = 10
    iter = 10 ** 7
    dim = 50
    from tqdm import tqdm
    Energy_over_time = np.zeros(int(iter/1000))
    j = 0

    A = create_random_array(dim, q)
    E_avr = calc_E(A) / (dim ** 2)
    for i in tqdm(range(iter)):
        row, col, s_new = random_spin(A.shape[0], q)
        dE = calc_dE(A, row, col, s_new)
        p = prop_to_accept_flip(dE, T)

        if accept_new_state(p):
            A[row, col] = s_new

        if i % 1000 == 0:
            E_avr = calc_E(A)/ (dim ** 2)
            Energy_over_time[j] = E_avr
            j += 1

    plt.plot(Energy_over_time)
    plt.title(f"q = 2, T = {T}")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.show()

Energy_over_Temp_plot_gen()
# Energy_over_time_plot_gen(1.5)