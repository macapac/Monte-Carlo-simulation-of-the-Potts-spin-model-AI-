
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_random_array(dim, q):

    return np.random.choice(np.arange(1.0,  q+1, 1.0), size=(dim, dim))

def create_constant_start_array(dim, state):

    return np.full((dim, dim), state)

def create_cluster_start_array(dim, q):

    array_size = (dim, dim)  # Größe des Arrays
    num_picks = 10*dim  # Anzahl der zufälligen Picks
    dist_threshold = 5  # Distanz, innerhalb der Nachbarn geändert werden

    # Erzeuge ein zufälliges 2D-Array mit Werten zwischen 1 und 10
    array = np.random.randint(1, 11, size=array_size)

    # Hilfsfunktion, um den euklidischen Abstand zu berechnen
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # 25 mal zufällig einen Punkt im Array wählen und Nachbarn ändern
    for _ in range(num_picks):
        # Zufällige Koordinaten eines Punktes auswählen
        x, y = np.random.randint(0, array_size[0]), np.random.randint(0, array_size[1])

        # Wert des zufällig gewählten Punktes
        value = array[x, y]

        # Durchlaufe das gesamte Array und ändere die Nachbarpunkte
        for i in range(array_size[0]):
            for j in range(array_size[1]):
                if distance(x, y, i, j) < dist_threshold:
                    array[i, j] = value
    return array


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

    return np.random.randint(0, dim), np.random.randint(0, dim), np.random.randint(1, q+1)

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

    plt.figure()
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

def heat_bath_algorithm(T, iter, dim):

    A = create_random_array(dim, 2)
    iter = 10 ** iter
    k_total = 0  # Zähler für akzeptierte Zustände
    Energy_over_time = np.zeros(int(iter / 10000))
    j = 0

    for i in tqdm(range(iter)):
        row, col, s_new = random_spin(A.shape[0], 2)
        #dE = calc_dE(A, row, col, s_new)
        dN = calc_dn(A, row, col)
        #p = prop_to_accept_flip(dE, T)
        p = prop_heat_bath(T, dN)

        if accept_new_state(p):
            A[row, col] = 1
            k_total += 1
        else:
            A[row, col] = 2
            k_total += 1

        # Aktualisiere alle 10.000 Schritte den Plot
        if i % 10000 == 0:
            E = calc_E(A)
            Energy_over_time[j] = E / (dim ** 2)
            j += 1

    plt.plot(Energy_over_time)
    plt.show()

# heat_bath_algorithm(A, T, q, iter)
# gen_plot(A, dim, T, q, iter)

def Energy_over_Temp_plot_gen():

    iter = 10**7
    dim = 50

    # x_axis_q2 = np.array([0.02, 0.4, 0.8, 1.0, 1.13, 1.25, 1.45, 1.75, 2.0])
    Energy_over_temp_q2 = np.zeros(9)
    x_axis_q10 = np.array([i/10 for i in range(1, 29, 4)])
    Energy_over_temp_q10 = np.zeros(7)
    from tqdm import tqdm

    j = 0
    # for T in (x_axis_q2):
    #     q = 2
    #     A = create_random_array(dim, q)
    #     for i in tqdm(range(iter)):
    #         row, col, s_new = random_spin(A.shape[0], q)
    #         dE = calc_dE(A, row, col, s_new)
    #         p = prop_to_accept_flip(dE, T)
    #
    #         if accept_new_state(p):
    #             A[row, col] = s_new
    #         '''if i % 10000 == 0 and i > 10**5:
    #             E_new = calc_E(A)/(dim**2)
    #             abs = np.abs(E_new - E_old)
    #             eps = 10**(-4)
    #             if abs < eps:
    #                 k += 1
    #                 if k == 5:
    #                     break
    #             else:
    #                 k = 0
    #             E_old = E_new'''
    #
    #     E = calc_E(A)
    #     Energy_over_temp_q2[j] = E / (dim ** 2)
    #     j += 1

    j = 0
    for T in (x_axis_q10):
        q  = 10
        A = create_random_array(dim, q)
        for i in tqdm(range(iter)):
            row, col, s_new = random_spin(A.shape[0], q)
            dE = calc_dE(A, row, col, s_new)
            p = prop_to_accept_flip(dE, T)

            if accept_new_state(p):
                A[row, col] = s_new
            '''if i % 10000 == 0 and i > 10**5:
                    E_new = calc_E(A)/(dim**2)
                    abs = np.abs(E_new - E_old)
                    eps = 10**(-4)
                    if abs < eps:
                        k += 1
                        if k == 5:
                            break
                    else:
                        k = 0
                    E_old = E_new'''
        E = calc_E(A)
        Energy_over_temp_q10[j] = E / (dim ** 2)
        j += 1

    # plt.plot(x_axis_q2, Energy_over_temp_q2)

    discontinuities = np.where(np.diff(Energy_over_temp_q10) >= 0.2)[0]  # Get indices where the gap is too large
    Energy_over_temp_q10_gaps = Energy_over_temp_q10.copy()
    x_axis_q10_gaps = x_axis_q10.copy()

    # Insert np.nan *between* values to break the line, but keep points plotted
    for idx in discontinuities:
        Energy_over_temp_q10_gaps = np.insert(Energy_over_temp_q10, idx + 1, np.nan)
        x_axis_q10_gaps = np.insert(x_axis_q10, idx+1, np.nan)

    plt.plot(x_axis_q10_gaps, Energy_over_temp_q10_gaps, marker='o', linestyle='-', color='black')
    plt.grid()
    plt.title(f"q = {q}, {iter} iterations, discontinuity at critical temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    # plt.ylim(-1.05, -0.45)
    plt.show()

def Energy_over_time_plot_gen(q, T, start, iter, dim):

    iter = 10 ** iter
    from tqdm import tqdm
    Energy_over_time = np.zeros(int(iter/1000))
    j = 0

    if start == 'cold':
        A = create_constant_start_array(dim, np.random.randint(1, q+1))
    elif start == 'cluster':
        A = create_cluster_start_array(dim, q)
    elif start == 'hot':
        A = create_random_array(dim, q)

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
    plt.ylim(-1.05, -0.8)
    plt.show()

def comp_metropolis_heatbath(T, dim, iter, start):

    q = 2
    if start == 'cold':
        A = create_constant_start_array(dim, np.random.randint(1, q+1))
    elif start == 'cluster':
        A = create_cluster_start_array(dim, q)
    elif start == 'hot':
        A = create_random_array(dim, q)

    iter = 10 ** iter
    Energy_over_time_metro = np.zeros(int(iter / 10000))
    j = 0

    for i in tqdm(range(iter)):
        row, col, s_new = random_spin(A.shape[0], q)
        dE = calc_dE(A, row, col, s_new)
        p = prop_to_accept_flip(dE, T)

        if accept_new_state(p):
            A[row, col] = s_new

        if i % 10000 == 0:
            E_avr = calc_E(A)/ (dim ** 2)
            Energy_over_time_metro[j] = E_avr
            j += 1

    Energy_over_time_heatbath = np.zeros(int(iter / 10000))
    j = 0
    q = 2
    if start == 'cold':
        B = create_constant_start_array(dim, np.random.randint(1, q + 1))
    elif start == 'cluster':
        B = create_cluster_start_array(dim, q)
    elif start == 'hot':
        B = create_random_array(dim, q)

    for i in tqdm(range(iter)):
        row, col, s_new = random_spin(B.shape[0], 2)
        # dE = calc_dE(A, row, col, s_new)
        dN = calc_dn(B, row, col)
        # p = prop_to_accept_flip(dE, T)
        p = prop_heat_bath(T, dN)

        if accept_new_state(p):
            B[row, col] = 1
        else:
            B[row, col] = 2

        # Aktualisiere alle 10.000 Schritte den Plot
        if i % 10000 == 0:
            E = calc_E(B)
            Energy_over_time_heatbath[j] = E / (dim ** 2)
            j += 1

    plt.plot(Energy_over_time_heatbath, 'b-', label="Heat-Bath")
    plt.plot(Energy_over_time_metro, 'r-', label="Metropolis")
    plt.legend()
    plt.show()


def comp_E_over_Temp(iter, dim):
    iter = 10 ** iter

    x_axis_q2 = np.array([0.02, 0.4, 0.8, 1.0, 1.13, 1.25, 1.45, 1.75, 2.0])
    Energy_over_temp_metro = np.zeros(9)
    Energy_over_temp_heatbath = np.zeros(9)
    from tqdm import tqdm

    j = 0
    k = 0
    for T in (x_axis_q2):
        q = 2
        A = create_random_array(dim, q)
        B = A
        iter_new = iter * 10
        if T > 1.13:
            iter_new = iter
        for i in tqdm(range(iter_new)):
            row, col, s_new = random_spin(A.shape[0], q)
            dE = calc_dE(A, row, col, s_new)
            p = prop_to_accept_flip(dE, T)

            if accept_new_state(p):
                A[row, col] = s_new

        E = calc_E(A)
        Energy_over_temp_metro[j] = E / (dim ** 2)
        j += 1

        for i in tqdm(range(iter)):
            row, col, s_new = random_spin(B.shape[0], q)
            dN = calc_dn(B, row, col)
            p = prop_heat_bath(T, dN)

            if accept_new_state(p):
                B[row, col] = 1
            else:
                B[row, col] = 2
        E = calc_E(B)
        Energy_over_temp_heatbath[k] = E / (dim ** 2)
        k += 1

    plt.plot(x_axis_q2, Energy_over_temp_heatbath, label='heatbath')
    plt.plot(x_axis_q2, Energy_over_temp_metro, label='metropolis')
    plt.title(f"Heat-bath vs. Metropolis")
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.ylim(-1.05, -0.85)
    plt.legend()
    plt.show()

# def plot_energy_distribution(array, q):
#     flat_array = array.flatten()
#
#     # Plot energy distribution
#     plt.figure(figsize=(8, 6))
#     plt.hist(flat_array, bins=q, edgecolor='black', alpha=0.7)
#     plt.title("Energy Distribution")
#     plt.xlabel("Energy levels")
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.show()


Energy_over_Temp_plot_gen()
# Energy_over_time_plot_gen(2, 0.02, 'hot', 6, 500)
# heat_bath_algorithm(2, 6, 500)
# comp_metropolis_heatbath(2, 500, 6, 'hot')
# comp_E_over_Temp(3, 500)