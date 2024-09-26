import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

def create_random_array(dim, q):
    # Hot start
    return np.random.choice(np.arange(1.0,  q+1, 1.0), size=(dim, dim))

def create_constant_start_array(dim, state):
    # Cold start
    return np.full((dim, dim), state)

def create_cluster_start_array(dim):
    # Note: This is not used in the presentation and report

    array_size = (dim, dim)  # initialise array with dim size
    num_picks = 10*dim  # Number of random picks
    dist_threshold = 5  # Distance to change neigbours

    # create the array, with q = 10
    array = np.random.randint(1, 11, size=array_size)

    # Euclidian distance
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Randomly choose num_picks points and change the neighbours
    for _ in range(num_picks):
        x, y = np.random.randint(0, array_size[0]), np.random.randint(0, array_size[1])

        value = array[x, y]

        # Changing neighbouring points
        for i in range(array_size[0]):
            for j in range(array_size[1]):
                if distance(x, y, i, j) < dist_threshold:
                    array[i, j] = value
    return array

def calc_dE_Enew(A, row, col, q):
    # Function to calculate the change in energy from the new connections to the flipped spin
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
    # Function to calculate the change in energy from the connections to the unflipped spin
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
    # Calculate the change in energy from flipping the spin
    return calc_dE_Enew(A, row, col, q) - calc_dE_Eold(A, row, col)

def calc_E(A):
    # Calculate the whole energy of the lattice by comparing each point with its neighbour
    A_up = np.roll(A, 1, axis=0)
    A_down = np.roll(A, -1, axis=0)
    A_left = np.roll(A, 1, axis=1)
    A_right = np.roll(A, -1, axis=1)

    return -np.sum((A_up == A) + (A_down == A) + (A_left == A) + (A_right == A))


def prop_to_accept_flip(dE, T):
    # whether to accept the flip in the metropolis method
    return min(1, np.exp((-1) * dE / T))

def random_spin(dim, q):
    # return random spins, this was changed to random.random from np.randint to speed up computation significantly
    return int(dim*random.random()), int(dim*random.random()), 1+int(q*random.random())

def accept_new_state(p):
    # Accept the flip or not based on probability
    return np.random.random() < p


def calc_dn(A, row, col):
    # For the heatbath model, calculate dn
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
    # Whether to accept the flip in the heatbatch model
    beta = 1/T
    return np.exp(beta*deltaN/2)/(np.exp(beta*deltaN/2)+np.exp(-beta*deltaN/2))

def accept_flip_heat_bath(p):
    # Accept or reject the flip in the heat bath model based on probability
    return np.random.random() < p

def heat_bath_algorithm(T, iter, dim):
    # plot the energy evolution over iterations in the heat batch algorithm
    A = create_random_array(dim, 2) # Create the array in hot start
    iter = 10 ** iter # To only have to give the order of magnitude
    k_total = 0  # counter for accepted states
    Energy_over_time = np.zeros(int(iter / 10000)) # Result matrix
    j = 0 # counter for the amount of points in our energy over time plot

    for i in tqdm(range(iter)): # For every iteration
        row, col, s_new = random_spin(A.shape[0], 2)
        dN = calc_dn(A, row, col)
        p = prop_heat_bath(T, dN)

        if accept_new_state(p):
            A[row, col] = 1
            k_total += 1

        else:
            A[row, col] = 2
            k_total += 1

        # update every 10000 iterations
        if i % 10000 == 0:
            E = calc_E(A)
            Energy_over_time[j] = E / (dim ** 2)
            j += 1

    plt.plot(Energy_over_time)
    plt.show()


def Energy_over_Temp_plot_gen():
    # Function to get the plot of energy over temperature
    iter = 10**7
    dim = 50

    def calculate_energy_over_temp(x_axis, q, avg_over, dim):
        j = 0
        k = 1
        Energy_over_temp = np.zeros(len(x_axis))

        for T in (x_axis): # for every temperature we want to calulate for
            A = create_random_array(dim, q)
            for i in tqdm(range(iter)): # run the actual algorithm
                row, col, s_new = random_spin(A.shape[0], q)
                dE = calc_dE(A, row, col, s_new)
                p = prop_to_accept_flip(dE, T)

                if accept_new_state(p): # if the spin flip gets accepted
                    A[row, col] = s_new # update with the acceptance

                    if i > iter - avg_over: # And we are over the burn-in period
                        Energy = Energy + dE # Then update the energy

                if i > iter - avg_over: # if we are over the burn-in period, no matter if it gets accepted
                    k = i - avg_over # update k
                    Energy_avr = (Energy_avr) * (k / (k + 1)) + (Energy) * (1 / (k + 1)) # Calculate the new avg energy

                elif i == iter - avg_over: # The exact iteration where we start logging after the burn-in period
                    Energy = calc_E(A)
                    Energy_avr = Energy

            Energy_over_temp[j] = Energy_avr / (dim ** 2) # When it's done running move on to next temp
            j += 1

        return Energy_over_temp

    x_axis_q2 = np.array([i/10 for i in range(1, 29, 4)]) # For q =2
    Energy_over_temp_q2 = calculate_energy_over_temp(x_axis_q2, 2, 100000, dim)

    plt.plot(x_axis_q2, Energy_over_temp_q2, marker='o', linestyle='-', color='black')
    plt.grid()
    plt.title(f"q = 2, {iter:.1E} iterations, No discontinuity at critical temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.show()

    x_axis_q10 = np.array([i/10 for i in range(1, 29, 4)]) # for q = 10
    Energy_over_temp_q10 = calculate_energy_over_temp(x_axis_q10, 10, 1000, dim)

    discontinuities = np.where(np.diff(Energy_over_temp_q10) >= 0.2)[0]  # Get indices where the gap is too large
    Energy_over_temp_q10_gaps = Energy_over_temp_q10.copy()
    x_axis_q10_gaps = x_axis_q10.copy()

    # Insert np.nan between values to break the line, but keep points plotted
    for idx in discontinuities:
        Energy_over_temp_q10_gaps = np.insert(Energy_over_temp_q10, idx + 1, np.nan)
        x_axis_q10_gaps = np.insert(x_axis_q10, idx+1, np.nan)

    plt.plot(x_axis_q10_gaps, Energy_over_temp_q10_gaps, marker='o', linestyle='-', color='black')
    plt.grid()
    plt.title(f"q = 10, {iter:.1E} iterations, discontinuity at critical temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.show()

def Energy_over_time_plot_gen(q, T, start, iter, dim):
    # Generate the plot of energy against time
    iter = 10 ** iter
    Energy_over_time = np.zeros(int(iter/1000))
    j = 0

    # easy way to choose cold, cluster or hot start
    if start == 'cold':
        A = create_constant_start_array(dim, np.random.randint(1, q+1))
    elif start == 'cluster':
        A = create_cluster_start_array(dim, q)
    elif start == 'hot':
        A = create_random_array(dim, q)

    for i in tqdm(range(iter)): # run the simulation
        row, col, s_new = random_spin(A.shape[0], q)
        dE = calc_dE(A, row, col, s_new)
        p = prop_to_accept_flip(dE, T)

        if accept_new_state(p):
            A[row, col] = s_new

        if i % 1000 == 0: # store the energy every 1000 iterations
            E_avr = calc_E(A)/ (dim ** 2)
            Energy_over_time[j] = E_avr
            j += 1

    plt.plot(Energy_over_time)
    plt.title(f"Energy evolution per iteration, q = {q}, T = {T}")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    # plt.ylim(-1.05, -0.8)
    plt.show()

def comp_metropolis_heatbath(T, dim, iter, start):
    # Plot the differences between the metropolis and heatbath algorithms in terms of energy against time
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
        dN = calc_dn(B, row, col)
        p = prop_heat_bath(T, dN)

        if accept_new_state(p):
            B[row, col] = 1
        else:
            B[row, col] = 2

        if i % 10000 == 0:
            E = calc_E(B)
            Energy_over_time_heatbath[j] = E / (dim ** 2)
            j += 1

    # Plot the two methods on the same graph
    plt.plot(Energy_over_time_heatbath, 'b-', label="Heat-Bath")
    plt.plot(Energy_over_time_metro, 'r-', label="Metropolis")
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title("Energy evolution per iteration in Metropolis and Heat-Bath models")
    plt.legend()
    plt.show()


def comp_E_over_Temp(iter, dim):
    # Plot the difference (or lack thereof) between the two methods for the energy against the temperature
    iter = 10 ** iter

    # the points to do this over
    x_axis_q2 = np.array([0.02, 0.4, 0.8, 1.0, 1.13, 1.25, 1.45, 1.75, 2.0])
    Energy_over_temp_metro = np.zeros(9)
    Energy_over_temp_heatbath = np.zeros(9)

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

    # Plot the two on the same graph
    plt.plot(x_axis_q2, Energy_over_temp_heatbath, label='heatbath')
    plt.plot(x_axis_q2, Energy_over_temp_metro, label='metropolis')
    plt.title(f"Heat-bath vs. Metropolis")
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.ylim(-1.05, -0.85)
    plt.legend()
    plt.show()

def plot_energy_distribution(q, dim, T, iterations, batch_size, burn_in):
    # Does one hot start and one cold start, and plots the energy distribution on the same histogram.
    # Useful for comparing the difference between them at the critical temperature
    def calculate_energy_distribution(start):
        if start == 'cold':
            A = create_constant_start_array(dim, np.random.randint(1, q + 1))

        elif start == 'hot':
            A = create_random_array(dim, q)

        energy_averages = []
        for i in tqdm(range(iterations)):

            row, col, s_new = random_spin(A.shape[0], q)
            dE = calc_dE(A, row, col, s_new)
            p = prop_to_accept_flip(dE, T)

            if accept_new_state(p): # If the new state has been accepted
                A[row, col] = s_new

                if i > burn_in: # And we finished the burn in
                    Energy = Energy + dE # update the energy count for this batch

            if i > burn_in: # If we finished the burn in
                current_batch_i += 1 # increment the iteration in the current batch

                # calculate the new energy average
                Energy_avr = (Energy_avr) * (current_batch_i / (current_batch_i + 1)) + (Energy) * (1 / (current_batch_i + 1))

                if current_batch_i == batch_size: # If we got to the end of the current batch
                    # Reset the variables
                    energy_averages.append(Energy_avr/(dim**2))
                    current_batch_i = 0
                    Energy = calc_E(A)
                    Energy_avr = Energy

            elif i == burn_in: # On the iteration where the burn-in ends
                Energy = calc_E(A)
                Energy_avr = Energy
                current_batch_i = 0

        return energy_averages

    # hot start
    hot_averages = calculate_energy_distribution('hot')

    # cold start
    cold_averages = calculate_energy_distribution('cold')

    # Plot energy distribution
    plt.figure(figsize=(8, 6))
    red = matplotlib.colors.to_rgba('red', alpha=1)
    blue = matplotlib.colors.to_rgba('blue', alpha=1)
    colours = [blue, red]

    plt.hist([cold_averages, hot_averages], bins=20, label=['Cold start', 'Hot start'], color=colours)
    plt.title(f"Energy distribution for cold and hot start, q = {q}, T = Tc = 0.7012")
    plt.xlabel("Energy Level")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()



Energy_over_Temp_plot_gen()
Energy_over_time_plot_gen(2, 0.02, 'hot', 7, 500)
Energy_over_time_plot_gen(2,5,'cold', 7, 500)
Energy_over_time_plot_gen(10, 0.02,'hot', 7, 500)
Energy_over_time_plot_gen(10,5,'cold', 7, 500)
heat_bath_algorithm(2, 6, 500)
comp_metropolis_heatbath(2, 500, 6, 'hot')
comp_E_over_Temp(6, 500)
plot_energy_distribution(2,50,0.7012,10**8,10**5,10**6)