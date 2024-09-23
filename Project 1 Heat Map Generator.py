import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from tqdm import tqdm


def create_random_array(dim, q):

    return np.random.choice(np.arange(1.0,  q+1, 1.0), size=(dim, dim))

def create_constant_start_array(dim, state):

    return np.full((dim, dim), state)

def create_cluster_start_array(dim, q):

    array_size = (dim, dim)  # Size of 2D array
    num_picks = 10*dim  # number of random picks
    dist_threshold = 5  # distance of points to center of cluster

    # random array w/ values in 1 to 10
    array = np.random.randint(1, 11, size=array_size)

    # Helper: calculate distance
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # draw 25 random points and set all points in a radius of 5 to the value of the random point
    for _ in range(num_picks):
        # random coordinates of center point
        x, y = np.random.randint(0, array_size[0]), np.random.randint(0, array_size[1])

        # value of center point
        value = array[x, y]

        # change neighbors
        for i in range(array_size[0]):
            for j in range(array_size[1]):
                if distance(x, y, i, j) < dist_threshold:
                    array[i, j] = value
    return array

# calculate the energy of a lattice site for the new spin 
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

# calculate energy of a lattice site for the old spin 
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

# calculate chance in energy for new state
def calc_dE(A, row, col, q):

    return calc_dE_Enew(A, row, col, q) - calc_dE_Eold(A, row, col)

# calculate energy of the state A
def calc_E(A):

    A_up = np.roll(A, 1, axis=0)
    A_down = np.roll(A, -1, axis=0)
    A_left = np.roll(A, 1, axis=1)
    A_right = np.roll(A, -1, axis=1)

    return -np.sum((A_up == A) + (A_down == A) + (A_left == A) + (A_right == A))

#calculate the probability to accept a new state
def prop_to_accept_flip(dE, T):

    return min(1, np.exp((-1) * dE / T))

# return a random new spin
def random_spin(dim, q):

    return int(dim*random.random()), int(dim*random.random()), 1+int(q*random.random())

# check if new state is accepted
def accept_new_state(p):

    return np.random.random() < p

# generate a heatmap plot for the Metropolis algorithm
def gen_plot(A, dim, T, q, iter):
    # create plot window
    plt.ion()  # activate interactive mode
    fig, ax = plt.subplots()

    heatmap = ax.imshow(A, cmap='autumn', vmin=1, vmax=q)  # show first heatmap
    plt.title("2-D Heat Map")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.colorbar(heatmap)

    k_total = 0  # count accepted flips

    for i in (range(iter)):
        row, col, s_new = random_spin(A.shape[0], q)
        dE = calc_dE(A, row, col, s_new)
        p = prop_to_accept_flip(dE, T)

        if accept_new_state(p):
            A[row, col] = s_new
            k_total += 1

        # update after every 10000 steps
        if i % 10000 == 0:
        # print Iteration and accepted flips
            E = calc_E(A)
            avr_E = E/(dim**2)
            plt.title(f"Batch Iteration: {i/10000}, accepted flips: {k_total}, avr energy: {avr_E} ")
            heatmap.set_array(A)  # update array in heatmap
            plt.draw()  # draw new image
            plt.pause(0.01)  # short pause to update image


    plt.ioff()  # deactivate interactive mode
    plt.show()

# calculate Delta N for the heat-bath algorithm
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

# calculate the probability to accept a new state for the heat-bath algorithm
def prop_heat_bath(T, deltaN):

    beta = 1/T
    return np.exp(beta*deltaN/2)/(np.exp(beta*deltaN/2)+np.exp(-beta*deltaN/2))

# check if new state is accepted for the heat-bath algorithm
def accept_flip_heat_bath(p):
    return np.random.random() < p

# generate a heatmap plot for the heat-bath algorithm
def heat_bath_algorithm(A, T, q, iter):

    assert q == 2, "q has to be 2 for heat-bath algorithm"

    plt.ion()  # activate interactive mode
    fig, ax = plt.subplots()

    heatmap = ax.imshow(A, cmap='autumn')  # show first heatmap
    plt.title("Heat-Bath")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.colorbar(heatmap)

    k_total = 0  # count accepted flips

    for i in (range(iter)):
        row, col, s_new = random_spin(A.shape[0], 2)
        dN = calc_dn(A, row, col)
        p = prop_heat_bath(T, dN)

        if accept_new_state(p):
            A[row, col] = 1
            k_total += 1
        else:
            A[row, col] = 2
            k_total += 1

        # update after every 10000 steps
        if i % 10000 == 0:
            E = calc_E(A)
            avr_E = E / (dim ** 2)
            plt.title(f"Batch Iteration: {i / 10000}, accepted flips: {k_total}, avr energy: {avr_E} ")
            heatmap.set_array(A)  # update array in heatmap
            plt.draw()  # draw new image
            plt.pause(0.01)  # short pause to update image

    plt.ioff()  # deactivate interactive mode
    plt.show()

####################################
# Type your system settings in here #
#################################### 

T = 0.02                                # Temperature
q = 10                                  # states
dim = 50                                # dimension of 2d array
iter = 10**7                            # iterations
A = create_random_array(dim, q)         # hot start 
# A = create_cluster_start_array(dim, q)  # clustered start
# A = create_constant_start_array(dim, 1) # cold start

# heat_bath_algorithm(A, T, q, iter)        # generate heatmap plot for Heat-Bath algorithm
gen_plot(A, dim, T, q, iter)                # generate heatmap plot for Metropolis algorithm