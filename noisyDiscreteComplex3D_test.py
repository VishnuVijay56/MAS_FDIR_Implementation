"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - Complex configuration from Shiraz's TAC paper with noise and discrete configuration changes
            - Implementation of " Collaborative Fault-Identification &
              Reconstruction in Multi-Agent Systems" by Khan et al.
            - Algorithm uses inter-agent distances to reconstruct a sparse
              vector of agents where the nonzero elements are exactly the faulty
              agents, with the elements being the attack vectors. The algorithm
              does not assume any anchors exist so entire network can be evaluated
              for faults.
            - Uses SCP to convexify the nonconvex problem.
            - Uses ADMM to split the convex problem into smaller problems that can
              be solved parallelly by each agent.
"""

###     Imports             - Public Libraries
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from time import time
from math import comb

###     Imports             - User-Defined Files
from generic_agent import GenericAgent as Agent
from iam_models import distance



###     Initializations     - Scalars
dim             =   3   # 2 or 3
num_agents      =   20
num_faulty      =   6   # must be << num_agents for sparse error assumption
n_scp           =   12  # Number of SCP iterations
n_admm          =   10  # Number of ADMM iterations
n_iter          =   n_admm * n_scp
show_prob1      =   False
show_prob2      =   False
use_threshold   =   False
rho             =   0.75
iam_noise       =   0.02
pos_noise       =   0.02

# Thresholds
warm_start      =   True
dual_var_check  =   True
R_diff_check    =   False
lam_thresh      =   1.5
mu_thresh       =   5e-3
R_diff_thresh   =   1

# Show
show_plots = False


###     Assertions


###     Initializations     - Agents
# 20 agents making up a complex 3d configuration
agents      =   [None] * num_agents
agents[0]   =   Agent(agent_id = 0, init_position = np.array([[0.1, 2.4, 5.4]]).T)
agents[1]   =   Agent(agent_id = 1, init_position = np.array([[2.8, 5.4, 6.1]]).T)
agents[2]   =   Agent(agent_id = 2, init_position = np.array([[2.15, 4.8, 4.3]]).T)
agents[3]   =   Agent(agent_id = 3, init_position = np.array([[1.15, 0.4, 3.9]]).T)
agents[4]   =   Agent(agent_id = 4, init_position = np.array([[3.0, 3.85, 5.4]]).T)
agents[5]   =   Agent(agent_id = 5, init_position = np.array([[3.4, 4.25, 2.0]]).T)
agents[6]   =   Agent(agent_id = 6, init_position = np.array([[3.45, 1.8, 2.2]]).T)
agents[7]   =   Agent(agent_id = 7, init_position = np.array([[5.2, 5.0, 5.25]]).T)
agents[8]   =   Agent(agent_id = 8, init_position = np.array([[5.3, 3.8, 0.1]]).T)
agents[9]   =   Agent(agent_id = 9, init_position = np.array([[5.2, 0.8, 3.15]]).T)
agents[10]  =   Agent(agent_id = 10, init_position = np.array([[6.2, 3.3, 5.6]]).T)
agents[11]  =   Agent(agent_id = 11, init_position = np.array([[5.05, 3.8, 3.6]]).T)
agents[12]  =   Agent(agent_id = 12, init_position = np.array([[4.15, 5.65, 3.4]]).T)
agents[13]  =   Agent(agent_id = 13, init_position = np.array([[0.15, 3.4, 2.45]]).T)
agents[14]  =   Agent(agent_id = 14, init_position = np.array([[1.85, 5.15, 0.65]]).T)
agents[15]  =   Agent(agent_id = 15, init_position = np.array([[2.4, 2.4, 1.6]]).T)
agents[16]  =   Agent(agent_id = 16, init_position = np.array([[1.4, 5.4, 2.4]]).T)
agents[17]  =   Agent(agent_id = 17, init_position = np.array([[3.2, 3.4, 0.2]]).T)
agents[18]  =   Agent(agent_id = 18, init_position = np.array([[5.4, 5.4, 1.4]]).T)
agents[19]  =   Agent(agent_id = 19, init_position = np.array([[4.7, 2.4, 5.4]]).T)

# Add error vector
# NOTE: may not work for some random cases
# random case -> np.random.randint(low=0, high=num_agents, size=4)
# tac paper case -> [0, 5, 7, 9, 10, 13]
faulty_id1 = [0, 5, 7]
faulty_id2 = [9, 10, 13]
faulty_id = faulty_id1 + faulty_id2

fault_vec1  =   [np.array([[0.275, 0.447, 0.130]]).T,
                 np.array([[-0.849, 0.170, 0.888]]).T,
                 np.array([[0.761, -0.408, 0.438]]).T]
fault_vec2  =   [np.array([[-0.640, 0.260, -0.941]]).T,
                 np.array([[0.879, 0.425, -0.710]]).T,
                 np.array([[-0.534, -0.543, -0.588]]).T]
fault_vec   =   fault_vec1 + fault_vec2

for index, agent_id in enumerate(faulty_id):
    agents[agent_id].error_vector = fault_vec[index][:,np.newaxis].reshape((dim, -1))

x_true = []
for id, agent in enumerate(agents):
    x_true.append(0.0*agent.error_vector)
    # x_true.append(agent.error_vector)


# Set Neighbors
edges       =  [[0,2], [0,3], [0,4], [0,16], 
                [1,2], [1,4], [1,7], [1,11],
                [2,4], [2,5], [2,7], [3,4],
                [4,5], [4,6], [4,7], [4,10],
                [5,6], [5,8], [6,7], [6,9],
                [7,10], [8,9], [8,11], [9,11],
                [9,10], [10,11], [12,5], [12,7],
                [12,11], [12,2], [13,14], [13,15],
                [14,15], [3,15], [5,15], [13,0],
                [14,5], [6,14], [19,10], [19,4],
                [19,9], [18,8], [18,17], [18,11],
                [18,12], [17,14], [17,15], [17,8],
                [17,18], [16,14], [16,2], [16,13],
                [18,5], [15,6], [16,3], [0,19],
                [7,19], [17,5]] 
edges_flip  =   deepcopy(edges)
for idx, dir_edge in enumerate(edges_flip):
    dir_edge.reverse()

edges       =   edges+edges_flip            # these edges are directed

for agent_id, agent in enumerate(agents):
    # Neighbor List
    nbr_list        =   []
    edge_list       =   []
    
    for edge_ind, edge in enumerate(edges):
        if (agent_id) == edge[0]:
            nbr_list.append(edge[1])
            edge_list.append(edge_ind)
    
    agent.set_neighbors(nbr_list)
    agent.set_edge_indices(edge_list)



###     Useful Functions
# Measurement function Phi
def measurements(p, x_hat):
    measurements = []

    for edge in edges:
        dist = distance((p[edge[0]] + x_hat[edge[0]]), (p[edge[1]] + x_hat[edge[1]]))
        measurements.append(dist)

    return measurements

# Finds row of R
def get_Jacobian_row(edge_ind, p, x):
    edge = edges[edge_ind]
    agent1_id = edge[0]
    agent2_id = edge[1]
    pos1 = p[edge[1]] + x[edge[1]]
    pos2 = p[edge[0]] + x[edge[0]]
    disp    = (pos1 - pos2)
    R_k = np.zeros((1, dim*num_agents))

    dist = distance(pos1, pos2)
    R_k[:, dim*agent2_id:dim*(agent2_id + 1)] = disp.T  / dist
    R_k[:, dim*agent1_id:dim*(agent1_id + 1)] = -disp.T / dist

    return R_k

# Computes whole R matrix
def get_Jacobian_matrix(p, x):
    R = []

    for edge_ind, edge in enumerate(edges):
        R.append(get_Jacobian_row(edge_ind, p, x))
    
    return R

# Returns rank of Jacobian matrix
def get_Jacobian_rank(R):
    
    jac_matrix = R[0]
    for row in R[1:]:
        jac_matrix = np.vstack((jac_matrix, row))
    
    return np.linalg.matrix_rank(jac_matrix)

# Returns the matrix norm of the difference between old and new R matrices
def get_Jacobian_matrix_norm_diff(R_old, R_new):
    
    # Find R_norm_diff for each agent
    R_norm_diff_arr = [0] * num_agents

    for id, _ in enumerate(agents):
        # Get Agent Edge List
        this_edge_list = agents[id].get_edge_indices()
        
        # Construct 2D Array of Old Matrix
        old_matrix = R_old[this_edge_list[0]]
        for edge_ind in this_edge_list[1:]:
            old_matrix = np.vstack((old_matrix, R_old[edge_ind]))
        
        # Construct 2D Array of New Matrix
        new_matrix = R_new[this_edge_list[0]]
        for edge_ind in this_edge_list[1:]:
            new_matrix = np.vstack((new_matrix, R_new[edge_ind]))

        R_norm_diff_arr[id] = np.linalg.norm(new_matrix - old_matrix)

    return R_norm_diff_arr



###     Initializations     - Measurements and Positions

# Error
x_star = [np.zeros((dim, 1)) for i in range(num_agents)]                    # Equivalent to last element in x_history (below)
x_history = [np.zeros((dim, (n_iter))) for i in range(num_agents)]          # Value of x at each iteration of algorithm
x_norm_history = [np.zeros((1, (n_iter))) for i in range(num_agents)]       # Norm of difference between x_history and x_true

# Position
p_est = [agents[i].get_estimated_pos() for i in range(num_agents)]          # Will be updated as algorithm loops and err vector is reconstructed
p_hat = deepcopy(p_est)                                                     # Reported positions of agents
p_true = [agents[i].get_true_pos() for i in range(num_agents)]              # CONSTANT: True pos
p_hat_history = [np.zeros((dim, n_iter)) for i in range(num_agents)]        # Value of p_hat at each iteration of algorithm

# Interagent measurement
y = measurements(p_true, x_star)                                            # CONSTANT: Phi(p_hat + x_hat), true interagent measurement

# Residuals
residuals = [np.zeros(n_iter) for i in range(num_agents)]                   # Running residuals of each agent (residual <= 1 is nominal)



###      Initializations    - Optimization Parameters

reset_lam = [False] * num_agents
reset_mu = [False] * num_agents
total_iterations = np.arange((n_iter))
for agent_id, agent in enumerate(agents):
    num_edges       = len(agent.get_edge_indices())
    num_neighbors   = len(agent.get_neighbors())

    # CVX variables
    agent.init_x_cp(cp.Variable((dim, 1)))
    agent.init_w_cp(cp.Variable((dim, 1)), agent.get_neighbors())

    # Parameters
    agent.init_x_bar(np.zeros((dim, 1)))
    agent.init_lam(np.zeros((1, 1)), agent.get_edge_indices())
    agent.init_mu(np.zeros((dim, 1)), agent.get_neighbors())
    agent.init_x_star(np.zeros((dim, 1)), agent.get_neighbors()) # own err is last elem
    agent.init_w(np.zeros((dim, 1)), agent.get_neighbors())



###     Initializations     - List Parameters
print("\n~ ~ ~ ~ PARAMETERS ~ ~ ~ ~")
print("rho:", rho)
print("Number of agents:", num_agents)
print("Faulty Agent ID and Vector:")
for i, id in enumerate(faulty_id):
    print(f"\tID: {id}\t\t Vector: {fault_vec[i].flatten()}")



### Store stuff
lam_norm_history = [np.zeros((len(agents[i].get_edge_indices()), n_iter)) for i in range(num_agents)]
mu_norm_history = [np.zeros((len(agents[i].get_neighbors()), n_iter)) for i in range(num_agents)]
R_norm_diff_history = [np.zeros(n_scp) for i in range(num_agents)]
R_norm_diff = [0] * num_agents
sum_err_rmse = 0.0
start_time = time()
solver_err = False
maximal_rank = dim*num_agents - comb(dim+1, 2)



###     Looping             - SCP Outer Loop
print("\nStarting Loop")
for outer_i in tqdm(range(n_scp), desc="SCP Loop", leave=False):

    # Set p_hat depending on iteration number
    for id, agent in enumerate(agents):
        if (outer_i >= round(n_scp*2/3)) and (id in faulty_id2):
            agent.faulty = True
            x_true[id] = fault_vec[faulty_id.index(id)]
            # print(f"\tAgent {id} - Case 1")
        elif ( (outer_i >= round(n_scp*1/3)) and (outer_i < round(n_scp*2/3)) ) and (id in faulty_id1):
            agent.faulty = True
            x_true[id] = fault_vec[faulty_id.index(id)]
            # print(f"\tAgent {id} - Case 2")
        else:
            agent.faulty = False
            x_true[id] *= 0.
        
        p_hat[id] = agent.get_estimated_pos()
        

    # Noise in Position Estimate
    p_hat_noise = deepcopy(p_hat)
    for i, _ in enumerate(p_hat_noise):
        p_hat_noise[i] = p_hat[i] + np.random.uniform(low=-pos_noise, high=pos_noise, size=(dim, 1))

    # Difference between expected and true measurements
    exp_meas    =   measurements(p_hat_noise, x_star)
    z           =   [(y[i] - meas) for i, meas in enumerate(exp_meas)]

    # Linearized Measurement Model
    R_new = get_Jacobian_matrix(p_hat_noise, x_star)
    if (outer_i > 0):
        R_norm_diff = get_Jacobian_matrix_norm_diff(R, R_new)
    R = R_new

    # Check norm difference between new and old R
    if R_diff_check and (not warm_start):
        for agent_id, agent in enumerate(agents):
            if (R_norm_diff[agent_id] >= R_diff_thresh):
                agent.init_lam(np.zeros((1, 1)), agent.get_edge_indices())
                agent.init_mu(np.zeros((dim, 1)), agent.get_neighbors())

    # Check if rank of R is maximal
    jac_rank = get_Jacobian_rank(R)
    if(jac_rank != maximal_rank):
        print(f"\nMaximal Condition not met @ SCP Iteration {outer_i}")
        print(f"\tExpected: {maximal_rank}; Got: {jac_rank}")

    # Reset extra primal variables for inner loop
    for agent in agents:
        agent.init_w(np.zeros((dim, 1)), agent.get_neighbors())



    ###     Looping             - ADMM Inner Loop
    for inner_i in tqdm(range(n_admm), desc="ADMM Loop", leave=False):

        ##      Noise               - Add noise to interagent measurements (and therefore z)
        z_noise = [(z[i] + np.random.uniform(low=-iam_noise, high=iam_noise)) for i, _ in enumerate(z)]


        ##      Minimization        - Primal Variable 1
        for id, agent in enumerate(agents):
            # Thresholding: Summation over edges
            term1 = 0
            for i, edge_ind in enumerate(agent.get_edge_indices()):
                R_k = R[edge_ind]
                constr_c = R_k[:, dim*id:dim*(id+1)] @ (-agent.x_star[id]) - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R_k[:, dim*nbr_id:dim*(nbr_id+1)] @ agent.w[nbr_id]
                
                term1 += R_k[:, dim*id:dim*(id+1)].T @ (constr_c + (agent.lam[edge_ind] / rho))

            # Thresholding: Summation over neighbors
            term2 = 0
            for nbr_id in agent.get_neighbors():
                constr_d = -agent.x_star[id] - agent.w[nbr_id]
                term2 += constr_d + (agent.mu[nbr_id] / rho)

            # Tresholding: Check threshold
            res = np.linalg.norm(term1 + term2)
            residuals[id][inner_i + outer_i*n_admm] = res
            if use_threshold and ((res*rho) <= 1):
                agent.x_bar = deepcopy(-agent.x_star[id])
            else:
            # Optimization: Find x_bar if over threshold
                objective = cp.norm(agent.x_star[id] + agent.x_cp)
                
                # Summation for c() constraint
                for _, edge_ind in enumerate(agent.get_edge_indices()): 
                    constr_c = R[edge_ind][:, dim*id:dim*(id+1)] @ agent.x_cp - z[edge_ind]
                    for nbr_id in agent.get_neighbors():
                        constr_c += R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w[id]
                    
                    objective += ((rho/2)*cp.power(cp.norm(constr_c), 2)
                                    + agent.lam[edge_ind].T @ (constr_c))
                
                # Summation for d() constraint
                for _, nbr_id in enumerate(agent.get_neighbors()): 
                    constr_d = agent.x_cp - agent.w[nbr_id]
                    objective += ((rho/2)*cp.power(cp.norm(constr_d), 2)
                                + agent.mu[nbr_id].T @ (constr_d))
                    
                prob1 = cp.Problem(cp.Minimize(objective), [])
                try:
                    prob1.solve(verbose=show_prob1)
                except cp.SolverError:
                    solver_err = True
                    break
                
                if prob1.status != cp.OPTIMAL:
                    print("\nERROR Problem 1: Optimization problem not solved @ (%d, %d, %d)" % (inner_i, outer_i, id))
                
                agent.x_bar = deepcopy(np.array(agent.x_cp.value).reshape((-1, 1)))
            
            # Store: Reconstructed Error
            new_x = deepcopy(agent.x_bar.flatten()) + deepcopy(agent.x_star[id].flatten())
            x_history[id][:, inner_i + outer_i*n_admm] = new_x.flatten()

            # Store: Convergence of Reconstructed Error Vector
            new_x_norm = np.linalg.norm(new_x.flatten() + x_true[id].flatten())
            x_norm_history[id][:, inner_i + outer_i*n_admm] = new_x_norm
            sum_err_rmse += new_x_norm


        ##      Minimization        - Primal Variable 2
        for agent_id, agent in enumerate(agents):
            objective = cp.norm(agent.x_star[agent_id] + agent.x_bar)

            # Summation for c() constraint
            for edge_ind in agent.get_edge_indices(): 
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_bar - z_noise[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c = constr_c + R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w_cp[agent_id]
                
                objective += ((rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[edge_ind].T @ (constr_c))
            
            # Summation for d() constraint
            for nbr_id in agent.get_neighbors():
                constr_d = agent.x_bar - agent.w_cp[nbr_id]
                objective += ((rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob2 = cp.Problem(cp.Minimize(objective), [])
            try:
                prob2.solve(verbose=show_prob2)
            except cp.SolverError:
                solver_err = True
                break

            if prob2.status != cp.OPTIMAL:
                print("\nERROR Problem 2: Optimization problem not solved @ (%d, %d, %d)" % (inner_i, outer_i, agent_id))

            for _, nbr_id in enumerate(agent.get_neighbors()):
                agent.w[nbr_id] = deepcopy(np.array(agent.w_cp[nbr_id].value).reshape((-1, 1)))


        ##      Multipliers         - Update Lagrangian Multipliers of Minimization Problem
        for agent_id, agent in enumerate(agents):
            
            # Summation for c() constraint
            for i, edge_ind in enumerate(agent.get_edge_indices()):
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_bar - z_noise[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w[agent_id]
                
                new_lam = (agent.lam[edge_ind] + rho * constr_c)
                agent.lam[edge_ind] = deepcopy(new_lam)
                lam_norm_history[agent_id][i, (inner_i + outer_i*n_admm)] = np.linalg.norm(deepcopy(new_lam))

                if (not warm_start) and (np.linalg.norm(new_lam) > lam_thresh):
                    # print(f"RESET LAM: Agent {agent_id} at SCP {outer_i}")
                    reset_lam[agent_id] = True

            # Summation for d() constraint
            for i, nbr_id in enumerate(agent.get_neighbors()):
                constr_d = agent.x_bar - agent.w[nbr_id]

                new_mu = (agent.mu[nbr_id] + rho * constr_d)
                agent.mu[nbr_id] = deepcopy(new_mu)
                mu_norm_history[agent_id][i, (inner_i + outer_i*n_admm)] = np.linalg.norm(deepcopy(new_mu))

                if (not warm_start) and (np.linalg.norm(new_mu) > mu_thresh):
                    # print(f"RESET MU: Agent {agent_id} at SCP {outer_i}")
                    reset_mu[agent_id] = True
                
        

        ##      Store           - Position and Error Vectors
        for id, agent in enumerate(agents):
            # True Position
            p_hat_history[id][:, inner_i + outer_i*n_admm] = p_hat[id].flatten()

        ##      Check for Solver Error
        if solver_err:
            break


    ###     END Looping         - ADMM Inner Loop

    ##          Check for Solver Error
    if solver_err:
        break
    
    # Update Error Vectors after ADMM subroutine
    for agent_id, agent in enumerate(agents): 
        for list_ind, nbr_id in enumerate(agent.get_neighbors()):
            agent.x_star[nbr_id] = agent.x_star[nbr_id] + agents[nbr_id].x_bar
        
        agent.x_star[agent_id] = agent.x_star[agent_id] + agent.x_bar
        x_star[agent_id] = agent.x_star[agent_id]
        
        # Update position and x_dev
        p_est[agent_id] = p_hat_noise[agent_id] + x_star[agent_id]

        # Store Norm Difference in R for each agent
        R_norm_diff_history[agent_id][outer_i] = R_norm_diff[agent_id]
    
    # Check if a reset flag was set
    if dual_var_check and (not warm_start):    
        for agent_id, agent in enumerate(agents):
            if (reset_lam[agent_id] or reset_mu[agent_id]):
                # print(f"RESET DUAL: Agent {agent_id} at SCP {outer_i}")
                agent.init_lam(np.zeros((1, 1)), agent.get_edge_indices())
                agent.init_mu(np.zeros((dim, 1)), agent.get_neighbors())
                reset_mu[agent_id] = False
                reset_lam[agent_id] = False

###     END Looping         - SCP Outer Loop

final_time = time()
print(f"==================================================================================")
print(f"IAM Noise: {iam_noise}")
print(f"Position Noise: {pos_noise}")
print(f"Penalty Parameter: {rho}")
print(f"Warm Start: {warm_start}")
print(f"----------------------------------------------------------------------------------")
print(f"Solver Error: {solver_err}")
print(f"Average RMSE: {sum_err_rmse / num_agents} m")
print(f"Elapsed Time: {final_time - start_time} seconds")
print(f"Average Time per Iteration: {(final_time - start_time)/n_iter} seconds")
print(f"Average Time per Iteration per Agent: {(final_time - start_time)/(n_iter*num_agents)} seconds")
print(f"==================================================================================")



###     Plotting            - Static Position Estimates
print("\nPlotting")
print()
plt.rcParams.update({'text.usetex': True,
                        'font.family': 'Helvetica'})

# Create position estimate over time data
p_est_hist = []
for id in range(num_agents):
    p_id = np.zeros((dim, n_iter))
    for iter in range(n_iter):
        p_id[:,iter] = p_hat_history[id][:, iter].flatten() + x_history[id][:, iter]
    p_est_hist.append(p_id)

# # Compare position estimates before and after reconstruction
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(projection='3d')
# ax1.set_title("Agent Position Estimates")
# ax1.set_xlabel("x")
# ax1.set_ylabel("y")
# ax1.set_zlabel("z")

# for agent_id, agent in enumerate(agents): # Draw points
#     ax1.scatter(p_est_hist[agent_id][0, -1], p_est_hist[agent_id][1, -1], p_est_hist[agent_id][2, -1], marker='*', c='m', label="After", s=100)
#     # ax1.scatter(p_est[agent_id][0], p_est[agent_id][1], p_est[agent_id][2], marker='*', c='m', label="After", s=100)
#     ax1.scatter(p_hat[agent_id][0], p_hat[agent_id][1], p_hat[agent_id][2], facecolors='none', edgecolors='orangered', label="Before", s=100)
#     ax1.scatter(p_true[agent_id][0], p_true[agent_id][1], p_true[agent_id][2], marker='x', c='g', label="True", s=100)
#     #TODO: Fix labels on plot
#     # ax1.text(p_true[agent_id][0], p_true[agent_id][1], p_true[agent_id][2], '%s' % (str(agent_id)))

# for i, edge in enumerate(edges): # Draw edges
#     p1 = p_est[edge[0]]
#     p2 = p_est[edge[1]]
#     x = [p1[0], p2[0]]
#     y = [p1[1], p2[1]]
#     z = [p1[2], p2[2]]
#     ax1.plot(x, y, z, c='k', linewidth=1, alpha=0.5)[0]
# plt.legend(["With Inter-agent Measurements", "Without Inter-agent Measurements", "True Position"], loc='best', fontsize=6, markerscale=0.4)
# plt.grid(True)



###     Plotting            - Error Convergence
# Show convergence of estimated error vector to true error vector over time
x_norm_history = [x_norm_history[i].flatten() for i in range(num_agents)]
fig_err = plt.figure(dpi=500, figsize=(9,3))
ax_err = fig_err.add_subplot()
lines = [None] * num_agents
for agent_id, agent in enumerate(agents):
    label_str = f"Agent {agent_id}"
    plt_color = 'slategray'
    if agent_id in faulty_id:
        plt_color = 'orangered'
    lines[agent_id] = ax_err.plot(total_iterations, x_norm_history[agent_id], c=plt_color, label=label_str)[0]

# plt.title(r'Error Vector Convergence ( $\rho = {}$ )'.format(rho))
plt.xlabel(r'{ADMM Iterations}')
plt.ylabel(r'$ \| \mathbf{x}[i] - ( \mathbf{x}^* [i] + \hat{\mathbf{x}}[i]) \| $')
plt.ylim((0, 1.25))
plt.xlim((0, (n_iter - 1)))
plt.xticks(ticks=np.arange(0, n_iter, n_admm))
plt.yticks(ticks=np.arange(0, 1.25, 0.25))
plt.legend([lines[1], lines[faulty_id[0]]], [r'$i \in$ Nominal Agents', r'$i \in$ Faulty Agents'])
plt.grid(True)

dt_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
fname_err = "fig/3D-NoisyDiscreteComplex/err_conv_" + dt_string + ".svg"
# plt.savefig(fname_err, dpi=500)


###     Plotting            - Animation

# Start figure
fig2 = plt.figure(dpi=500, figsize=(16,9))
ax2_1 = fig2.add_subplot(1, 2, 1, projection='3d')
ax2_2 = fig2.add_subplot(1, 2, 2)
ax2_1.set_title(r"$\textnormal{Swarm States}$", fontsize=18)
ax2_1.set_xlabel(r"$x\textnormal{-position}$", fontsize=16)
ax2_1.set_ylabel(r"$y\textnormal{-position}$", fontsize=16)
ax2_1.set_zlabel(r"$z\textnormal{-position}$", fontsize=16)
scat_pos_est = [None] * num_agents # Position estimate during reconstruction
scat_pos_hat = [None] * num_agents # Initial position estimate
scat_pos_true = [None] * num_agents # True positions
line_pos_est = [None] * len(edges) # Inter-agent communication
ax2_1.set_xlim((0, 7))
ax2_1.set_ylim((0, 7))
ax2_1.set_zlim((0, 7))

# Draw each agent's original estimated, current estimated, and true positions
for agent_id, _ in enumerate(agents):
    scat_pos_hat[agent_id] = ax2_1.plot(p_hat_history[agent_id][0, 0], p_hat_history[agent_id][1, 0], p_hat_history[agent_id][2, 0], 
                                        marker='o', markerfacecolor='none', c='orangered', linestyle='None', label="Before", markersize=15)[0]
    scat_pos_true[agent_id] = ax2_1.plot(p_true[agent_id][0], p_true[agent_id][1], p_true[agent_id][2], 
                                        marker='x', c='yellowgreen', linestyle='None', label="True", markersize=8)[0]
    scat_pos_est[agent_id] = ax2_1.plot(p_est_hist[agent_id][0, 0], p_est_hist[agent_id][1, 0], p_est_hist[agent_id][2, 0], 
                                        marker='*', c='c', linestyle='None', label="After", markersize=10)[0]
    # ax2.text(p_hat[agent_id][0], p_hat[agent_id][1], p_hat[agent_id][2],
    #         "%s" % (agent_id), size=10, zorder=1, color='k')

# Draw line for each edge of network
for i, edge in enumerate(edges):
    p1 = p_est_hist[edge[0]][:, 0]
    p2 = p_est_hist[edge[1]][:, 0]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    line_pos_est[i] = ax2_1.plot(x, y, z, c='k', linewidth=1, alpha=0.25)[0]
ax2_1.set_aspect('equal')
ax2_1.legend([r"$\textnormal{Estimated State}$", r"$\textnormal{True State}$", r"$\textnormal{Reconstructed State}$"],
           fancybox=True, loc='upper left', bbox_to_anchor=(-0.35, 0.6), ncols=1, fontsize=16)
ax2_1.grid(True)

# Error Convergence Plot Animation
ax2_2.set_title(r"$\textnormal{Error Convergence}$", fontsize=18)
err_lines = [None] * num_agents
for agent_id, agent in enumerate(agents):
    plt_color = 'slategray'
    if agent_id in faulty_id:
        plt_color = 'orangered'
    err_lines[agent_id] = ax2_2.plot(total_iterations[0], x_norm_history[agent_id][0], c=plt_color)[0]

ax2_2.set_xlabel(r'\textnormal{ADMM Iterations}', fontsize=16)
ax2_2.set_ylabel(r'$ \| \mathbf{x}[i] - ( \mathbf{x}^* [i] + \hat{\mathbf{x}}[i]) \|_2 $', fontsize=16)
ax2_2.set_ylim((0, 1.25))
ax2_2.set_xlim((0, (n_iter - 1)))
ax2_2.set_xticks(ticks=np.arange(0, (n_iter-1), n_admm))
ax2_2.set_yticks(ticks=np.arange(0, 1.25, 0.25))
ax2_2.legend([lines[1], lines[faulty_id[0]]], [r'$i \in \textnormal{Nominal Agents}$', r'$i \in \textnormal{Faulty Agents}$'], fontsize=14)
ax2_2.grid(True)

# Update function
def update_pos_plot(frame):
    updated_ax = []
    # Draw each agent's original estimated, current estimated, and true positions
    for agent_id, _ in enumerate(agents):
        # State Reconstruction
        new_pos = (float(p_est_hist[agent_id][0, frame]), float(p_est_hist[agent_id][1, frame]), float(p_est_hist[agent_id][2, frame]))
        scat_pos_est[agent_id].set_data([new_pos[0]], [new_pos[1]])
        scat_pos_est[agent_id].set_3d_properties([new_pos[2]])
        updated_ax.append(scat_pos_est[agent_id])
        
        # State Estimation
        new_p_hat = (float(p_hat_history[agent_id][0, frame]), float(p_hat_history[agent_id][1, frame]), float(p_hat_history[agent_id][2, frame]))
        scat_pos_hat[agent_id].set_data([new_p_hat[0]], [new_p_hat[1]])
        scat_pos_hat[agent_id].set_3d_properties([new_p_hat[2]])
        updated_ax.append(scat_pos_hat[agent_id])
        
        # Error
        err_lines[agent_id].set_data(total_iterations[0:frame], x_norm_history[agent_id][0:frame])
        updated_ax.append(err_lines[agent_id])
        
    
    # Draw line for each edge of network
    for i, edge in enumerate(edges):
        p1 = p_est_hist[edge[0]][:, frame]
        p2 = p_est_hist[edge[1]][:, frame]
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        z = [p1[2], p2[2]]

        line_pos_est[i].set_data(x, y)
        line_pos_est[i].set_3d_properties(z)

        updated_ax.append(line_pos_est[i])
    
    return updated_ax
    
# Call update function
pos_ani = animation.FuncAnimation(fig=fig2, func=update_pos_plot, frames=n_iter, interval=100, blit=False, repeat=True)
fname_ani = "fig/3D-NoisyDiscreteComplex/combined_ani-" + dt_string + ".mp4"
pos_ani.save(filename=fname_ani)#, writer="pillow")


###     Plotting            - Residuals and Threshold

# # Start figure
# fig2, ax2 = plt.subplots()
# ax2.set_title("Residual monitor")
# ax2.set_xlabel("Iteration")
# ax2.set_ylabel("Residual")

# # Plot residuals of each agent
# for id, this_res_hist in enumerate(residuals):
#     ax2.plot(np.arange(n_iter), this_res_hist, label=f"Agent {id}")
# ax2.plot(range(n_iter), [1/rho]*n_iter, label=f"Threshold")
# ax2.legend(loc='best')
# ax2.set_ylim(bottom=0, top=10)
# ax2.grid(True)


###     Plotting            - Dual Variables: Lambda

fig_lam, ax_lam = plt.subplots(dpi=200)
ax_lam.set_title(f"Lambda for agent {faulty_id}")
ax_lam.set_xlabel("Iteration")
ax_lam.set_ylabel("Lambda")
for id, _ in enumerate(agents):
    for i in range(lam_norm_history[id].shape[0]):
        ax_lam.plot(np.arange(n_iter), lam_norm_history[id][i, :].flatten(), label=f"Agent {id}, Edge {i}")
# ax_lam.legend(loc='best')
ax_lam.grid(True)


###     Plotting            - Dual Variables: Mu

# fig_mu, ax_mu = plt.subplots()
# ax_mu.set_title(f"Mu for agent {faulty_id}")
# ax_mu.set_xlabel("Iteration")
# ax_mu.set_ylabel("Mu")
# for id, _ in enumerate(agents):
#     for i in range(mu_norm_history[id].shape[0]):
#         ax_mu.plot(np.arange(n_iter), mu_norm_history[id][i, :].flatten(), label=f"Agent {id}, Neighbor {i}")
# # ax_mu.legend(loc='best')
# ax_mu.grid(True)



###     Plotting            - Norm Difference in R

fig_R_diff, ax_R_diff = plt.subplots(dpi=200)
ax_R_diff.set_title("Norm Difference between old R and new R")
ax_R_diff.set_xlabel("Outer Loop Iteration")
ax_R_diff.set_ylabel("|| R_old - R_new ||")
for agent_id, _ in enumerate(agents):
    ax_R_diff.plot(np.arange(n_scp), R_norm_diff_history[agent_id], label=f"Agent {agent_id}")
ax_R_diff.set_xlim((0, (n_scp-1)))
ax_R_diff.set_ylim((0, 3))
ax_R_diff.grid(True)


###     Plotting            - Show Plots
if show_plots:
    plt.show()
else:
    plt.close("all")