"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - Complex configuration from Shiraz's TAC paper with noise
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
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation

from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from time import time

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
rho             =   1.25
iam_noise       =   0.05
pos_noise       =   0.02
warm_start      =   False
lam_lim         =   1
mu_lim          =   1

# Show
show_plots = False

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
faulty_id = [0, 5, 7, 9, 10, 13]
err_scaling = 1
fault_vec   =   [np.array([[0.275, 0.447, 0.130]]).T,
                 np.array([[-0.849, 0.170, 0.888]]).T,
                 np.array([[0.761, -0.408, 0.438]]).T,
                 np.array([[-0.640, 0.260, -0.941]]).T,
                 np.array([[0.879, 0.425, -0.710]]).T,
                 np.array([[-0.534, -0.543, -0.588]]).T]
for index, agent_id in enumerate(faulty_id):
    # fault_vec.append( np.random.uniform(low=-err_scaling, high=err_scaling, size=(dim, 1)) )
    agents[agent_id].faulty = True
    agents[agent_id].error_vector = fault_vec[index][:,np.newaxis].reshape((dim, -1))

x_true = []
for id, agent in enumerate(agents):
    x_true.append(agent.error_vector)


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



###     Initializations     - Measurements and Positions
x_star = [np.zeros((dim, 1)) for i in range(num_agents)]                    # Equivalent to last element in x_history (below)
x_history = [np.zeros((dim, (n_iter))) for i in range(num_agents)]          # Value of x at each iteration of algorithm
x_norm_history = [np.zeros((1, (n_iter))) for i in range(num_agents)]       # Norm of difference between x_history and x_true
p_est = [agents[i].get_estimated_pos() for i in range(num_agents)]          # Will be updated as algorithm loops and err vector is reconstructed
p_hat = deepcopy(p_est)                                                     # CONSTANT: Reported positions of agents
p_true = [agents[i].get_true_pos() for i in range(num_agents)]              # CONSTANT: True pos
y = measurements(p_true, x_star)                                            # CONSTANT: Phi(p_hat + x_hat), true interagent measurement
residuals = [np.zeros(n_iter) for i in range(num_agents)]                   # Running residuals of each agent (residual <= 1 is nominal)



###      Initializations    - Optimization Parameters
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
print("Noise (pos, iam):", (pos_noise, iam_noise))
print("Number of agents:", num_agents)
print("Faulty Agent ID and Vector:")
for i, id in enumerate(faulty_id):
    print(f"\tID: {id}\t\t Vector: {fault_vec[i].flatten()}")


###     Store stuff
lam_norm_history = [np.zeros((len(agents[i].get_edge_indices()), n_iter)) for i in range(num_agents)]
mu_norm_history = [np.zeros((len(agents[i].get_neighbors()), n_iter)) for i in range(num_agents)]
sum_err_rmse = 0.0
start_time = time()
solver_err = False

###     Looping             - SCP Outer Loop
print("\nStarting Loop")
for outer_i in tqdm(range(n_scp), desc="SCP Loop", leave=True):
    # Noise in Position Estimate
    p_hat_noise = deepcopy(p_hat)
    for i, _ in enumerate(p_hat_noise):
        p_hat_noise[i] = p_hat[i] + np.random.uniform(low=-pos_noise, high=pos_noise, size=(dim, 1))

    new_measurement = measurements(p_hat_noise, x_star)
    z       =   [(y[i] - meas) for i, meas in enumerate(new_measurement)]
    R       =   get_Jacobian_matrix(p_hat_noise, x_star)

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
                    prob1.solve(solver=cp.MOSEK, verbose=show_prob1)
                except KeyboardInterrupt:
                    print("\t-> Keyboard Interrupt")
                    quit()
                # except:
                    solver_err = True
                    print("\t-> Solver Error")
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
                prob2.solve(solver=cp.MOSEK, verbose=show_prob2)
            except KeyboardInterrupt:
                print("\t-> Keyboard Interrupt")
                quit()
            except:
                solver_err = True
                print("\t-> Solver Error")
                break
            
            if prob2.status != cp.OPTIMAL:
                print("\nERROR Problem 2: Optimization problem not solved @ (%d, %d, %d)" % (inner_i, outer_i, agent_id))

            for _, nbr_id in enumerate(agent.get_neighbors()):
                agent.w[nbr_id] = deepcopy(np.array(agent.w_cp[nbr_id].value).reshape((-1, 1)))

        ##      Check               - Solver Error
        if solver_err:
            break
        
        
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


            # Summation for d() constraint
            for i, nbr_id in enumerate(agent.get_neighbors()):
                constr_d = agent.x_bar - agent.w[nbr_id]
                new_mu = (agent.mu[nbr_id] + rho * constr_d)
                agent.mu[nbr_id] = deepcopy(new_mu)
                mu_norm_history[agent_id][i, (inner_i + outer_i*n_admm)] = np.linalg.norm(deepcopy(new_mu))


    ###     END Looping         - ADMM Inner Loop
    
    # Check Solver Error
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

###     END Looping         - SCP Outer Loop

final_time = time()
print("==================================================================================")
print(f"IAM Noise: {iam_noise}")
print(f"Position Noise: {pos_noise}")
print(f"Penalty Parameter: {rho}")
print(f"Warm Start: {warm_start}")
print("----------------------------------------------------------------------------------")
print(f"Average RMSE: {sum_err_rmse / num_agents} m")
print(f"Elapsed Time: {final_time - start_time} seconds")
print(f"Average Time per Iteration: {(final_time - start_time)/n_iter} seconds")
print("==================================================================================")

final_iter = 1 + inner_i + outer_i*n_admm


###     Plotting            - Static Position Estimates
print("\nPlotting")
print()
plt.rcParams.update({'text.usetex': True,
                        'font.family': 'Helvetica'})

# Arrow Class
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

# Create position estimate over time data
p_hist = []
for id in range(num_agents):
    p_id = np.zeros((dim, n_iter))
    for iter in range(n_iter):
        p_id[:,iter] = p_hat[id].flatten() + x_history[id][:, iter]
    p_hist.append(p_id)

# Compare position estimates before and after reconstruction
fig1 = plt.figure(dpi=500,figsize=(4,4))
ax1 = fig1.add_subplot(projection='3d')
# ax1.set_title(r"Agent Position Estimates")
ax1.set_xlabel(r"$x\textnormal{-position}$")
ax1.set_ylabel(r"$y\textnormal{-position}$")
ax1.set_zlabel(r"$z\textnormal{-position}$")

ax1.set_xlim((0, 7))
ax1.set_ylim((0, 7))
ax1.set_zlim((0, 7))

err_marker = ax1.scatter([], [], marker=r'$\leftarrow$', color='k')
for agent_id, agent in enumerate(agents): # Draw points
    # ax1.scatter(p_hist[agent_id][0, -1], p_hist[agent_id][1, -1], p_hist[agent_id][2, -1], marker='*', c='c', label="After", s=100)
    # ax1.scatter(p_est[agent_id][0], p_est[agent_id][1], p_est[agent_id][2], marker='*', c='m', label="After", s=100)
    est_plot = ax1.scatter(p_hat[agent_id][0], p_hat[agent_id][1], p_hat[agent_id][2], facecolors='none', edgecolors='orangered', label="Estimated State", s=60, zorder=(num_faulty + 2*agent_id))
    true_plot = ax1.scatter(p_true[agent_id][0], p_true[agent_id][1], p_true[agent_id][2], marker='x', c='yellowgreen', label="True State", s=30, zorder=(num_faulty + 2*agent_id + 1))

arrow_prop_dict = dict(mutation_scale=4, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
for _, agents_id in enumerate(faulty_id):
    arrow_temp = Arrow3D([p_hat[agents_id][0,0], p_true[agents_id][0,0]], [p_hat[agents_id][1,0], p_true[agents_id][1,0]], [p_hat[agents_id][2,0], p_true[agents_id][2,0]], **arrow_prop_dict, zorder=(agents_id))
    ax1.add_artist(arrow_temp)

for i, edge in enumerate(edges): # Draw edges
    p1 = p_true[edge[0]]
    p2 = p_true[edge[1]]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    ax1.plot(x, y, z, c='k', linewidth=1, alpha=0.05, zorder=(num_faulty + num_agents + i))[0]
# plt.legend(["With Inter-agent Measurements", "Without Inter-agent Measurements", "True Position"], loc='best', fontsize=6, markerscale=0.4)


ax1.set_aspect('equal')
plt.legend([est_plot, true_plot, err_marker], [r"$\textnormal{Estimated State}$", r"$\textnormal{True State}$", r"$\textnormal{True Error}$"],
           fancybox=True, loc='upper left', bbox_to_anchor=(-0.22, 1.0), ncols=3, fontsize=12)
plt.grid(True)


dt_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
fname_poses = "fig/3D-NoisyComplex/positions-" + dt_string + ".svg"
# plt.savefig(fname_poses, dpi=500, bbox_inches='tight')



###     Plotting            - Error Convergence

# Show convergence of estimated error vector to true error vector over time
x_norm_history = [x_norm_history[i].flatten() for i in range(num_agents)]
fig_err = plt.figure(dpi=500, figsize=(9,4))
ax_err = fig_err.add_subplot()
lines = [None] * num_agents

# Plot
for agent_id, agent in enumerate(agents):
    label_str = f"Agent {agent_id}"
    plt_color = 'slategray'
    if agent_id in faulty_id:
        plt_color = 'orangered'
    lines[agent_id] = ax_err.plot(total_iterations, x_norm_history[agent_id], c=plt_color, label=label_str)[0]

# ax_err.set_title(r'Error Vector Convergence ( $\rho = {}$ )'.format(rho))
ax_err.set_xlabel(r'\textnormal{ADMM Iterations}', fontsize=16)
ax_err.set_ylabel(r'$ \| \mathbf{x}[i] - ( \mathbf{x}^* [i] + \hat{\mathbf{x}}[i]) \| $')
ax_err.set_ylim((0, 1.25))
ax_err.set_xlim((0, (n_iter - 1)))
ax_err.set_xticks(ticks=np.arange(0, n_iter, n_admm))
ax_err.set_yticks(ticks=np.arange(0, 1.25, 0.25))
ax_err.legend([lines[1], lines[faulty_id[0]]], [r'$i \in \textnormal{Nominal Agents}$', r'$i \in \textnormal{Faulty Agents}$'])
ax_err.grid(True)

fname_err = "fig/3D-NoisyComplex/err_conv_" + dt_string + ".svg"
# plt.savefig(fname_err, dpi=500)



###     Plotting            - Animation

# Start figure
fig2 = plt.figure(dpi=500, figsize=(16,9))
ax2_1 = fig2.add_subplot(1, 2, 1, projection='3d')
ax2_2 = fig2.add_subplot(1, 2, 2)
# ax2.set_title(r"Swarm Position ( $\rho = {}$ )".format(rho))
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
    scat_pos_hat[agent_id] = ax2_1.plot(p_hat[agent_id][0], p_hat[agent_id][1], p_hat[agent_id][2], 
                                        marker='o', markerfacecolor='none', c='orangered', linestyle='None', 
                                        label="Before", markersize=15)[0]
    scat_pos_true[agent_id] = ax2_1.plot(p_true[agent_id][0], p_true[agent_id][1], p_true[agent_id][2], 
                                        marker='x', c='yellowgreen', linestyle='None', label="True", markersize=8)[0]
    scat_pos_est[agent_id] = ax2_1.plot(p_hist[agent_id][0, 0], p_hist[agent_id][1, 0], p_hist[agent_id][2, 0], 
                                        marker='*', c='c', linestyle='None', label="After", markersize=10)[0]
    # ax2.text(p_hat[agent_id][0], p_hat[agent_id][1], p_hat[agent_id][2],
    #         "%s" % (agent_id), size=10, zorder=1, color='k')

# Draw line for each edge of network
for i, edge in enumerate(edges):
    p1 = p_hist[edge[0]][:, 0]
    p2 = p_hist[edge[1]][:, 0]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    line_pos_est[i] = ax2_1.plot(x, y, z, c='k', linewidth=1, alpha=0.05)[0]
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
    # Dont reanimate
    if frame >= final_iter:
        return
    
    updated_ax = []
    # Draw each agent's original estimated, current estimated, and true positions
    # Also Draw error convergence plots
    for agent_id, _ in enumerate(agents):
        # Positions 
        new_pos = (float(p_hist[agent_id][0, frame]), float(p_hist[agent_id][1, frame]), float(p_hist[agent_id][2, frame]))
        scat_pos_est[agent_id].set_data([new_pos[0]], [new_pos[1]])
        scat_pos_est[agent_id].set_3d_properties([new_pos[2]])
        updated_ax.append(scat_pos_est[agent_id])
        
        # Error
        err_lines[agent_id].set_data(total_iterations[0:frame], x_norm_history[agent_id][0:frame])
        updated_ax.append(err_lines[agent_id])
        
    
    # Draw line for each edge of network
    for i, edge in enumerate(edges):
        p1 = p_hist[edge[0]][:, frame]
        p2 = p_hist[edge[1]][:, frame]
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        z = [p1[2], p2[2]]

        line_pos_est[i].set_data(x, y)
        line_pos_est[i].set_3d_properties(z)

        updated_ax.append(line_pos_est[i])
    
    return updated_ax
    
# Call update function
pos_ani = animation.FuncAnimation(fig=fig2, func=update_pos_plot, frames=n_iter, interval=100, blit=False, repeat=True)
fname = "fig/3D-NoisyComplex/combined_ani-" + dt_string + ".mp4"
pos_ani.save(filename=fname)#, writer="pillow")


###     Plotting            - Residuals and Threshold

# Start figure
fig2, ax2 = plt.subplots()
ax2.set_title("Residual monitor")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Residual")

# Plot residuals of each agent
for id, this_res_hist in enumerate(residuals):
    ax2.plot(np.arange(n_iter), this_res_hist, label=f"Agent {id}")
ax2.plot(range(n_iter), [1/rho]*n_iter, label=f"Threshold")
ax2.legend(loc='best')
ax2.set_ylim(bottom=0, top=10)
ax2.grid(True)


###     Plotting            - Dual Variables: Lambda
fig_lam, ax_lam = plt.subplots()
ax_lam.set_title(f"Lambda for agent {faulty_id}")
ax_lam.set_xlabel("Iteration")
ax_lam.set_ylabel("Lambda")
for id, _ in enumerate(agents):
    for i in range(lam_norm_history[id].shape[0]):
        ax_lam.plot(np.arange(n_iter), lam_norm_history[id][i, :].flatten(), label=f"Agent {id}, Edge {i}")
# ax_lam.legend(loc='best')
ax_lam.grid(True)


###     Plotting            - Dual Variables: Mu
fig_mu, ax_mu = plt.subplots()
ax_mu.set_title(f"Mu for agent {faulty_id}")
ax_mu.set_xlabel("Iteration")
ax_mu.set_ylabel("Mu")
for id, _ in enumerate(agents):
    for i in range(mu_norm_history[id].shape[0]):
        ax_mu.plot(np.arange(n_iter), mu_norm_history[id][i, :].flatten(), label=f"Agent {id}, Neighbor {i}")
# ax_mu.legend(loc='best')
ax_mu.grid(True)


###     Plotting            - Show Plots
if show_plots:
    plt.show()
else:
    plt.close("all")