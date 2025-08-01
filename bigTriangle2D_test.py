"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - 2D Project
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
import matplotlib.animation as animation

from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


###     Imports             - User-Defined Files
from generic_agent import GenericAgent as Agent
from iam_models import distance, bearing



###     Initializations     - Scalars
dim             =   2   # 2 or 3
ydim            =   dim # 1 for dist measurements, dim for bearing measurements
num_agents      =   10
num_faulty      =   3   # must be << num_agents for sparse error assumption
n_scp           =   50  # Number of SCP iterations
n_admm          =   20  # Number of ADMM iterations
n_iter          =   n_admm * n_scp
show_prob1      =   False
show_prob2      =   False

###     Initializations     - Agents
# 3 agents at vertices of equilateral triangle, 3 agents at midpoints of edges
agents      =   [None] * num_agents
d           =   3   # small triangle side length
agents[0]   =   Agent(agent_id= 0,
                      init_position= d*np.array([[-1.5, 0]]).T)
agents[1]   =   Agent(agent_id= 1,
                      init_position= d*np.array([[-1, np.sqrt(3)/2]]).T)
agents[2]   =   Agent(agent_id= 2,
                      init_position= d*np.array([[-0.5, 0]]).T)
agents[3]   =   Agent(agent_id= 3,
                      init_position= d*np.array([[0, np.sqrt(3)/2]]).T)
agents[4]   =   Agent(agent_id= 4,
                      init_position= d*np.array([[-0.5, np.sqrt(3)]]).T)
agents[5]   =   Agent(agent_id= 5,
                      init_position= d*np.array([[0, 3*np.sqrt(3)/2]]).T)
agents[6]   =   Agent(agent_id= 6,
                      init_position= d*np.array([[0.5, np.sqrt(3)]]).T)
agents[7]   =   Agent(agent_id= 7,
                      init_position= d*np.array([[1, np.sqrt(3)/2]]).T)
agents[8]   =   Agent(agent_id= 8,
                      init_position= d*np.array([[0.5, 0]]).T)
agents[9]   =   Agent(agent_id= 9,
                      init_position= d*np.array([[1.5, 0]]).T)

def random_err_vec(norm_bound, low_bound=1.0):
    v = np.random.normal(size=(dim, 1))
    unit_v = v / np.linalg.norm(v)
    err = unit_v * np.random.uniform(low=low_bound, high=norm_bound)
    return err

# Add random error vector
# faulty_ids = []
# faulty_vecs = []
# for i in range(num_faulty):
#     succ = False
#     while not succ:
#         id = np.random.randint(0, num_agents)
#         err = random_err_vec(d/2)
#         if id not in faulty_ids:
#             faulty_ids.append(id)
#             faulty_vecs.append(err)
#             succ = True
#             agents[id].faulty = True
#             agents[id].error_vector = err

# Select Agents
# faulty_ids = [] # No faulty
# faulty_ids = [1, 6, 9] # Unclustered
faulty_ids = [1, 6]
# faulty_ids = [7, 8, 9] # Bottom Right Cluster
num_faulty = len(faulty_ids)

# Add error vector
faulty_vecs = []
for i in range(num_faulty):
    faulty_vecs.append(random_err_vec(d/3, d/6))
    agents[faulty_ids[i]].faulty = True
    agents[faulty_ids[i]].error_vector = faulty_vecs[i]
# agents[faulty_ids[0]].faulty = True
# agents[faulty_ids[0]].error_vector = faulty_vecs[0]
# agents[faulty_ids[1]].faulty = True
# agents[faulty_ids[1]].error_vector = faulty_vecs[1]
# agents[faulty_ids[2]].faulty = True
# agents[faulty_ids[2]].error_vector = faulty_vecs[2]

x_true = []
for id, agent in enumerate(agents):
    x_true.append(agent.error_vector)


# Set Neighbors
edges_1d                = [[0,1], [0,2], [1,2],
                           [1,3], [1,4], [2,3],
                           [2,8], [3,4], [3,8],
                           [3,6], [3,7], [4,5],
                           [4,6], [5,6], [6,7],
                           [7,8], [7,9], [8,9]] # edges are only one way
edges = []
for arc in edges_1d:
    edges.append([arc[0], arc[1]])
    edges.append([arc[1], arc[0]])
                           
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
        # this_iam = distance((p[edge[0]] + x_hat[edge[0]]), (p[edge[1]] + x_hat[edge[1]]))
        this_iam = bearing((p[edge[0]] + x_hat[edge[0]]), (p[edge[1]] + x_hat[edge[1]]))
        measurements.append(this_iam)
    return measurements

# Finds row of R
def get_Jacobian_row(edge_ind, p, x):
    edge = edges[edge_ind]
    agent1_id = edge[0]
    agent2_id = edge[1]
    pos1 = p[edge[1]] + x[edge[1]]
    pos2 = p[edge[0]] + x[edge[0]]
    disp    = (pos1 - pos2)
    # R_k = np.zeros((1, dim*num_agents))
    R_k = np.zeros((ydim, dim*num_agents))

    dist = distance(pos1, pos2)
    # R_k[:, dim*agent2_id:dim*(agent2_id + 1)] = disp.T  / dist
    # R_k[:, dim*agent1_id:dim*(agent1_id + 1)] = -disp.T / dist
    mat1 = ((1*(disp @ disp.T)) / np.power(dist, 3)) + np.eye(dim) * np.power(dist, -1)
    mat2 = ((1*(disp @ disp.T)) / np.power(dist, 3)) + np.eye(dim) * np.power(dist, -1)

    R_k[:, dim*agent2_id:dim*(agent2_id + 1)] = mat2
    R_k[:, dim*agent1_id:dim*(agent1_id + 1)] = -1*mat1
    
    return R_k

# Computes whole R matrix
def get_Jacobian_matrix(p, x):
    R = []

    for edge_ind, edge in enumerate(edges):
        R_k = get_Jacobian_row(edge_ind, p, x)
        # print("Edge:", edge_ind, " - R_k:", R_k)
        R.append(R_k)
    
    return R


###     Initializations     - Measurements and Positions
x_star = [np.zeros((dim, 1)) for i in range(num_agents)]                    # Equivalent to last element in x_history (below)
x_history = [np.zeros((dim, (n_iter))) for i in range(num_agents)]          # Value of x at each iteration of algorithm
x_norm_history = [np.zeros((1, (n_iter))) for i in range(num_agents)]       # Norm of difference between x_history and x_true
p_est = [agents[i].get_estimated_pos() for i in range(num_agents)]          # Will be updated as algorithm loops and err vector is reconstructed
p_hat = deepcopy(p_est)                                                     # CONSTANT: Reported positions of agents
p_true = [agents[i].get_true_pos() for i in range(num_agents)]              # CONSTANT: True pos
y = measurements(p_true, x_star)                                            # CONSTANT: Phi(p_hat + x_hat), true interagent measurement


###      Initializations    - Optimization Parameters
rho = 0.5
total_iterations = np.arange((n_iter))
for agent_id, agent in enumerate(agents):
    num_edges       = len(agent.get_edge_indices())
    num_neighbors   = len(agent.get_neighbors())

    # CVX variables
    agent.init_x_cp(cp.Variable((dim, 1)))
    agent.init_w_cp(cp.Variable((dim, 1)), agent.get_neighbors())

    # Parameters
    agent.init_x_bar(np.zeros((dim, 1)))
    agent.init_lam(np.zeros((ydim, 1)), agent.get_edge_indices())
    agent.init_mu(np.zeros((dim, 1)), agent.get_neighbors())
    agent.init_x_star(np.zeros((dim, 1)), agent.get_neighbors()) # own err is last elem
    agent.init_w(np.zeros((dim, 1)), agent.get_neighbors())


###     Initializations     - List Parameters
print("\n~ ~ ~ ~ PARAMETERS ~ ~ ~ ~")
print("rho:", rho)
print("Number of agents:", num_agents)
print("Faulty Agents' ID and Vector:")
for i in range(num_faulty):
    print(f" Agent {faulty_ids[i]} with vector {faulty_vecs[i].flatten()}, norm = {np.linalg.norm(faulty_vecs[i])}")


###     Looping             - SCP Outer Loop
print("\nLooping")
for outer_i in tqdm(range(n_scp), desc="SCP Loop", leave=False):
    new_measurement = measurements(p_hat, x_star)
    z       =   [(y[i] - meas) for i, meas in enumerate(new_measurement)]
    R       =   get_Jacobian_matrix(p_hat, x_star)

    for agent in agents:
        agent.init_w(np.zeros((dim, 1)), agent.get_neighbors())


    ###     Looping             - ADMM Inner Loop
    for inner_i in tqdm(range(n_admm), desc="ADMM Loop", leave=False):

        ##      Minimization        - Primal Variable 1
        for agent_id, agent in enumerate(agents):
            objective = cp.norm(agent.x_star[agent_id] + agent.x_cp)
            
            # Summation for c() constraint
            for _, edge_ind in enumerate(agent.get_edge_indices()): 
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_cp - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w[agent_id]
                
                objective += ((rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[edge_ind].T @ (constr_c))
            
            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()): 
                constr_d = agent.x_cp - agent.w[nbr_id]
                objective += ((rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob1 = cp.Problem(cp.Minimize(objective), [])
            prob1.solve(verbose=show_prob1, solver=cp.MOSEK)
            if prob1.status != cp.OPTIMAL:
                print("\nERROR Problem 1: Optimization problem not solved @ (%d, %d, %d)" % (inner_i, outer_i, agent_id))
            
            agent.x_bar = deepcopy(np.array(agent.x_cp.value).reshape((-1, 1)))
            new_x = deepcopy(agent.x_bar.flatten()) + x_star[agent_id].flatten()

            x_history[agent_id][:, inner_i + outer_i*n_admm] = new_x.flatten()
            x_norm_history[agent_id][:, inner_i + outer_i*n_admm] = np.linalg.norm(new_x.flatten() + x_true[agent_id].flatten())

        ##      Minimization        - Thresholding Parameter
        # TODO: Implement
        # Used for identifying faults, not pressing issue

        ##      Minimization        - Primal Variable 2
        for agent_id, agent in enumerate(agents):
            objective = cp.norm(agent.x_star[agent_id] + agent.x_bar)

            # Summation for c() constraint
            for edge_ind in agent.get_edge_indices(): 
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_bar - z[edge_ind]
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
            prob2.solve(verbose=show_prob2, solver=cp.MOSEK)
            if prob2.status != cp.OPTIMAL:
                print("\nERROR Problem 2: Optimization problem not solved @ (%d, %d, %d)" % (inner_i, outer_i, agent_id))

            for _, nbr_id in enumerate(agent.get_neighbors()):
                agent.w[nbr_id] = deepcopy(np.array(agent.w_cp[nbr_id].value).reshape((-1, 1)))


        ##      Multipliers         - Update Lagrangian Multipliers of Minimization Problem
        for agent_id, agent in enumerate(agents):
            
            # Summation for c() constraint
            for _, edge_ind in enumerate(agent.get_edge_indices()):
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w[agent_id]
                
                agent.lam[edge_ind] = deepcopy(agent.lam[edge_ind] + rho * constr_c)

            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()):
                constr_d = agent.x_bar - agent.w[nbr_id]
                agent.mu[nbr_id] = deepcopy(agent.mu[nbr_id] + rho * constr_d)

    ###     END Looping         - ADMM Inner Loop
    
    # Update Error Vectors after ADMM subroutine
    for agent_id, agent in enumerate(agents): 
        for list_ind, nbr_id in enumerate(agent.get_neighbors()):
            agent.x_star[nbr_id] = agent.x_star[nbr_id] + agents[nbr_id].x_bar
        
        agent.x_star[agent_id] = agent.x_star[agent_id] + agent.x_bar
        x_star[agent_id] = agent.x_star[agent_id]
        
        # Update position and x_dev
        p_est[agent_id] = p_hat[agent_id] + x_star[agent_id]

###     END Looping         - SCP Outer Loop



###     Plotting            - Static Position Estimates
print("\nPlotting")
print()
plt.rcParams.update({'text.usetex': True,
                        'font.family': 'Helvetica'})
dt_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


# Compare position estimates before and after reconstruction
fig1 = plt.figure(dpi=500,figsize=(4,4))
ax1 = fig1.add_subplot()
# ax1.set_title(r"Agent Position Estimates")
ax1.set_xlabel(r"$x\textnormal{-position}$")
ax1.set_ylabel(r"$y\textnormal{-position}$")

ax1.set_xlim((-6, 6))
ax1.set_ylim((-2, 10))

err_marker = ax1.scatter([], [], marker=r'$\leftarrow$', color='k')
for agent_id, agent in enumerate(agents):
    est_plot = ax1.scatter(p_hat[agent_id][0], p_hat[agent_id][1], facecolors='none', edgecolors='orangered', label="Estimated State", s=60)
    # plt.scatter(p_est[agent_id][0], p_est[agent_id][1], marker='*', c='m', label="After Reconstruction")
    true_plot = ax1.scatter(p_true[agent_id][0], p_true[agent_id][1], marker='x', c='yellowgreen', label="True State", s=30)

# arrow_prop_dict = dict(mutation_scale=4, arrowstyle='-|>', color='k')
for _, agents_id in enumerate(faulty_ids):
    arrow_temp = FancyArrowPatch((p_hat[agents_id][0,0], p_hat[agents_id][1,0]), (p_true[agents_id][0,0], p_true[agents_id][1,0]),
                                 mutation_scale=4, arrowstyle='-|>', color='k')
    ax1.add_patch(arrow_temp)

for i, edge in enumerate(edges): # Draw edges
    p1 = p_true[edge[0]]
    p2 = p_true[edge[1]]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    ax1.plot(x, y, c='k', linewidth=1, alpha=0.05)[0]

ax1.set_aspect('equal')
plt.legend([est_plot, true_plot, err_marker], [r"$\textnormal{Estimated State}$", r"$\textnormal{True State}$", r"$\textnormal{True Error}$"],
           fancybox=True, loc='upper right', ncols=1, fontsize=8)
plt.grid(False)

fname_poses = "fig/2D-BigTriangle/positions-" + dt_string + ".svg"
# plt.savefig(fname_poses, dpi=500, bbox_inches='tight')


###     Plotting            - Error Convergence
# Show convergence of estimated error vector to true error vector over time
#TODO: This needs to be fixed.
# x_norm_history = [x_norm_history[id].flatten() for i in range(num_agents)]
# plt.figure()
# for agent_id, agent in enumerate(agents):
#     label_str = "Agent " + str(agent_id)
#     plt.plot(total_iterations, x_norm_history[agent_id], label=label_str)
# plt.title("Convergence of Error Vector")
# plt.xlabel("Iterations")
# plt.ylabel("||x* - x||")
# plt.xlim((0, n_scp*n_admm))
# plt.legend(loc='best')
# plt.grid(True)



###     Plotting            - Animation
# Create position estimate over time data
p_hist = []
for id in range(num_agents):
    p_id = np.zeros((dim, n_iter))
    for iter in range(n_iter):
        p_id[:,iter] = p_hat[id].flatten() + x_history[id][:, iter]
    p_hist.append(p_id)

# Start figure
fig, ax = plt.subplots(dpi=200)
ax.set_title("Agent Estimated Position")
ax.set_xlabel("x position")
ax.set_ylabel("y position")
scat_pos_est = [None] * num_agents # Position estimate during reconstruction
scat_pos_hat = [None] * num_agents # Initial position estimate
scat_pos_true = [None] * num_agents # True positions
line_pos_est = [None] * len(edges) # Inter-agent communication

# Draw each agent's original estimated, current estimated, and true positions
for agent_id, _ in enumerate(agents):
    scat_pos_est[agent_id] = ax.scatter(p_hist[agent_id][0, 0], p_hist[agent_id][1, 0], marker='*', c='c', label="After", s=100)
    scat_pos_hat[agent_id] = ax.scatter(p_hat[agent_id][0], p_hat[agent_id][1], facecolors='none', edgecolors='orangered', label="Before", s=100)
    scat_pos_true[agent_id] = ax.scatter(p_true[agent_id][0], p_true[agent_id][1], marker='x', c='g', label="True", s=100)

# Draw line for each edge of network
for i, edge in enumerate(edges):
    p1 = p_hist[edge[0]][:, 0]
    p2 = p_hist[edge[1]][:, 0]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    line_pos_est[i] = ax.plot(x, y, c='k', linewidth=1, alpha=0.5)[0]
ax.legend(["With Inter-agent Measurements", "Without Inter-agent Measurements", "True Position"], loc='best', fontsize=6, markerscale=0.4)
ax.grid(True)

# Update function
def update_pos_plot(frame):
    updated_ax = []
    # Draw each agent's original estimated, current estimated, and true positions
    for agent_id, _ in enumerate(agents):
        scat_pos_est[agent_id].set_offsets(p_hist[agent_id][:, frame])
        # scat_pos_hat[agent_id].set_offsets(p_hat[agent_id][:, frame])
        # scat_pos_true[agent_id].set_offsets(p_true[agent_id])
        updated_ax.append(scat_pos_est[agent_id])
    
    # Draw line for each edge of network
    for i, edge in enumerate(edges):
        p1 = p_hist[edge[0]][:, frame]
        p2 = p_hist[edge[1]][:, frame]
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        line_pos_est[i].set_xdata(x)
        line_pos_est[i].set_ydata(y)
        updated_ax.append(line_pos_est[i])
    
    return updated_ax

# Call update function
pos_ani = animation.FuncAnimation(fig=fig, func=update_pos_plot, frames=n_iter, interval=100)
dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fname = "fig/2D-BigTriangle/pos2D_ani_" + dt_string + ".gif"
# pos_ani.save(filename=fname, writer="pillow")



###     Plotting            - Error Convergence
# Show convergence of estimated error vector to true error vector over time
x_norm_history = [x_norm_history[i].flatten() for i in range(num_agents)]
fig_err = plt.figure(dpi=500, figsize=(9,4))
ax_err = fig_err.add_subplot()
lines = [None] * num_agents
for agent_id, agent in enumerate(agents):
    label_str = f"Agent {agent_id}"
    plt_color = 'slategray'
    if agent_id in faulty_ids:
        plt_color = 'orangered'
    lines[agent_id] = ax_err.plot(total_iterations, x_norm_history[agent_id], c=plt_color, label=label_str)[0]

# plt.title(r'Error Vector Convergence ( $\rho = {}$ )'.format(rho))
plt.xlabel(r'\textnormal{ADMM Iterations}', fontsize=16)
plt.ylabel(r'$ \| \mathbf{x}[i] - ( \mathbf{x}^* [i] + \hat{\mathbf{x}}[i]) \| $')
plt.ylim((0, 1.5))
plt.xlim((0, (n_iter - 1)))
plt.xticks(ticks=np.arange(0, n_iter, n_admm))
plt.yticks(ticks=np.arange(0, 1.5, 0.25))
# plt.legend([lines[0], lines[faulty_ids[0]]], [r'$i \in \textnormal{Nominal Agents}$', r'$i \in \textnormal{Faulty Agents}$'])
plt.grid(True)


fname_err = "fig/2D-BigTriangle/err_conv-" + dt_string + ".pdf"
# plt.savefig(fname_err, dpi=500, bbox_inches='tight')



###     Plotting            - Error Reconstruction
# Show growth of reconstructed error vector over time
x_list = [np.linalg.norm(x_history[i], axis=0) for i in range(num_agents)]
fig_err = plt.figure(dpi=500, figsize=(9,4))
ax_err = fig_err.add_subplot()
lines = [None] * num_agents
for agent_id, agent in enumerate(agents):
    label_str = f"Agent {agent_id}"
    plt_color = 'slategray'
    if agent_id in faulty_ids:
        plt_color = 'orangered'
    lines[agent_id] = ax_err.plot(total_iterations, x_list[agent_id], c=plt_color, label=label_str)[0]

# plt.title(r'Error Vector Convergence ( $\rho = {}$ )'.format(rho))
plt.xlabel(r'\textnormal{ADMM Iterations}', fontsize=16)
plt.ylabel(r'$ \| \mathbf{x}^* [i] + \hat{\mathbf{x}}[i] \| $')
plt.ylim((0, 1.5))
plt.xlim((0, (n_iter - 1)))
plt.xticks(ticks=np.arange(0, n_iter, n_admm))
plt.yticks(ticks=np.arange(0, 1.5, 0.25))
# plt.legend([lines[0], lines[faulty_ids[0]]], [r'$i \in \textnormal{Nominal Agents}$', r'$i \in \textnormal{Faulty Agents}$'])
plt.grid(True)


fname_recons = "fig/2D-BigTriangle/err_recons-" + dt_string + ".pdf"
plt.savefig(fname_recons, dpi=500, bbox_inches='tight')


###     Plotting            - Show Plots
plt.show()