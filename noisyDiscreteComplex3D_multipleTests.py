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
import multiprocessing as mp

from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from time import time

###     Imports             - User-Defined Files
from generic_agent import GenericAgent as Agent
from iam_models import distance


def main_func(rho, warm_start, trial_num, return_dict):
    ###     Start
    print(f"Starting ({rho}, {warm_start}, {trial_num})")


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
    # rho             =   1.
    iam_noise       =   0.04
    pos_noise       =   0.02
    # warm_start      =   True
    lam_lim         =   1.5
    mu_lim          =   5e-3
    show_plots      =   False
    save_ani        =   False
    show_prints     =   False

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
    if show_prints:
        print("\n~ ~ ~ ~ PARAMETERS ~ ~ ~ ~")
        print("rho:", rho)
        print("Number of agents:", num_agents)
        print("Faulty Agent ID and Vector:")
        for i, id in enumerate(faulty_id):
            print(f"\tID: {id}\t\t Vector: {fault_vec[i].flatten()}")


    ### Store stuff
    lam_norm_history = [np.zeros((len(agents[i].get_edge_indices()), n_iter)) for i in range(num_agents)]
    mu_norm_history = [np.zeros((len(agents[i].get_neighbors()), n_iter)) for i in range(num_agents)]
    sum_err_rmse = 0.0
    start_time = time()
    solver_err = False

    ###     Looping             - SCP Outer Loop

    for outer_i in tqdm(range(n_scp), desc="SCP Loop", leave=False, disable=(not show_prints)):

        # Set p_hat depending on iteration number
        for id, agent in enumerate(agents):
            if (id in faulty_id2) and (outer_i >= round(n_scp*2/3)):
                agent.faulty = True
                x_true[id] = fault_vec[faulty_id.index(id)]

            elif (id in faulty_id1) and ( (outer_i >= round(n_scp*1/3)) and (outer_i < round(n_scp*2/3)) ):
                agent.faulty = True
                x_true[id] = fault_vec[faulty_id.index(id)]

            else:
                agent.faulty = False
                x_true[id] *= 0.
            
            p_hat[id] = agent.get_estimated_pos()
            

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
        for inner_i in tqdm(range(n_admm), desc="ADMM Loop", leave=False, disable=(not show_prints)):

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
                    except KeyboardInterrupt:
                        print("\n\n~~~ KEYBOARD INTERRUPT ~~~\n\n")
                        quit()
                    except:
                        print("\t -> SOLVER ERROR")
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
                except KeyboardInterrupt:
                    print("\n\n~~~ KEYBOARD INTERRUPT ~~~\n\n")
                    quit()
                except:
                    print("\t -> SOLVER ERROR")
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
                    
                    if (not warm_start) and (np.linalg.norm(new_lam) > lam_lim):
                        # print(f"RESET LAM: Agent {agent_id} at SCP {outer_i}")
                        reset_lam[agent_id] = True

                # Summation for d() constraint
                for i, nbr_id in enumerate(agent.get_neighbors()):
                    constr_d = agent.x_bar - agent.w[nbr_id]

                    new_mu = (agent.mu[nbr_id] + rho * constr_d)
                    agent.mu[nbr_id] = deepcopy(new_mu)
                    mu_norm_history[agent_id][i, (inner_i + outer_i*n_admm)] = np.linalg.norm(deepcopy(new_mu))

                    if (not warm_start) and (np.linalg.norm(new_mu) > mu_lim):
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

        ##      Check for Solver Error
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
        
        # Check if a reset flag was set
        for agent_id, agent in enumerate(agents):
            if (not warm_start) and (reset_lam[agent_id] or reset_mu[agent_id]):
                # print(f"RESET DUAL: Agent {agent_id} at SCP {outer_i}")
                agent.init_lam(np.zeros((1, 1)), agent.get_edge_indices())
                agent.init_mu(np.zeros((dim, 1)), agent.get_neighbors())
                reset_mu[agent_id] = False
                reset_lam[agent_id] = False

    ###     END Looping         - SCP Outer Loop

    final_time = time()
    # print(f"Inner: {inner_i}; \t Outer: {outer_i}")
    final_iteration = 1 + (inner_i) + (outer_i)*n_admm
    if show_prints:
        print(f"==================================================================================")
        print(f"IAM Noise: {iam_noise}")
        print(f"Position Noise: {pos_noise}")
        print(f"Penalty Parameter: {rho}")
        print(f"Warm Start: {warm_start}")
        print(f"----------------------------------------------------------------------------------")
        print(f"Average RMSE: {sum_err_rmse / num_agents} m")
        print(f"Elapsed Time: {final_time - start_time} seconds")
        print(f"Average Time per Iteration: {(final_time - start_time)/n_iter} seconds")
        print(f"Average Time per Iteration per Agent: {(final_time - start_time)/(n_iter*num_agents)} seconds")
        print(f"==================================================================================")

    # return (sum_err_rmse, num_agents, final_iteration)
    return_dict[(rho, warm_start, trial_num)] = (sum_err_rmse, num_agents, final_iteration)
    return
    


### Main
if __name__ == '__main__':
    # Multiprocessing Handling
    manager = mp.Manager()
    return_dict = manager.dict()

    # Trial Parameters and Related Variables
    num_trials = 100
    rho_arr = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5]
    warm_start_arr = [True, False]
    trials_rmse_arr = np.zeros((len(warm_start_arr), len(rho_arr), num_trials))
    avg_rmse_arr = np.zeros((len(warm_start_arr), len(rho_arr)))

    # Create Processes
    pool = mp.Pool(processes=11)
    args = []
    for i, warm_start in enumerate(warm_start_arr):
        for j, rho in enumerate(rho_arr):
            for trial_num in range(num_trials):
                args.append((rho, warm_start, trial_num, return_dict))
    pool.starmap(main_func, args)
    
    # Get results in useful format
    for key in return_dict.keys():
        # Extract Key Values
        rho = key[0] 
        warm_start = key[1]
        trial_num = key[2]

        # Extract Dictionary Value
        val = return_dict[key]
        trial_rmse = val[0]
        num_agents = val[1]
        final_iter = val[2]

        # Compute and Assign
        this_avg_rmse = trial_rmse / (num_agents * final_iter)
        warm_ind = warm_start_arr.index(warm_start)
        rho_ind = rho_arr.index(rho)
        trials_rmse_arr[warm_ind, rho_ind, trial_num] = this_avg_rmse
        avg_rmse_arr[warm_ind, rho_ind] += this_avg_rmse

    # Save numpy array binary for later use
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname_rmse = f"data/multiple_discrete/discrete_data_{dt_string}"
    np.save(fname_rmse, trials_rmse_arr)

    # Print data results to terminal
    print(f"\nAll {num_trials} trials completed. Results: ")
    print("========================================================")
    for i, rho in enumerate(rho_arr):
        for j, warm_start in enumerate(warm_start_arr):
            this_avg = np.mean(trials_rmse_arr[j, i, :].flatten())
            this_std = np.std(trials_rmse_arr[j, i, :].flatten())

            print(f"Warm Start: {warm_start}; \tRho: {rho}; \tAvg RMSE: {this_avg}; \tStd RMSE: {this_std}")
        print(f" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("========================================================\n")


    ## Plot avg RMSE vs. rho

    fig = plt.figure(dpi=500)
    ax = fig.add_subplot()
    ax.loglog(rho_arr, avg_rmse_arr[0, :]/num_trials, c='orangered', label="Warm Start")
    ax.loglog(rho_arr, avg_rmse_arr[1, :]/num_trials, c='aquamarine', label="Cold Start")
    ax.set_title("")
    ax.set_xlabel(r"$ \rho $")
    ax.set_ylabel("Average RMSE")
    ax.legend(loc='best')
    ax.grid(True)

    fname_plt = f"fig/3D-NoisyDiscreteComplex/rmse_compare_{dt_string}.svg"
    plt.savefig(fname_plt, dpi=500)

    plt.show()