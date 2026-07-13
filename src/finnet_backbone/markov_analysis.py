"""
Markov chain models for financial regime transitions and random walks on networks.
"""
import numpy as np
import networkx as nx


class MarketRegimeChain:
    """
    Discrete-time Markov chain modeling transitions between market regimes.
    
    States
    ------
    0 : Bull market
    1 : Bear market
    2 : High volatility
    """
    
    def __init__(self, transition_matrix=None):
        if transition_matrix is None:
            # Empirical transition probabilities (can be calibrated from data)
            self.P = np.array([
                [0.90, 0.05, 0.05],
                [0.15, 0.75, 0.10],
                [0.20, 0.15, 0.65]
            ])
        else:
            self.P = np.asarray(transition_matrix)
        
        self.states = ['Bull', 'Bear', 'Volatile']
        self.current = 0
    
    def step(self):
        """Advance one time step."""
        probs = self.P[self.current]
        self.current = np.random.choice(len(self.states), p=probs)
        return self.states[self.current]
    
    def simulate(self, n_steps):
        """Generate a trajectory of length n_steps."""
        trajectory = [self.states[self.current]]
        for _ in range(n_steps):
            trajectory.append(self.step())
        return trajectory
    
    def stationary_distribution(self):
        """
        Compute the stationary distribution as the left eigenvector 
        corresponding to eigenvalue 1.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        return pi / np.sum(pi)


def random_walk_centrality(G, n_steps=10000, n_walkers=100):
    """
    Estimate node centrality via random walk visitation frequency.
    
    Parameters
    ----------
    G : networkx.Graph
        Weighted financial network.
    n_steps : int
        Length of each random walk.
    n_walkers : int
        Number of independent walkers.
    
    Returns
    -------
    centrality : dict
        Mapping of node -> visitation frequency.
    """
    nodes = list(G.nodes())
    visit_counts = {node: 0 for node in nodes}
    total_visits = 0
    
    for _ in range(n_walkers):
        current = np.random.choice(nodes)
        for _ in range(n_steps):
            visit_counts[current] += 1
            total_visits += 1
            
            neighbors = list(G.neighbors(current))
            if not neighbors:
                current = np.random.choice(nodes)
                continue
            
            # Weight-biased random walk
            weights = np.array([
                G[current][nbr].get('weight', 1.0) 
                for nbr in neighbors
            ])
            current = np.random.choice(neighbors, p=weights / weights.sum())
    
    centrality = {node: count / total_visits 
                  for node, count in visit_counts.items()}
    return centrality