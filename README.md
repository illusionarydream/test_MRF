### MRF
In this project, I use MRF method to denoise.
#### MRF Introduction
- MRF(Markov Random Field)
- Inner potential function: $\phi(x_i)$ is defined on node $x_i$, denotes inner constriants.
- Outer potential function: $\psi(x_i^{(k)},x_j)$ is defined on edge $<x_i,x_j>$, denotes the $x_j$ 's belief on the $k$ label of $x_i$.
- Massage function: $m_{i\rightarrow j}$ is defined on edge $<x_i,x_j>$, denotes the massage stream.
- Margin function: $b(x_i)$ denotes the final optimal label of $x_i$.
#### Optimize Process
- Update massage function: $m_{i \to j}(x_j) = \sum_{x_i} \psi_{i,j}(x_i, x_j) \cdot \psi_i(x_i) \cdot \prod_{k \in \text{neigh}(i) \setminus j} m_{k \to i}(x_i)$
- Calculate margin function: $b(x_i) \propto \psi_i(x_i) \cdot \prod_{k \in \text{neigh}(i)} m_{k \to i}(x_i)$ , which need to be normalized.
