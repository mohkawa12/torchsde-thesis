'''
Date:

A script to store some model definitions.
'''
import torch
# This is the linear oscillator model from the julia project
# Fully connected N population couple linear harmonic oscillator
# See chapter 2 of Mesbahi graph theory book for the coupling
def LindxCouple(x, p, t):
    # x is shape (batch_size, xdim)
    # Convert to the easier to use (xdim, batch_size)
    x = x.transpose(0, 1)
    xdim = x.shape[0]
    N = int(xdim/2)
    weight_start_idx = N

    # Calculate the derivatives without coupling (isolated harmonic oscillators)
    Ax = torch.zeros(2*N, 2*N)
    for iPop in range(0,N):
        iA = iPop*2
        submatrix = torch.tensor([[0.0, 1.0],[-p[iPop]**2, 0.0]])
        Ax[iA:iA+2, iA:iA+2] = submatrix
    dx = Ax@x


    # Calculate the adjacency matrix
    noParams = N*(N-1)
    A = getAdjacencyOOP(p=p[weight_start_idx:weight_start_idx+noParams], N=N)

    # calculate the (inwards) degree diagonal
    Delta = torch.sum(A, dim=1)

    # Calculate the influence of coupling on the derivative
    # Each population is a second order linear harmonic osc, take the first state from each population
    x_state = x[0::2,:]
    dx_couple = -A@x_state + (x_state.transpose(0,1)*Delta).transpose(0,1)

    # Only the second order term is influenced by the coupling
    dx[1::2,:] -= dx_couple

    return dx.transpose(0, 1)

# Calculate the adjacency matrix
def getAdjacencyOOP(p, N):
    A = torch.zeros(N,N)
    A = getAdjacencyIP(A, p, N)
    return A

# Calculate the adjacency matrix
def getAdjacencyIP(A, p, N):
    noParams = N-1
    row = torch.cat((torch.zeros(1), p[0:noParams]))
    A[0,:] = row
    for iPop in range(1,N):
        iStart = (iPop-1)*noParams
        row = torch.cat((p[iStart:iStart+iPop-2], torch.zeros(1), p[iStart+iPop-1:iStart+noParams-1]))
        A[iPop,:] = row 
    row = torch.cat((p[-noParams:], torch.zeros(1)))
    A[N-1,:] = row
    return A
