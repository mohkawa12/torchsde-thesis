'''
Date:

A script to store some model definitions.
'''
import torch

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'
if not is_cuda:
    print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

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
    Ax = torch.zeros(2*N, 2*N, device=device)
    for iPop in range(0,N):
        iA = iPop*2
        submatrix = torch.tensor([[0.0, 1.0],[-p[iPop]**2, 0.0]])
        Ax[iA:iA+2, iA:iA+2] = submatrix
    dx = Ax@x


    noParams = N*(N-1)
    # Each population is a second order linear harmonic osc, take the first state from each population
    x_state = x[0::2,:]
    if device=='cpu':
        # This was written first but is not GPU-friendly
        # Calculate the adjacency matrix
        A = getAdjacencyOOP(p=p[weight_start_idx:weight_start_idx+noParams], N=N)

        # calculate the (inwards) degree diagonal
        Delta = torch.sum(A, dim=1)

        # Calculate the influence of coupling on the derivative
        dx_couple = -A@x_state + (x_state.transpose(0,1)*Delta).transpose(0,1)
    else:
        # This is rewritten to be GPU friendly
        x_state = x[0::2,:]
        dx_couple = getCouplingComponentGPU(p[weight_start_idx:weight_start_idx+noParams], x_state, N)

    # Only the second order term is influenced by the coupling
    dx[1::2,:] -= dx_couple

    return dx.transpose(0, 1)


# GPU efficient adjacency and degree diagonal component calculation
# When using GPU it is more efficient to compute the contribution to the derivative
# directly rather than building an adjacency matrix, to avoid individually setting matrix elements
def getCouplingComponentGPU(p, x, N):
    noParams = N-1
    dx_Adj = torch.zeros(x.shape, device=device)
    dx_Delta = torch.zeros(x.shape, device=device)
    for iPop in range(0, N):
        # the iPopth component of dx from the Adjacency matrix are the parameters (already formatted in the correct order)
        # vector multiplied by the adjacent states x (i.e. not including the iPopth state)
        iStart = iPop*noParams
        # Masking is more GPU efficient than building the adj matrix w/ 0 on the diagonal (as in getAdjacencyIP)
        indices = torch.cat((torch.arange(iPop), torch.arange(iPop+1, N)))
        x_masked = x[indices, :]
        dx_Adj[iPop,:] = -p[iStart:iStart+noParams]@x_masked
        # The Delta component is the iPopth state contribution
        dx_Delta[iPop,:] = torch.sum(p[iStart:iStart+noParams])*x[iPop,:]

    return dx_Adj + dx_Delta

# Calculate the adjacency matrix
def getAdjacencyOOP(p, N):
    A = torch.zeros(N,N).to(device)
    A = getAdjacencyIP(A, p, N)
    return A

# TODO I am not completely convinced that the indices here are correct?
# Calculate the adjacency matrix
def getAdjacencyIP(A, p, N):
    noParams = N-1
    row = torch.cat((torch.zeros(1).to(device), p[0:noParams]))
    A[0,:] = row
    for iPop in range(1,N):
        iStart = (iPop-1)*noParams
        row = torch.cat((p[iStart:iStart+iPop-2], torch.zeros(1).to(device), p[iStart+iPop-1:iStart+noParams-1]))
        A[iPop,:] = row 
    row = torch.cat((p[-noParams:], torch.zeros(1).to(device)))
    A[N-1,:] = row
    return A
