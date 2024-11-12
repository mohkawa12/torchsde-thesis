from thesis import models as mod
import torch
import torchsde
import matplotlib.pyplot as plt

# From sde_gan.py
class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

# From sde_gan.py
class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 LipSwish()]

        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

class LindxModel(torch.nn.Module):
    def __init__(self, p0):
        super().__init__()
        self.p = torch.nn.Parameter(p0)

    def forward(self, x):
        return mod.LindxCouple(x, self.p, 0)

class Func(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, xdim):
        super().__init__()
        self._noise_size = 1
        self._hidden_size = 4
        p0 = torch.tensor([62.0, 31.0, 0.0, 0.0])
        self._drift = LindxModel(p0)
        self._diffusion = MLP(self._hidden_size, self._hidden_size*self._noise_size, 4, 2, tanh=True)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        # t = t.expand(x.size(0), 1)
        # tx = torch.cat([t, x], dim=1)
        return self._drift(x), self._diffusion(x).view(x.size(0), self._hidden_size, self._noise_size)

ts = torch.linspace(0, 1, 500)
xdim = 4
func = Func(xdim)
x0 = torch.ones(1, xdim)
xs = torchsde.sdeint_adjoint(func, x0, ts, method='reversible_heun', dt=1/300,
                                adjoint_method='adjoint_reversible_heun',)
                                
with torch.no_grad():
    plt.plot(ts, xs[:,0,:])
    plt.show()