# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train an SDE as a GAN, on data from a time-dependent Ornstein--Uhlenbeck process.

Training SDEs as GANs was introduced in "Neural SDEs as Infinite-Dimensional GANs".
https://arxiv.org/abs/2102.03657

This reproduces the toy example in Section 4.1 of that paper.

This additionally uses the improvements introduced in "Efficient and Accurate Gradients for Neural SDEs".
https://arxiv.org/abs/2105.13493

To run this file, first run the following to install extra requirements:
pip install fire
pip install git+https://github.com/patrick-kidger/torchcde.git

To run, execute:
python -m examples.sde_gan
"""
import fire
import matplotlib.pyplot as plt
import torch
import torch.optim.swa_utils as swa_utils
import torchcde
import torchsde
import tqdm

from thesis import models as mod
import scipy.io as sio
import torch.autograd.profiler as profiler


###################
# First some standard helper objects.
###################

class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


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

# Linear harmonic oscillator
class LindxModel(torch.nn.Module):
    def __init__(self, p0):
        super().__init__()
        self.p = torch.nn.Parameter(p0)

    def forward(self, x):
        return mod.LindxCouple(x, self.p, 0)

class Drift(torch.nn.Module):
    def __init__(self, hidden_size, mlp_size, num_layers, p0, tanh=True):
        super().__init__()
        self._nn = MLP(hidden_size, hidden_size, mlp_size, num_layers, tanh=tanh)
        self._osc = LindxModel(p0)

    def forward(self, x):
        return self._osc.forward(x) + self._nn.forward(x)

# NN module to select only odd elements of input
class OddSel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x has shape (batch_size, time, hidden_size)
        state_dim = x.shape[1]
        return x[:,:,0:state_dim:2]

# NN module to scale the input
class ScaleLinear(torch.nn.Module):
    def __init__(self, init_scale):
        super().__init__()
        self.scale = torch.nn.Parameter(init_scale)

    def forward(self, x):
        return self.scale * x

###################
# Now we define the SDEs.
#
# We begin by defining the generator SDE.
###################
class GeneratorFunc(torch.nn.Module):
    sde_type = 'stratonovich'
    # noise_type = 'general'
    noise_type = 'diagonal'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers, p0):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        # If you have problems with very high drift/diffusions then consider scaling these so that they squash to e.g.
        # [-3, 3] rather than [-1, 1].
        ###################
        self._drift = Drift(hidden_size, mlp_size, num_layers, p0, tanh=True)
        # self._diffusion = MLP(hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)
        # Try a diagonal scale for noise
        scale = torch.tensor([1.0, 1.0, 1.0, 1.0])
        self._diffusion = ScaleLinear(scale)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        # No time dependence of drift and diffucion
        # return self._drift(x), self._diffusion(x).view(x.size(0), self._hidden_size, self._noise_size)
        return self._drift(x), self._diffusion(x)


###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers, p0):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers, p0)
        # self._readout = torch.nn.Identity()
        # self._readout = torch.nn.Linear(hidden_size, data_size)
        self._readout = OddSel()

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1/300,
                                     adjoint_method='adjoint_reversible_heun',)
        # xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun',
                                    #  adjoint_method='adjoint_reversible_heun',)
        # xs = torchsde.sdeint_adjoint(self._func, x0, ts)
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs as Infinite-Dimensional GANs" paper. (You could use other things as well,
# but this is a natural choice.)
#
# There's actually a few different (roughly equivalent) ways of making the discriminator work. The curious reader is
# encouraged to have a read of the comment at the bottom of this file for an in-depth explanation.
###################
class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, ys_coeffs):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        # hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', backend='torchsde', dt=1.0,
                            #  adjoint_method='adjoint_reversible_heun',
                            #  adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, backend='torchsde', dt=1.0,
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()


###################
# Generate some data. For this example we generate some synthetic data from a time-dependent Ornstein-Uhlenbeck SDE.
###################
def get_data(batch_size, filename, device):
    matvar_name = "region_sel"
    matstruct = sio.loadmat(filename, squeeze_me=True)

    for iCond in range(len(matstruct[matvar_name])):
        # TODO ignore the distractor condition for now
        if iCond==1:
            continue

        ts = torch.tensor(matstruct[matvar_name][iCond]['time'].item()).float().to(device)
        ts = ts - min(ts)

        t_size = len(ts)
        data = torch.tensor(matstruct[matvar_name][iCond]['mom'].item()).float().to(device)
        # (iRegion, iTrial, iTime)
        # Convert to (iTime, iTrial, iRegion) to match the expected format from torchsde.sdeint
        ys = torch.permute(data, (2, 1, 0))
        # TODO take only v1 and l-ips
        ys = ys[:,:,1:3]
        dataset_size = ys.size(dim=1)


    ###################
    # Typically important to normalise data. Note that the data is normalised with respect to the statistics of the
    # initial data, _not_ the whole time series. This seems to help the learning process, presumably because if the
    # initial condition is wrong then it's pretty hard to learn the rest of the SDE correctly.
    ###################
    y0_flat = ys[0].reshape(-1)
    y0_not_nan = y0_flat.masked_select(~torch.isnan(y0_flat))
    ys = (ys - y0_not_nan.mean()) / y0_not_nan.std()

    ###################
    # As discussed, time must be included as a channel for the discriminator.
    ###################
    ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1),
                    ys.transpose(0, 1)], dim=2)
    # print(ys.shape)
    # shape (dataset_size=1000, t_size=100, 1 + data_size=3)

    ###################
    # Package up.
    ###################
    data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
    ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
    dataset = torch.utils.data.TensorDataset(ys_coeffs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Check that the linear interpolation is working
    # Ys = torchcde.LinearInterpolation(ys_coeffs)
    # ys_recon = Ys.evaluate(Ys._t)
    # testfig, testax = plt.subplots()
    # testax.plot(ts.cpu(), ys[1,:,1].cpu())
    # testax.plot(ts.cpu(), ys_coeffs[1,:,1].cpu(), linestyle="dashed")
    # testax.plot(ts.cpu(), ys_recon[1,:,1].cpu())
    # plt.show()

    return ts, data_size, dataloader


###################
# We'll plot some results at the end.
###################
def plot(ts, generator, dataloader, num_plot_samples, plot_locs, ):
    # Get samples
    real_samples, = next(iter(dataloader))
    assert num_plot_samples <= real_samples.size(0)
    # Linear interpolation is not necessary for the real data, it is sampled at 300 Hz
    # real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    # Only plot the first variable
    real_samples = real_samples[..., 1]

    with torch.no_grad():
        generated_samples = generator(ts, real_samples.size(0)).cpu()
    # generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
    generated_samples = generated_samples[..., 1]

    # # Plot histograms
    # for prop in plot_locs:
    #     time = int(prop * (real_samples.size(1) - 1))
    #     real_samples_time = real_samples[:, time]
    #     generated_samples_time = generated_samples[:, time]
    #     _, bins, _ = plt.hist(real_samples_time.cpu().numpy(), bins=32, alpha=0.7, label='Real', color='dodgerblue',
    #                           density=True)
    #     bin_width = bins[1] - bins[0]
    #     num_bins = int((generated_samples_time.max() - generated_samples_time.min()).item() // bin_width)
    #     plt.hist(generated_samples_time.cpu().numpy(), bins=num_bins, alpha=0.7, label='Generated', color='crimson',
    #              density=True)
    #     plt.legend()
    #     plt.xlabel('Value')
    #     plt.ylabel('Density')
    #     plt.title(f'Marginal distribution at time {time}.')
    #     plt.tight_layout()
    #     plt.show()

    real_samples = real_samples[:num_plot_samples]
    generated_samples = generated_samples[:num_plot_samples]

    # Plot samples
    real_first = True
    generated_first = True
    fig1, (ax1, ax2) = plt.subplots(2,1)
    for real_sample_ in real_samples:
        kwargs = {'label': 'Real'} if real_first else {}
        ax1.plot(ts.cpu(), real_sample_.cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        real_first = False
    for generated_sample_ in generated_samples:
        kwargs = {'label': 'Generated'} if generated_first else {}
        ax2.plot(ts.cpu(), generated_sample_.cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
        generated_first = False
    ax1.legend()
    ax2.legend()
    ax1.title.set_text(f"{num_plot_samples} samples from both real and generated distributions.")
    plt.tight_layout()
    return fig1


###################
# Now do normal GAN training, and plot the results.
#
# GANs are famously tricky and SDEs trained as GANs are no exception. Hopefully you can learn from our experience and
# get these working faster than we did -- we found that several tricks were often helpful to get this working in a
# reasonable fashion:
# - Stochastic weight averaging (average out the oscillations in GAN training).
# - Weight decay (reduce the oscillations in GAN training).
# - Final tanh nonlinearities in the architectures of the vector fields, as above. (To avoid the model blowing up.)
# - Adadelta (interestingly seems to be a lot better than either SGD or Adam).
# - Choosing a good learning rate (always important).
# - Scaling the weights at initialisation to be roughly the right size (chosen through empirical trial-and-error).
###################

def evaluate_loss(ts, batch_size, dataloader, generator, discriminator):
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples

def print_linosc_params(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name=="_func._drift._osc.p":
                print("Harmonic oscillator trained params: ", param.data)

def plot_linosc_params(ax, iter, model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name=="_func._drift._osc.p":
                params = param.cpu().data[2:]
                ax.scatter(iter*torch.ones(params.shape), params, c=['g', 'b'])

def main(
        # Architectural hyperparameters. These are quite small for illustrative purposes.
        initial_noise_size=4,  # How many noise dimensions to sample at the start of the SDE.
        noise_size=4,          # How many dimensions the Brownian motion has.
        hidden_size=4,        # How big the hidden size of the generator SDE and the discriminator CDE are.
        mlp_size=16,           # How big the layers in the various MLPs are.
        num_layers=3,          # How many hidden layers to have in the various MLPs.

        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        generator_lr=1e-1,      # Learning rate often needs careful tuning to the problem. (1e-1 was good)
        discriminator_lr=1e-2,  # Learning rate often needs careful tuning to the problem. (1e-1 was good)
        batch_size=700,        # Batch size.# TODO originally 1024
        steps=20,            # How many steps to train generator for.
        disc_steps = 15,     # How many steps to train the discriminator for (per generator step)
        init_mult1=4,           # Changing the initial parameter size can help.
        init_mult2=0.5,         #
        weight_decay=0.01,      # Weight decay.
        swa_step_start=1,    # When to start using stochastic weight averaging.

        # Evaluation and plotting hyperparameters
        steps_per_print=1,                   # How often to print the loss.
        steps_per_plot=5,                    # how often to save a plot.
        num_plot_samples=5,                  # How many samples to use on the plots at the end.
        plot_locs=(0.1, 0.3, 0.5, 0.7, 0.9),  # Plot some marginal distributions at this proportion of the way along.
):
    # TODO loss print statement is a bit misleading, as the loss printed at the end of the 
    # training cycle is from before the generator update
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

    # Data
    filename = "sub703_region_30Hzbp.mat"
    ts, data_size, train_dataloader = get_data(batch_size=batch_size, filename=filename, device=device)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    # Models
    p0 = torch.tensor([62.0, 62.0, 0.1, 0.1])
    generator = Generator(data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers, p0).to(device)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers).to(device)
    # Weight averaging really helps with GAN training.
    # averaged_generator = swa_utils.AveragedModel(generator)
    # averaged_discriminator = swa_utils.AveragedModel(discriminator)

    # Picking a good initialisation is important!
    # In this case these were picked by making the parameters for the t=0 part of the generator be roughly the right
    # size that the untrained t=0 distribution has a similar variance to the t=0 data distribution.
    # Then the func parameters were adjusted so that the t>0 distribution looked like it had about the right variance.
    # What we're doing here is very crude -- one can definitely imagine smarter ways of doing things.
    # (e.g. pretraining the t=0 distribution)
    with torch.no_grad():
        for param in generator._initial.parameters():
            param *= init_mult1
        for name, param in generator._func.named_parameters():
            if name=="_drift._osc.p":
            # Don't change the oscillator parameters, these are initialsed intentionally
                continue
            param *= init_mult2
    ###################
    # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
    # LipSwish activation functions).
    ###################
    with torch.no_grad():
        for module in discriminator.modules():
            if isinstance(module, torch.nn.Linear):
                lim = 1 / module.out_features
                module.weight.clamp_(-lim, lim)

    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    # generator_optimiser = torch.optim.Adadelta(generator.parameters(), lr=generator_lr, weight_decay=weight_decay)
    generator_optimiser = torch.optim.Adam(generator.parameters(), lr=generator_lr, weight_decay=weight_decay)
    # Discriminator should maximise the function
    # discriminator_optimiser = torch.optim.Adadelta(discriminator.parameters(), lr=discriminator_lr,
                                                #    weight_decay=weight_decay, maximize=True)
    discriminator_optimiser = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr,
                                                   weight_decay=weight_decay, maximize=True)
    # Train both generator and discriminator.
    fig4, ax4 = plt.subplots()
    ax4.title.set_text("Linear Oscillator Coupling over Iterations")
    losses_gen = torch.zeros(steps*(disc_steps+1))
    losses_disc = torch.zeros(steps*(disc_steps+1))
    score_real = torch.zeros(steps*(disc_steps+1))
    trange = tqdm.tqdm(range(steps))
    for step in trange:
        # with profiler.profile(use_cuda=True) as prof:

        # Train the discriminator (possibly multiple times)
        print("Training Discriminator...")
        real_samples, = next(infinite_train_dataloader)
        generated_samples = generator(ts, batch_size).detach()
        for disc_step in range(0, disc_steps):
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            loss.backward()
            discriminator_optimiser.step()
            # set_to_none apparently requires fewer memory operations: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            discriminator_optimiser.zero_grad(set_to_none=True)
            total_iter = step*(disc_steps+1)+disc_step
            with torch.no_grad():
                losses_disc[total_iter] = loss.item()
                losses_gen[total_iter] = generated_score.item()
                score_real[total_iter] = real_score.item()
                print("Loss: ", loss.item())
            plot_linosc_params(ax4, total_iter, generator)
            ###################
            # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
            # LipSwish activation functions).
            ###################
            with torch.no_grad():
                for module in discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features
                        module.weight.clamp_(-lim, lim)
            # print(prof.key_averages().table(sort_by="cuda_time_total"))

        # Train the generator
        print("Training Generator...")
        generated_samples = generator(ts, batch_size)
        generated_score = discriminator(generated_samples)
        loss = generated_score
        total_iter = step*(disc_steps+1)+disc_steps
        with torch.no_grad():
            record_loss = generated_score.cpu() - real_score.cpu()
            losses_disc[total_iter] = record_loss
            losses_gen[total_iter] = loss.item()
            score_real[total_iter] = real_score.item()
            print(f"Step: {step} Loss: {record_loss:.15f}")
        plot_linosc_params(ax4, total_iter, generator)
        loss.backward()

        # Moved this to discriminator to be more consistent with the literature, practically it makes no difference
        # for param in generator.parameters():
            # param.grad *= -1
        # Update the parameters
        generator_optimiser.step()
        # set_to_none apparently requires fewer memory operations: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        generator_optimiser.zero_grad(set_to_none=True)

        print("Parameters updated, applying stochastic weight averaging (if requested)...")
        # Stochastic weight averaging typically improves performance.
        # if step > swa_step_start:
        #     averaged_generator.update_parameters(generator)
        #     averaged_discriminator.update_parameters(discriminator)

        # if (step % steps_per_print) == 0 or step == steps - 1:
        #     # TODO the evaluate loss takes forever!
        #     # total_unaveraged_loss = evaluate_loss(ts, batch_size, train_dataloader, generator, discriminator)
        #     if step > swa_step_start:
        #         # total_averaged_loss = evaluate_loss(ts, batch_size, train_dataloader, averaged_generator.module,
        #                                             # averaged_discriminator.module)
        #         # trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
        #                     #  f"Loss (averaged): {total_averaged_loss:.4f}")
        #         trange.write(f"Step: {step:3} Loss (averaged): {record_loss:.4f} ")
        #     else:
        #         trange.write(f"Step: {step:3} Loss (unaveraged): {record_loss:.4f}")
        if steps_per_plot>0 and (step % steps_per_plot) == 0:
            _, _, test_dataloader = get_data(batch_size=batch_size, filename=filename, device=device)
            fig = plot(ts, generator, test_dataloader, num_plot_samples, plot_locs)
            title = f"output-{step}.png"
            fig.savefig(title)
            
        print_linosc_params(generator)

    # generator.load_state_dict(averaged_generator.module.state_dict())
    # discriminator.load_state_dict(averaged_discriminator.module.state_dict())


    # Print the harmonic oscillator parameters
    print_linosc_params(generator)
    _, _, test_dataloader = get_data(batch_size=batch_size, filename=filename, device=device)

    fig = plot(ts, generator, test_dataloader, num_plot_samples, plot_locs)
    title = "finaloutput.png"
    fig.savefig(title)
    title = "parameters.png"
    fig4.savefig(title)
    with torch.no_grad():
        # Generator and discriminator losses
        fig2, ax2 = plt.subplots()
        ax2.plot(losses_disc, label='disc')
        ax2.plot(losses_gen, label='gen')
        ax2.legend()
        ax2.title.set_text(f"Loss over iterations")
        fig2.savefig("losses.png")
        
        # Real and fake discriminator scores
        fig3, ax3 = plt.subplots()
        ax3.plot(losses_gen, label='fake' )
        ax3.plot(score_real, label='real' )
        ax3.legend()
        ax3.title.set_text("Discriminator score over iterations")
        fig3.savefig("scores.png")
    # plt.show() # too many plots to output


if __name__ == '__main__':
    fire.Fire(main)

###################
# And that's (one way of doing) an SDE as a GAN. Have fun.
###################

###################
# Appendix: discriminators for a neural SDE
#
# This is a little long, but should all be quite straightforward. By the end of this you should have a comprehensive
# knowledge of how these things fit together.
#
# Let Y be the real/generated sample, and let H be the hidden state of the discriminator.
# For real data, then Y is some interpolation of an (irregular) time series. (As with neural CDEs, if you're familiar -
# for a nice exposition on this see https://github.com/patrick-kidger/torchcde/blob/master/example/irregular_data.py.)
# In the case of generated data, then Y is _either_ the continuous-time sample produced by sdeint, _or_ it is an
# interpolation (probably linear interpolation) of the generated sample between particular evaluation points, We'll
# refer to these as cases (*) and (**) respectively.
#
# In terms of the mathematics, our options for the discriminator are:
# (a1) Solve dH(t) = f(t, H(t)) dt + g(t, H(t)) dY(t),
# (a2) Solve dH(t) = (f, g)(t, H(t)) d(t, Y(t))
# (b) Solve dH(t) = f(t, H(t), Y(t)) dt.
# Option (a1) is what is stated in the paper "Neural SDE as Infinite-Dimensional GANs".
# Option (a2) is theoretically the same as (a1), but the drift and diffusion have been merged into a single function,
# and the sample Y has been augmented with time. This can sometimes be a more helpful way to think about things.
# Option (b) is a special case of the first two, by Appendix C of arXiv:2005.08926.
# [Note that just dH(t) = g(t, H(t)) dY(t) would _not_ be enough, by what's known as the tree-like equivalence property.
#  It's a bit technical, but the basic idea is that the discriminator wouldn't be able to tell how fast we traverse Y.
#  This is a really easy mistake to make; make sure you don't fall into it.]
#
# Whether we use (*) or (**), and (a1) or (a2) or (b), doesn't really affect the quality of the discriminator, as far as
# we know. However, these distinctions do affect how we solve them in terms of code. Depending on each combination, our
# options are to use a solver of the following types:
#
#      | (a1)   (a2)   (b)
# -----+----------------------
#  (*) | SDE           SDE
# (**) |        CDE    ODE
#
# So, (*) implies using an SDE solver: the continuous-time sample is only really available inside sdeint, so if we're
# going to use the continuous-time sample then we need to solve generator and discriminator together inside a single SDE
# solve. In this case, as our generator takes the form
# Y(t) = l(X(t)) with dX(t) = μ(t, X(t)) dt + σ(t, X(t)) dW(t),
# then
# dY(t) = l(X(t)) dX(t) = l(X(t))μ(t, X(t)) dt + l(X(t))σ(t, X(t)) dW(t).
# Then for (a1) we get
# dH(t) = ( f(t, H(t)) + g(t, H(t))l(X(t))μ(t, X(t)) ) dt + g(t, H(t))l(X(t))σ(t, X(t)) dW(t),
# which we can now put together into one big SDE solve:
#  ( X(t) )   ( μ(t, X(t)                                )      ( σ(t, X(t))                  )
# d( Y(t) ) = ( l(X(t))μ(t, X(t)                         ) dt + ( l(X(t))σ(t, X(t))           ) dW(t)
#  ( H(t) )   ( f(t, H(t)) + g(t, H(t))l(X(t))μ(t, X(t)) )      ( g(t, H(t))l(X(t))σ(t, X(t)) ),
# whilst for (b) we can put things together into one big SDE solve:
#  ( X(t) )   ( μ(t, X(t))       )      ( σ(t, X(t))        )
# d( Y(t) ) = ( l(X(t))μ(t, X(t) ) dt + ( l(X(t))σ(t, X(t)) ) dW(t)
#  ( H(t) )   ( f(t, H(t), Y(t)) )      ( 0                 )
#
# Phew, what a lot of stuff to write down. Don't be put off by this: there's no complicated algebra, it's literally just
# substituting one equation into another. Also, note that all of this is for the _generated_ data. If using real data,
# then Y(t) is as previously described always an interpolation of the data. If you're able to evaluate the derivative of
# the interpolation then you can then apply (a1) by rewriting it as dY(t) = (dY/dt)(t) dt and substituting in. If you're
# able to evaluate the interpolation itself then you can apply (b) directly.
#
# The benefit of using (*) is that everything can be done inside a single SDE solve, which is important if you're
# thinking about using adjoint methods and the like, for memory efficiency. The downside is that the code gets a bit
# more complicated: you need to be able to solve just the generator on its own (to produce samples at inference time),
# just the discriminator on its own (to evaluate the discriminator on the real data), and the combined
# generator-discriminator system (to evaluate the discriminator on the generated data).
#
# Right, let's move on to (**). In comparison, this is much simpler. We don't need to substitute in anything. We're just
# taking our generated data, sampling it at a bunch of points, and then doing some kind of interpolation (probably
# linear interpolation). Then we either solve (a2) directly with a CDE solver (regardless of whether we're using real or
# generated data), or solve (b) directly with an ODE solver (regardless of whether we're using real or generated data).
#
# The benefit of this is that it's much simpler to code: unlike (*) we can separate the generator and discriminator, and
# don't ever need to combine them. Also, real and generated data is treated the same in the discriminator. (Which is
# arguably a good thing anyway.) The downside is that we can't really take advantage of things like adjoint methods to
# backpropagate efficiently through the generator, because we need to produce (and thus store) our generated sample at
# lots of time points, which reduces the memory efficiency.
#
# Note that the use of ODE solvers for (**) is only valid because we're using _interpolated_ real or generated data,
# and we're assuming that we're using some kind of interpolation that is at least piecewise smooth. (For example, linear
# interpolation is piecewise smooth.) It wouldn't make sense to apply ODE solvers to some rough signal like Brownian
# motion - that's what case (*) and SDE solvers are about.
#
# Right, let's wrap up this wall of text. Here, we use option (**), (a2). This is arguably the simplest option, and
# is chosen as we'd like to keep the code readable in this example. To solve the CDEs we use the CDE solvers available
# through torchcde: https://github.com/patrick-kidger/torchcde.
###################
