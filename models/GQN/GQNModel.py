import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.pyjunk.models.Model import Model
from repos.pyjunk.junktools.image import image

from repos.pyjunk.models.GQN.TowerRepresentationNetwork import TowerRepresentationNetwork
from repos.pyjunk.models.GQN.PyramidRepresentationNetwork import PyramidRepresentationNetwork
from repos.pyjunk.models.GQN.GQNCores import *

# Generative Query Network Model

class GQNModel(Model):
    def __init__(self, representation="pyramid", num_layers=12, fSharedCore=False, *args, **kwargs):
        super(GQNModel, self).__init__(*args, **kwargs)

        self.num_layers = num_layers
        self.representation = representation.lower()

        if(self.representation == "pyramid"):
            self.representation_network = PyramidRepresentationNetwork()
        elif(self.representation == "tower"):
            self.representation_network = TowerRepresentationNetwork(fPoolingEnabled=False)
        elif(self.representation == "pool"):
            self.representation_network = TowerRepresentationNetwork(fPoolingEnabled=True)
        else:
            self.representation_network = None
            raise NotImplementedError

        self.fSharedCore = fSharedCore
        if(self.fSharedCore):
            self.inference_core = InferenceCore()
            self.generation_core = GenerationCore()
        else:
            self.inference_core = []
            self.generation_core = []

            for l in range(self.num_layers):
                self.inference_core.append(InferenceCore(id=l))
                self.generation_core.append(GenerationCore(id=l))

            self.inference_core = nn.ModuleList([*self.inference_core])
            self.generation_core = nn.ModuleList([*self.generation_core])

        self.eta_pi = nn.Conv2d(128, 2 * 3, kernel_size=5, stride=1, padding=2)
        self.eta_generation = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        self.eta_inference = nn.Conv2d(128, 3 * 2, kernel_size=5, stride=1, padding=2)

    def loss(self, in_x, in_view, query_view, query_x, gen_sigma):
        in_x = in_x.permute(0, 3, 1, 2)
        query_x = query_x.permute(0, 3, 1, 2)

        in_x = (in_x * 2.0) - 1.0
        query_x = (query_x * 2.0) - 1.0

        B = 1
        M, C, H, W = in_x.size()

        # Representation Network
        if(self.representation == "tower"):
            r = in_x.new_zeros((B, 256, 16, 16))
        else:
            r = in_x.new_zeros((B, 256, 1, 1))

        # Generate the representation of the scene
        #for k in range(C):
        for k in range(M):
            r_k = self.representation_network(in_x[k].unsqueeze(0), in_view[k])
            r += r_k

        print("representation shape: ")
        print(r.shape)

        # Generator Initial State
        cell_state_gen = in_x.new_zeros((B, 128, 16, 16))
        hidden_state_gen = in_x.new_zeros((B, 128, 16, 16))
        u = in_x.new_zeros((B, 128, 64, 64))

        # Inference Initial State
        cell_state_inf = in_x.new_zeros((B, 128, 16, 16))
        hidden_state_inf = in_x.new_zeros((B, 128, 16, 16))

        elbo_loss = 0
        kl_loss = 0
        for l in range(self.num_layers):
            # prior distribution
            mu_pi, log_std_dev_pi = torch.split(self.eta_pi(hidden_state_gen), 3, dim=1)
            std_dev_pi = torch.exp(0.5 * log_std_dev_pi)
            pi_dist = torch.distributions.Normal(mu_pi, std_dev_pi)

            # Update Inference State
            if(self.fSharedCore):
                cell_state_inf, hidden_state_inf = self.inference_core(
                    query_x, query_view, r,
                    cell_state_inf, hidden_state_inf,
                    hidden_state_gen, u
                )
            else:
                cell_state_inf, hidden_state_inf = self.inference_core[l](
                    query_x, query_view, r,
                    cell_state_inf, hidden_state_inf,
                    hidden_state_gen, u
                )

            # Posterior
            mu_query, log_std_dev_query = torch.split(self.eta_inference(hidden_state_inf), 3, dim=1)
            std_dev_query = torch.exp(0.5 * log_std_dev_query)
            query_dist = torch.distributions.Normal(mu_query, std_dev_query)

            # Sample posterior
            z = query_dist.sample()

            # Update Generator State
            if(self.fSharedCore):
                cell_state_gen, hidden_state_gen, u = self.generation_core(
                    query_view, r, cell_state_gen, hidden_state_gen, u, z
                )
            else:
                cell_state_gen, hidden_state_gen, u = self.generation_core[l](
                    query_view, r, cell_state_gen, hidden_state_gen, u, z
                )

            # ELBO kl contribution
            kl_div = torch.distributions.kl.kl_divergence(pi_dist, query_dist)
            elbo_loss -= torch.sum(kl_div, dim=[1, 2, 3])
            kl_loss += torch.sum(kl_div, dim=[1, 2, 3])

        mu_gen = self.eta_generation(u)
        gen_distro = torch.distributions.Normal(mu_gen, gen_sigma)

        ll_loss = gen_distro.log_prob(query_x)
        print("ll_loss: %d" % torch.sum(ll_loss, dim=[1, 2, 3]).item())

        elbo_loss += torch.sum(ll_loss, dim=[1, 2, 3])

        # From paper (section 2 - Optimization)
        elbo_loss = -elbo_loss

        return elbo_loss, kl_loss


    def generate(self, in_x, in_view, query_view, gen_sigma=0):
        in_x = in_x.permute(0, 3, 1, 2)
        in_x = (in_x * 2.0) - 1.0

        B = 1
        M, C, H, W = in_x.size()

        # Scene Encoder
        if (self.representation == "tower"):
            r = in_x.new_zeros((B, 256, 16, 16))
        else:
            r = in_x.new_zeros((B, 256, 1, 1))

            # Generate the representation of the scene
        for k in range(M):
            #r_k = self.representation_network(in_x[:, k], in_view[:, k])
            r_k = self.representation_network(in_x[k].unsqueeze(0), in_view[k])
            r += r_k

        # Initialize state
        cell_state_gen = in_x.new_zeros((B, 128, 16, 16))
        hidden_state_gen = in_x.new_zeros((B, 128, 16, 16))
        u = in_x.new_zeros((B, 128, 64, 64))

        for l in range(self.num_layers):
            # prior
            mu_pi, log_std_dev_pi = torch.split(self.eta_pi(hidden_state_gen), 3, dim=1)
            std_dev_pi = torch.exp(0.5 * log_std_dev_pi)
            pi_distro = torch.distributions.Normal(mu_pi, std_dev_pi)

            # sample prior
            z = pi_distro.sample()

            # Update state
            if(self.fSharedCore):
                cell_state_gen, hidden_state_gen, u = self.generation_core(
                    query_view, r, cell_state_gen, hidden_state_gen, u, z
                )
            else:
                cell_state_gen, hidden_state_gen, u = self.generation_core[l](
                    query_view, r, cell_state_gen, hidden_state_gen, u, z
                )

        # Sample Image
        mu = self.eta_generation(u)

        # Allow for some variability
        # Not sure if this is actually useful or not
        if(gen_sigma == 0):
            x_tilda = torch.clamp(mu, 0, 1)
        else:
            x_tilda = torch.distributions.Normal(mu, gen_sigma)

        x_tilda = x_tilda.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        x_tilda_image = image(torchBuffer=x_tilda)

        return x_tilda_image

    def reconstruct(self, in_x, in_view, query_view, query_x):
        #B, C, H, W = in_x.size()

        in_x = in_x.permute(0, 3, 1, 2)
        in_x = (in_x * 2.0) - 1.0

        B = 1
        M, C, H, W = in_x.size()

        # Scene Encoder
        if (self.representation == "tower"):
            r = in_x.new_zeros((B, 256, 16, 16))
        else:
            r = in_x.new_zeros((B, 256, 1, 1))

            # Generate the representation of the scene
        for k in range(M):
            #r_k = self.representation_network(in_x[:, k], in_view[:, k])
            r_k = self.representation_network(in_x[k].unsqueeze(0), in_view[k])
            r += r_k

        # Initialize inference state
        cell_state_inf = in_x.new_zeros((B, 128, 16, 16))
        hidden_state_inf = in_x.new_zeros((B, 128, 16, 16))

        # Initialize generator state
        cell_state_gen = in_x.new_zeros((B, 128, 16, 16))
        hidden_state_gen = in_x.new_zeros((B, 128, 16, 16))
        u = in_x.new_zeroes((B, 128, 64, 64))

        for l in range(self.num_layers):
            # inference core
            if(self.fSharedCore):
                cell_state_inf, hidden_state_inf = self.inference_core(
                    query_x, query_view, r, cell_state_inf, hidden_state_inf, u
                )
            else:
                cell_state_inf, hidden_state_inf = self.inference_core[l](
                    query_x, query_view, r, cell_state_inf, hidden_state_inf, u
                )

            # Posterior
            mu_query, log_std_dev_query = torch.split(self.eta_inference(hidden_state_inf), 3, dim=1)
            std_dev_query = torch.exp(0.5 * log_std_dev_query)
            query_dist = torch.distributions.Normal(mu_query, std_dev_query)

            # Sample posterior
            z = query_dist.sample()

            # Update state
            if (self.fSharedCore):
                cell_state_gen, hidden_state_gen, u = self.generation_core(
                    query_view, r, cell_state_gen, hidden_state_gen, u, z
                )
            else:
                cell_state_gen, hidden_state_gen, u = self.generation_core[l](
                    query_view, r, cell_state_gen, hidden_state_gen, u, z
                )

        # Sample Image
        mu = self.eta_generation(u)

        # TODO: allow for some variability
        x_tilda = torch.clamp(mu, 0, 1)

        # TODO: allow for some variability
        x_tilda = torch.clamp(mu, 0, 1)

        x_tilda = x_tilda.squeeze().permute(1, 2, 0) * 0.5 + 0.5
        x_tilda_image = image(torchBuffer=x_tilda)

        return x_tilda_image

    #def generate_with_frames(self, in_frames, query_frame):
    def generate_with_frames_view(self, in_frames, torchQueryViewBuffer):
        # Pixel standard-deviation
        sigma_i, sigma_f = 2.0, 0.7
        sigma = sigma_i
        sigma = 0.0

        # TODO: Seems like a utility function - or frameset thing
        # Combine in_frames into one contextBuffer
        torchContextImageBuffer = None
        torchContextViewBuffer = None
        for f in in_frames:
            npFrameBuffer = f.GetNumpyBuffer()
            torchImageBuffer = torch.FloatTensor(npFrameBuffer)
            if (torchContextImageBuffer == None):
                torchContextImageBuffer = torchImageBuffer.unsqueeze(0)
            else:
                torchContextImageBuffer = torch.cat((torchContextImageBuffer, torchImageBuffer.unsqueeze(0)), dim=0)

            frame_view = f.GetFrameCameraView()
            if (torchContextViewBuffer == None):
                torchContextViewBuffer = frame_view.unsqueeze(0)
            else:
                torchContextViewBuffer = torch.cat((torchContextViewBuffer, frame_view.unsqueeze(0)), dim=0)

        # # Retrieve Query frame view
        # torchQueryViewBuffer = query_frame.GetFrameCameraView().unsqueeze(0)

        # Run the model to generate an image

        gen_image = self.generate(
            in_x=torchContextImageBuffer,
            in_view=torchContextViewBuffer,
            query_view=torchQueryViewBuffer,
            gen_sigma=sigma
        )

        # return the image
        return gen_image

    def generate_with_frames(self, in_frames, query_frame):
        # Retrieve Query frame view
        torchQueryViewBuffer = query_frame.GetFrameCameraView().unsqueeze(0)
        print(torchQueryViewBuffer.dtype)

        return self.generate_with_frames_view(in_frames, torchQueryViewBuffer)

    def loss_with_frames(self, in_frames, query_frame, sigma):
        # TODO: Seems like a utility function - or frameset thing
        # Combine in_frames into one contextBuffer
        torchContextImageBuffer = None
        torchContextViewBuffer = None
        for f in in_frames:
            npFrameBuffer = f.GetNumpyBuffer()
            torchImageBuffer = torch.FloatTensor(npFrameBuffer)
            if(torchContextImageBuffer == None):
                torchContextImageBuffer = torchImageBuffer.unsqueeze(0)
            else:
                torchContextImageBuffer = torch.cat((torchContextImageBuffer, torchImageBuffer.unsqueeze(0)), dim=0)

            frame_view = f.GetFrameCameraView()
            if(torchContextViewBuffer == None):
                torchContextViewBuffer = frame_view.unsqueeze(0)
            else:
                torchContextViewBuffer = torch.cat((torchContextViewBuffer, frame_view.unsqueeze(0)), dim=0)


        # print(torchContextImageBuffer.shape)
        # print(torchContextViewBuffer)
        print(torchContextViewBuffer.shape)

        # Retrieve Query Image Buffer
        npFrameBuffer = query_frame.GetNumpyBuffer()
        torchImageBuffer = torch.FloatTensor(npFrameBuffer)
        torchQueryImageBuffer = torchImageBuffer.unsqueeze(0)
        torchQueryViewBuffer = query_frame.GetFrameCameraView().unsqueeze(0)

        # print(torchQueryImageBuffer.shape)
        # print(torchQueryViewBuffer)
        print(torchQueryViewBuffer.shape)

        # Run the model
        torchLoss = self.loss(
            in_x = torchContextImageBuffer,
            in_view = torchContextViewBuffer,
            query_x = torchQueryImageBuffer,
            query_view = torchQueryViewBuffer,
            gen_sigma = sigma
        )

        # return an image
        return torchLoss
