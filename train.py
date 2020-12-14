import os
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
from IPython import display
import pickle
from skimage import img_as_ubyte
import time
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.estimators import HillClimbSearch

from models import Generator, Discriminator
from evaluation import *
from visualization import *


class Trainer():
    def __init__(self, mode, network, bayesmodel, layout, encoder, nfeatures, expname, generator, discriminator, gen_optimizer, dis_optimizer, gen_scheduler,
                 gp_weight=10, critic_iterations=5, print_every=50, use_cuda=torch.cuda.is_available()):
        assert mode in ['W', 'JS', 'SH', 'KL', 'TV'], 'Invalid mode.'
        self.mode = mode
        self.network = network
        self.bayesmodel = bayesmodel
        self.layout = layout
        self.encoder = encoder
        self.nfeatures = nfeatures
        self.feature_idxs = np.split(np.arange(nfeatures.sum()), np.cumsum(nfeatures)[:-1])
        self.expname = expname
        self.G = generator
        self.G_opt = gen_optimizer
        self.G_sch = gen_scheduler
        self.G_state_dict = None
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'gradient_norm': [], 'd_real': [], 'd_generated': [], 'energy_statistics': [], 'relative_structure_score': [], 
                       'structure_prediction_iteration': [], 'structure_prediction_accuracy': [], 'bayesmodel_predicted': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.pca = PCA(n_components=2)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        self.prev_num_steps = 0
        self.max_bic_score = -np.inf
        self.pre_generated_data = None
        self.pre_G_state_dict = None

    def _critic_train_iteration(self, data, latent_samples, tau, pretrain=False):
        # Get generated data
        generated_data = self.G(latent_samples, tau).detach()

        # Calculate probabilities on real and generated data
        d_real, d_generated = 0, 0
        for idx, local in enumerate(self.network):
            local_feature_idxs = np.arange(self.nfeatures.sum())[np.concatenate([self.feature_idxs[node] for node in local])]
            d_real += self.D[idx](data[:,local_feature_idxs])
            d_generated += self.D[idx](generated_data[:,local_feature_idxs])
            
        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = self._fdiv_activation(d_real, d_generated)
        
        # Record loss
        if not pretrain:
            self.losses['d_real'].append(d_real.mean().data.item())
            self.losses['d_generated'].append(d_generated.mean().data.item())
            self.losses['D'].append(-d_loss.data.item())        
        
        if self.mode == 'W':
            # Get gradient penalty
            gradient_penalty = self._gradient_penalty(data, generated_data)
            d_loss += gradient_penalty
            
        # Optimization
        d_loss.backward()
        self.D_opt.step()

    def _generator_train_iteration(self, data, latent_samples, tau):
        # Get generated data
        generated_data = self.G(latent_samples, tau)

        # Calculate loss and optimize
        d_generated = 0
        for idx, local in enumerate(self.network):
            local_feature_idxs = np.arange(self.nfeatures.sum())[np.concatenate([self.feature_idxs[node] for node in local])]
            d_generated += self.D[idx](generated_data[:,local_feature_idxs])
        
        self.G_opt.zero_grad()
        g_loss = -self._fdiv_activation(torch.zeros_like(d_generated), d_generated)
        
        # Record loss
        self.losses['G'].append(g_loss.data.item())
        
        # Optimization
        g_loss.backward()
        self.G_opt.step()
        
    def _fdiv_activation(self, d_real, d_generated):
        # F-divergence Functions
        def gf(v):
            if self.mode == 'SH':
                return 1-torch.exp(-v)
            elif self.mode == 'KL':
                return v
            elif self.mode == 'JS':
                return np.log(2)-torch.log(1+torch.exp(-v))
            elif self.mode == 'TV':
                return torch.tanh(v)/2
            elif self.mode == 'W':
                return v
        def fs(t):
            if self.mode == 'SH':
                return t/(1-t)
            elif self.mode == 'KL':
                return torch.exp(t-1)
            elif self.mode == 'JS':
                return -torch.log(2-torch.exp(t))
            elif self.mode == 'TV':
                return t
            elif self.mode == 'W':
                return t
        return (fs(gf(d_generated)) - gf(d_real)).mean()
            

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = 0
        for idx, local in enumerate(self.network):
            local_feature_idxs = np.arange(self.nfeatures.sum())[np.concatenate([self.feature_idxs[node] for node in local])]
            prob_interpolated += self.D[idx](interpolated[:,local_feature_idxs])
        
        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() 
                               if self.use_cuda else torch.ones(prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * (F.relu(gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            
            batch_size = data.size()[0]
            data = Variable(data)
            if self.use_cuda:
                data = data.cuda()
            latent_samples = Variable(self.G.sample_latent(batch_size))
            if self.use_cuda:
                latent_samples = latent_samples.cuda()
            
            self._critic_train_iteration(data, latent_samples, self.tau)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data, latent_samples, self.tau)
            
            # Visualization
            if self.num_steps % self.print_every == 0 and self.num_steps > self.critic_iterations:
                display.clear_output(wait=True)
                self.fig = plt.figure(figsize=(17, 16))
                self.fig.suptitle("{}, Epoch: {}/{}".format(self.expname, self.epoch, self.epochs), fontsize=24)
                gs0 = self.fig.add_gridspec(2, 2)
                gs1 = gs0[0].subgridspec(2, 1)
                gs2 = gs0[1].subgridspec(2, 1)
                ax1 = self.fig.add_subplot(gs1[0])
                ax2 = self.fig.add_subplot(gs1[1])
                ax3 = self.fig.add_subplot(gs2[0])
                ax4 = self.fig.add_subplot(gs2[1])
                ax5 = self.fig.add_subplot(gs0[2])
                ax6 = self.fig.add_subplot(gs0[3])
                
                ax1.plot(self.losses['D'])
                ax1.set_ylabel('Discriminator Loss', fontsize=16)
                
                self.generated_data = self.G(self.fixed_latents, self.tau).detach().cpu().data.numpy()
                self.losses['energy_statistics'].append(energy_statistics(self.sampled_data, self.generated_data))
                ax2.plot(self.print_every*np.arange(len(self.losses['energy_statistics'])), self.losses['energy_statistics'])
                ax2.set_xlabel('Iterations', fontsize=16)
                ax2.set_ylabel('Energy Statistics', fontsize=16)
                
                self.losses['relative_structure_score'].append(relative_structure_score(self.sampled_data, self.generated_data, encoder=self.encoder, bayesmodel_true=self.bayesmodel))
                ax3.plot(self.print_every*np.arange(len(self.losses['relative_structure_score'])), self.losses['relative_structure_score'])
                ax3.set_ylabel('Rel. Structure Score', fontsize=16)
                
                generated_dots = self.pca.transform(self.generated_data)
                ax5.scatter(self.true_dots[:,0], self.true_dots[:,1])
                ax5.scatter(generated_dots[:,0], generated_dots[:,1])
                
                if len(self.losses['relative_structure_score']) > 3 and self.losses['relative_structure_score'][-1] < self.losses['relative_structure_score'][-2] and \
                    self.losses['relative_structure_score'][-3] < self.losses['relative_structure_score'][-2] and (self.num_steps - self.prev_num_steps) >= 100 and False:
                    self.prev_num_steps = self.num_steps
                    
                    self.losses['structure_prediction_iteration'].append(self.num_steps-self.print_every)
                    
                    self.losses['bayesmodel_predicted'].append(structure_prediction(self.pre_generated_data, self.encoder, self.bayesmodel, method='hc', score_type='bic'))
                    self.losses['structure_prediction_accuracy'].append(structure_prediction_accuracy(self.bayesmodel, self.losses['bayesmodel_predicted'][-1]))
                    
                    if self.losses['relative_structure_score'][-2] > self.max_bic_score:
                        self.max_bic_score = self.losses['relative_structure_score'][-1]
                        self.G_state_dict = self.pre_G_state_dict
                
                ax4.plot(self.losses['structure_prediction_iteration'], self.losses['structure_prediction_accuracy'])
                ax4.set_xlabel('Iterations', fontsize=16)
                ax4.set_ylabel('Structure Prediction Acc.', fontsize=16)
                
                if len(self.losses['bayesmodel_predicted']) > 0:
                    bayesnet_compare_plot(self.bayesmodel, self.losses['bayesmodel_predicted'][-1], ax=ax6, pos=self.layout)
                
                plt.show()
                self.training_progress_images.append(img_as_ubyte(self._draw_img_grid(generated_dots)))

        self.G_sch.step()
                
    def _draw_img_grid(self, generated_dots):
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(self.true_dots[:,0], self.true_dots[:,1])
        plt.scatter(generated_dots[:,0], generated_dots[:,1])
        plt.title('Epoch {}'.format(self.epoch), fontsize=12)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image

    def train(self, data_loader, test_data_loader, expdesc, epochs):
        # Fix latents to see how image generation improves during training
        iter_test_data = iter(test_data_loader)
        sampled_data = [next(iter_test_data).data.numpy() for _ in range(8)]
        self.sampled_data = np.concatenate(sampled_data, axis=0)
        self.fixed_latents = Variable(self.G.sample_latent(self.sampled_data.shape[0]))
        if self.use_cuda:
            self.fixed_latents = self.fixed_latents.cuda()
        self.pca.fit(self.sampled_data)
        self.true_dots = self.pca.transform(self.sampled_data)
        self.training_progress_images = []
        
        display.clear_output(wait=True)
    
        self.epochs = epochs
        for epoch in range(epochs):
            self.epoch = epoch + 1
            self.tau = 1 - 0.9*(self.epoch/self.epochs)
            self._train_epoch(data_loader)
        
        self.expdesc = expdesc
        newpath = './results/{}/'.format(self.expdesc) 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        imageio.mimsave(newpath+'{}.gif'.format(self.expname), self.training_progress_images, 
                        format='GIF', duration=10.0 / len(self.training_progress_images))
        with open(newpath+'{}_losses.pickle'.format(self.expname), 'wb') as handle:
            pickle.dump(self.losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.fig.savefig(newpath+'{}.png'.format(self.expname))
        torch.save(self.G.state_dict(), newpath+'{}.pt'.format(self.expname+'_G'))
        torch.save(self.D.state_dict(), newpath+'{}.pt'.format(self.expname+'_D'))
        return self.G_state_dict