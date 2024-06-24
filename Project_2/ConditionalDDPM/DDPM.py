import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.  

        # # Normalize the time steps
        # t_s_normalized = (t_s.float() - 1) / (T - 1)
    
        # Compute beta_t for the given time step
        beta_t = (beta_1 + (beta_T - beta_1)/(T - 1) * (t_s - 1)).to(device)
    
        # Compute alpha_t for all time steps from 0 to t_s
        alpha_t_prev = 1 - torch.linspace(beta_1, beta_T, T)
    
        # Compute alpha_t_bar as the cumulative product of alpha_t
        alpha_t_bar = torch.cumprod(alpha_t_prev.to(device), dim = 0)[t_s.long().to(device)-1]
    
        # Compute other related constants
        alpha_t = 1 - beta_t
        sqrt_beta_t = torch.sqrt(beta_t)
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        oneover_sqrt_alpha = 1 / torch.sqrt(alpha_t)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)

        beta_t = beta_t.to(device)
        alpha_t_prev = alpha_t_prev.to(device)
        alpha_t = alpha_t.to(device)
        alpha_t_bar = alpha_t_bar.to(device)
        sqrt_beta_t = sqrt_beta_t.to(device)
        sqrt_alpha_bar = sqrt_alpha_bar.to(device)
        oneover_sqrt_alpha = oneover_sqrt_alpha.to(device)
        sqrt_oneminus_alpha_bar = sqrt_oneminus_alpha_bar.to(device)
        

        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function.  
        
        # uniformly sample steps from 1 to T
        batch_size = images.size(0)

        # import pdb; pdb.set_trace()
        sampled_time_steps = torch.randint(1, T+1, (batch_size, 1), device=device) # (B, 1)
        
        # turn conditions into one-hot encoding
        conditions = F.one_hot(conditions, num_classes=self.dmconfig.num_classes).float().to(device)

        # mask condition 
        masked_conditions = torch.where(
            torch.rand(batch_size, 1, device=device).repeat(1, conditions.shape[1]) < float(self.dmconfig.mask_p),
            torch.full_like(conditions, self.dmconfig.condition_mask_value),
            conditions,
        )

        # import pdb; pdb.set_trace()

        # sample noise (for the forward pass)
        noise = torch.randn_like(images).to(device)
        noise_schedule_dict = self.scheduler(sampled_time_steps)
        noised_image = (
            noise_schedule_dict["sqrt_alpha_bar"].view(-1, 1, 1, 1) * images 
            + noise_schedule_dict["sqrt_oneminus_alpha_bar"].view(-1, 1, 1, 1) * noise
        )

        # normalize the time steps before sending into UNet
        normalized_sampled_time_steps = sampled_time_steps.float() / T
        normalized_sampled_time_steps = normalized_sampled_time_steps.view(-1, 1, 1, 1).to(device)

        # input the noised image, timestep and the conditions to the network
        # import pdb; pdb.set_trace()
        noise_pred = self.network(noised_image, normalized_sampled_time_steps, masked_conditions)

        # compute noise loss
        noise_loss = self.loss_fn(noise_pred, noise)


        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        batch_size = conditions.size(0)

        X_t = torch.randn(
            batch_size, 
            self.dmconfig.num_channels,
            self.dmconfig.input_dim[0],
            self.dmconfig.input_dim[1], 
            device=device
        )
        with torch.no_grad():
            for t in reversed(range(1, T+1)):
                # compute nomralized time step
                nt = float(t / T)

                # get noise_schedule_dict 
                noise_schedule_dict = self.scheduler(t * torch.ones(batch_size, 1, dtype=torch.int, device=device))

                # sample z from N(0, I)
                z = torch.randn_like(X_t) if t > 1 else 0.

                # turn conditions into one-hot encoding
                # NOTE: Somtimes conditions are already one-hot encoded
                if conditions.shape != (batch_size, self.dmconfig.num_classes):
                    conditions = F.one_hot(conditions, num_classes=self.dmconfig.num_classes).float()

                masked_conditions = self.dmconfig.condition_mask_value * torch.ones_like(conditions)

                # import pdb; pdb.set_trace()
                # get conditional noise prediction
                normalized_time_steps = nt * torch.ones(batch_size, 1, dtype=torch.float, device=device)
                normalized_time_steps = normalized_time_steps.view(-1, 1, 1, 1)
                cond_noise_pred = self.network(X_t, normalized_time_steps, conditions)

                # get unconditional noise prediction
                noise_pred = self.network(X_t, normalized_time_steps, masked_conditions)

                # get weighted noise
                weighted_noise = (1 + omega) * cond_noise_pred - omega * noise_pred

                # update X_t
                weighted_noise_coeff = (
                    (1 - noise_schedule_dict["alpha_t"].view(-1, 1, 1, 1)) 
                    / noise_schedule_dict["sqrt_oneminus_alpha_bar"].view(-1, 1, 1, 1)
                )
                
                std_t = noise_schedule_dict["sqrt_beta_t"].view(-1, 1, 1, 1)
                X_pre = X_t
                # import pdb; pdb.set_trace()
                X_t = (
                    noise_schedule_dict["oneover_sqrt_alpha"].view(-1, 1, 1, 1) * 
                    (X_t - weighted_noise_coeff * weighted_noise)
                    + std_t * z
                )

        
        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images