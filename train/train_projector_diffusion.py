"""
Diffusion-based Projector training script
Replaces the original projector with a diffusion model, input and output are the same as the original framework
"""

import sys
import os

# Add resources directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

import numpy as np
import tquat
import txform
import quat
import bvh

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.neighbors import BallTree

from models_diffusion import DiffusionProjector
from train_common import load_database, load_features, load_latent, save_network

# Training procedure

if __name__ == '__main__':
    
    # Data paths
    resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
    database_path = os.path.join(resources_dir, 'database.bin')
    features_path = os.path.join(resources_dir, 'features.bin')
    latent_path = os.path.join(resources_dir, 'latent.bin')
    output_dir = resources_dir
    
    # Load data
    database = load_database(database_path)
    range_starts = database['range_starts']
    range_stops = database['range_stops']
    del database
    
    X = load_features(features_path)['features'].copy().astype(np.float32)
    Z = load_latent(latent_path)['latent'].copy().astype(np.float32)
    
    nframes = X.shape[0]
    nfeatures = X.shape[1]
    nlatent = Z.shape[1]
    
    # Parameters
    seed = 1234
    batchsize = 64
    lr = 0.001
    niter = 500000
    num_timesteps = 1000  # Diffusion timesteps
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    
    # Fit acceleration structure for nearest neighbors search
    tree = BallTree(X)
    
    # Compute means/stds (same as original projector)
    X_scale = X.std()
    X_noise_std = X.std(axis=0) + 1.0
    
    projector_mean_out = torch.as_tensor(np.hstack([
        X.mean(axis=0).ravel(),
        Z.mean(axis=0).ravel(),
    ]).astype(np.float32))
    
    projector_std_out = torch.as_tensor(np.hstack([
        X.std(axis=0).ravel(),
        Z.std(axis=0).ravel(),
    ]).astype(np.float32))
    
    projector_mean_in = torch.as_tensor(np.hstack([
        X.mean(axis=0).ravel(),
    ]).astype(np.float32))
    
    projector_std_in = torch.as_tensor(np.hstack([
        X_scale.repeat(nfeatures),
    ]).astype(np.float32))
    
    # Make networks
    network_projector = DiffusionProjector(nfeatures, nfeatures + nlatent, num_timesteps=num_timesteps)
    
    # Function to generate test predictions
    def generate_predictions():
        with torch.no_grad():
            # Get slice of database for first clip
            start = range_starts[2]
            stop = min(start + 1000, range_stops[2])
            
            nsigma = np.random.uniform(size=[stop-start, 1]).astype(np.float32)
            noise = np.random.normal(size=[stop-start, nfeatures]).astype(np.float32)
            Xhat = X[start:stop] + X_noise_std * nsigma * noise
            
            # Find nearest
            nearest = tree.query(Xhat, k=1, return_distance=False)[:,0]
            
            Xgnd = torch.as_tensor(X[nearest])
            Zgnd = torch.as_tensor(Z[nearest])
            Xhat = torch.as_tensor(Xhat)
            Dgnd = torch.sqrt(torch.sum(torch.square(Xhat - Xgnd), dim=-1))
            
            # Normalize input
            Xhat_norm = (Xhat - projector_mean_in) / projector_std_in
            
            # Project using diffusion (fast inference mode - uses timestep=0 for fully denoised prediction)
            # This is compatible with the original projector interface
            # For better quality, can use: network_projector.sample(Xhat_norm, num_steps=50)
            # But for real-time use, timestep=0 is sufficient
            with torch.no_grad():
                output = network_projector(Xhat_norm, timestep=torch.zeros(stop-start, dtype=torch.long))
            
            # Denormalize output
            output = output * projector_std_out + projector_mean_out
            
            Xtil = output[:,:nfeatures]
            Ztil = output[:,nfeatures:]
            Dtil = torch.sqrt(torch.sum(torch.square(Xhat - Xtil), dim=-1))
            
            # Write features
            fmin, fmax = Xhat.cpu().numpy().min(), Xhat.cpu().numpy().max()
            
            fig, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2*nfeatures))
            for i in range(nfeatures):
                axs[i].plot(Xgnd[:500:4,i].cpu().numpy(), marker='.', linestyle='None', label='Ground Truth')
                axs[i].plot(Xtil[:500:4,i].cpu().numpy(), marker='.', linestyle='None', label='Diffusion')
                axs[i].plot(Xhat[:500:4,i].cpu().numpy(), marker='.', linestyle='None', label='Query')
                axs[i].set_ylim(fmin, fmax)
                if i == 0:
                    axs[i].legend()
            plt.tight_layout()
            
            try:
                plt.savefig(os.path.join(output_dir, 'projector_diffusion_X.png'))
            except IOError as e:
                print(e)

            plt.close()
            
            # Write latent
            lmin, lmax = Zgnd.cpu().numpy().min(), Zgnd.cpu().numpy().max()
            
            fig, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2*nlatent))
            for i in range(nlatent):
                axs[i].plot(Zgnd[:500:4,i].cpu().numpy(), marker='.', linestyle='None', label='Ground Truth')
                axs[i].plot(Ztil[:500:4,i].cpu().numpy(), marker='.', linestyle='None', label='Diffusion')
                axs[i].set_ylim(lmin, lmax)
                if i == 0:
                    axs[i].legend()
            plt.tight_layout()
            
            try:
                plt.savefig(os.path.join(output_dir, 'projector_diffusion_Z.png'))
            except IOError as e:
                print(e)

            plt.close()

    # Train
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'runs', 'projector_diffusion'))

    optimizer = torch.optim.AdamW(
        network_projector.parameters(), 
        lr=lr,
        amsgrad=True,
        weight_decay=0.001)
        
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    rolling_loss = None
    
    sys.stdout.write('\n')
    
    for i in range(niter):
    
        optimizer.zero_grad()
        
        # Extract batch
        samples = np.random.randint(0, nframes, size=[batchsize])
        
        # Get ground truth features and latents
        Xgnd = torch.as_tensor(X[samples])
        Zgnd = torch.as_tensor(Z[samples])
        
        # Create noisy query features
        nsigma = np.random.uniform(size=[batchsize, 1]).astype(np.float32)
        noise = np.random.normal(size=[batchsize, nfeatures]).astype(np.float32)
        Xhat = X[samples] + X_noise_std * nsigma * noise
        Xhat = torch.as_tensor(Xhat)
        
        # Normalize input
        Xhat_norm = (Xhat - projector_mean_in) / projector_std_in
        
        # Prepare target (normalized)
        target = torch.cat([Xgnd, Zgnd], dim=-1)
        target_norm = (target - projector_mean_out) / projector_std_out
        
        # Sample random timesteps
        # Use smaller timesteps more frequently (80% of the time) to focus on final output quality
        if np.random.rand() < 0.8:
            timesteps = torch.randint(0, min(200, num_timesteps), (batchsize,), device=Xhat_norm.device).long()
        else:
            timesteps = torch.randint(0, num_timesteps, (batchsize,), device=Xhat_norm.device).long()
        
        # Sample noise to add
        noise_target = torch.randn_like(target_norm)
        
        # Compute noisy target at timestep t
        # Using linear noise schedule: sqrt(alpha_t) and sqrt(1 - alpha_t)
        alpha_t = 1.0 - (timesteps.float() / num_timesteps)
        alpha_t = alpha_t.unsqueeze(-1)
        
        noisy_target = alpha_t.sqrt() * target_norm + (1.0 - alpha_t).sqrt() * noise_target
        
        # Predict noise
        predicted_noise = network_projector.predict_noise(Xhat_norm, timesteps)
        
        # Compute losses
        # Main loss: predict the noise (for diffusion training)
        loss_noise = torch.mean(torch.square(predicted_noise - noise_target))
        
        # Direct prediction loss - separate X and Z to match original projector
        predicted_output = network_projector(Xhat_norm, timesteps)
        predicted_denorm = predicted_output * projector_std_out + projector_mean_out
        Xtil = predicted_denorm[:,:nfeatures]
        Ztil = predicted_denorm[:,nfeatures:]
        
        # Separate X and Z losses with original weights
        loss_xval = torch.mean(1.0 * torch.abs(Xgnd - Xtil))
        loss_zval = torch.mean(5.0 * torch.abs(Zgnd - Ztil))  # ⭐ Key: Restore 5.0 weight for Z
        
        # Distance loss (same as original)
        Dgnd = torch.sqrt(torch.sum(torch.square(Xhat - Xgnd), dim=-1))
        Dtil = torch.sqrt(torch.sum(torch.square(Xhat - Xtil), dim=-1))
        loss_dist = torch.mean(0.3 * torch.abs(Dgnd - Dtil))
        
        # Combined loss: reduce noise loss weight, emphasize direct prediction
        # This matches the original projector's focus on direct mapping
        loss = 0.2 * loss_noise + loss_xval + loss_zval + loss_dist
        
        # Backprop
        loss.backward()

        optimizer.step()
    
        # Logging
        writer.add_scalar('projector_diffusion/loss', loss.item(), i)
        
        writer.add_scalars('projector_diffusion/loss_terms', {
            'noise': loss_noise.item(),
            'xval': loss_xval.item(),
            'zval': loss_zval.item(),
            'dist': loss_dist.item(),
        }, i)
        
        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01
        
        if i % 10 == 0:
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))
        
        if i % 1000 == 0:
            generate_predictions()
            # Save model - create a simplified sequential model for C++ compatibility
            # The C++ framework expects a simple feed-forward network with ReLU activations
            # We create a wrapper that approximates the diffusion model at timestep=0
            
            # Create a simplified sequential model for C++ compatibility
            # This approximates the diffusion model at timestep=0 (fully denoised)
            from models import Projector as SimpleProjector
            
            # Create a simple projector that approximates the diffusion model
            simple_projector = SimpleProjector(nfeatures, nfeatures + nlatent)
            
            # Approximate diffusion model by training simple projector on diffusion outputs
            # Sample diverse inputs to learn the mapping
            test_samples = np.random.randint(0, nframes, size=[min(5000, nframes)])
            test_inputs = torch.as_tensor(X[test_samples])
            test_inputs_norm = (test_inputs - projector_mean_in) / projector_std_in
            
            # Get diffusion model outputs at timestep=0 (fully denoised)
            # Use no_grad for diffusion model inference
            with torch.no_grad():
                diffusion_outputs = network_projector(test_inputs_norm, timestep=torch.zeros(len(test_samples), dtype=torch.long))
            
            # Detach diffusion_outputs since it's only used as target, not for gradient computation
            diffusion_outputs = diffusion_outputs.detach()
            
            # Train simple projector to approximate diffusion outputs
            optimizer_simple = torch.optim.Adam(simple_projector.parameters(), lr=0.01)
            for epoch in range(200):  # Quick approximation training
                optimizer_simple.zero_grad()
                simple_outputs = simple_projector(test_inputs_norm)
                loss_simple = torch.mean(torch.square(simple_outputs - diffusion_outputs))
                loss_simple.backward()
                optimizer_simple.step()
                
                if epoch % 50 == 0:
                    print(f"\n  Approximation epoch {epoch}, loss: {loss_simple.item():.6f}")
            
            # Save the simplified model (compatible with C++ framework)
            projector_path = os.path.join(output_dir, 'projector_diffusion.bin')
            save_network(projector_path,
                [simple_projector.linear0, 
                 simple_projector.linear1, 
                 simple_projector.linear2, 
                 simple_projector.linear3,
                 simple_projector.linear4],
                projector_mean_in,
                projector_std_in,
                projector_mean_out,
                projector_std_out)
            
            # Also save the full diffusion model for Python use (optional)
            diffusion_path = os.path.join(output_dir, 'projector_diffusion.pt')
            torch.save({
                'model_state_dict': network_projector.state_dict(),
                'mean_in': projector_mean_in,
                'std_in': projector_std_in,
                'mean_out': projector_mean_out,
                'std_out': projector_std_out,
                'num_timesteps': num_timesteps,
            }, diffusion_path)
            
        if i % 1000 == 0:
            scheduler.step()

