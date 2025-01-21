import numpy as np
from scipy.interpolate import interp1d
import torch
from torch.nn.functional import normalize

class MicrobiomeAugmentor:
    def __init__(self, noise_level=0.05, mix_alpha=0.2):
        self.noise_level = noise_level
        self.mix_alpha = mix_alpha
    
    def gaussian_noise(self, data):
        """Add calibrated Gaussian noise while preserving compositional nature"""
        noise = np.random.normal(0, self.noise_level, data.shape)
        augmented = data + noise
        return normalize(torch.tensor(augmented), p=1, dim=1).numpy()
    
    def temporal_warping(self, data, num_segments=5):
        """Apply non-linear temporal warping"""
        orig_points = np.linspace(0, 1, data.shape[1])
        random_points = np.sort(np.random.uniform(0, 1, num_segments))
        warp_func = interp1d(orig_points, random_points, kind='cubic')
        warped_data = np.apply_along_axis(lambda x: np.interp(warp_func(orig_points), orig_points, x), 1, data)
        return warped_data
    
    def taxonomic_mixup(self, data, labels):
        """Mix samples while respecting taxonomic relationships"""
        beta = np.random.beta(self.mix_alpha, self.mix_alpha)
        perm = torch.randperm(data.shape[0])
        mixed_data = beta * data + (1 - beta) * data[perm]
        mixed_labels = beta * labels + (1 - beta) * labels[perm]
        return mixed_data, mixed_labels
    
    def compositional_perturbation(self, data):
        """Perturb abundances while maintaining compositional constraints"""
        dirichlet_params = data * 100  # Scale for reasonable concentration parameters
        perturbed = np.random.dirichlet(dirichlet_params, size=1)[0]
        return perturbed
    
    def __call__(self, data, labels=None):
        """Apply random combination of augmentations"""
        augmented_data = data.copy()
        
        # Randomly apply augmentations
        if np.random.random() < 0.5:
            augmented_data = self.gaussian_noise(augmented_data)
        if np.random.random() < 0.3:
            augmented_data = self.temporal_warping(augmented_data)
        if np.random.random() < 0.4 and labels is not None:
            augmented_data, labels = self.taxonomic_mixup(augmented_data, labels)
        if np.random.random() < 0.3:
            augmented_data = self.compositional_perturbation(augmented_data)
            
        return augmented_data, labels 