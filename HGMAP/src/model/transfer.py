import torch
import torch.nn as nn
from typing import Dict, Optional, List
import numpy as np

class TransferLearningModule:
    def __init__(
        self,
        base_model: nn.Module,
        source_datasets: List[Dict[str, np.ndarray]],
        n_epochs: int = 50,
        lr: float = 1e-4
    ):
        self.base_model = base_model
        self.source_datasets = source_datasets
        self.n_epochs = n_epochs
        self.lr = lr
        self.pretrained_layers = {}
        
    def _pretrain_on_source(
        self,
        source_data: Dict[str, np.ndarray],
        layer_name: str
    ):
        """Pretrain specific layers using source domain data"""
        X_source = torch.FloatTensor(source_data['X'])
        y_source = torch.FloatTensor(source_data['y'])
        
        # Create temporary optimizer for pretraining
        optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        
        # Pretrain
        for epoch in range(self.n_epochs):
            self.base_model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.base_model(X_source)
            loss = nn.BCEWithLogitsLoss()(outputs, y_source)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Store pretrained layer weights
        self.pretrained_layers[layer_name] = {
            name: param.clone().detach()
            for name, param in self.base_model.named_parameters()
            if layer_name in name
        }
    
    def apply_transfer_learning(
        self,
        target_data: Dict[str, np.ndarray],
        layer_mapping: Optional[Dict[str, str]] = None
    ):
        """Apply transfer learning to target domain"""
        # Pretrain on source datasets
        for source_data in self.source_datasets:
            for layer_name in self.base_model.layer_names:
                self._pretrain_on_source(source_data, layer_name)
        
        # Initialize layer weights using pretrained weights
        if layer_mapping is None:
            # Use direct mapping
            for name, param in self.base_model.named_parameters():
                for layer_name, pretrained_weights in self.pretrained_layers.items():
                    if layer_name in name and name in pretrained_weights:
                        param.data = pretrained_weights[name].clone()
        else:
            # Use custom mapping
            for target_layer, source_layer in layer_mapping.items():
                if source_layer in self.pretrained_layers:
                    for name, param in self.base_model.named_parameters():
                        if target_layer in name:
                            source_name = name.replace(target_layer, source_layer)
                            if source_name in self.pretrained_layers[source_layer]:
                                param.data = self.pretrained_layers[source_layer][source_name].clone()
        
        # Fine-tune on target data
        X_target = torch.FloatTensor(target_data['X'])
        y_target = torch.FloatTensor(target_data['y'])
        
        optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.lr * 0.1,  # Lower learning rate for fine-tuning
            weight_decay=0.01
        )
        
        # Gradual unfreezing
        for epoch in range(self.n_epochs):
            if epoch == self.n_epochs // 2:
                # Unfreeze all layers
                for param in self.base_model.parameters():
                    param.requires_grad = True
            
            self.base_model.train()
            optimizer.zero_grad()
            
            outputs = self.base_model(X_target)
            loss = nn.BCEWithLogitsLoss()(outputs, y_target)
            
            loss.backward()
            optimizer.step()
        
        return self.base_model 