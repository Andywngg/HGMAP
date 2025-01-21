def train_model(self, train_loader, val_loader, epochs=100):
    # Mixup augmentation
    def mixup(x, y, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0))
        mixed_x = lam * x + (1 - lam) * x[idx]
        return mixed_x, y, y[idx], lam

    # Advanced optimizer with lookahead
    base_opt = AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer = Lookahead(base_opt, k=5, alpha=0.5)
    
    # Cosine annealing with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * 5,
        num_training_steps=len(train_loader) * epochs
    )
    
    # Automatic Mixed Precision
    scaler = GradScaler()
    
    for epoch in range(epochs):
        self.model.train()
        for batch in train_loader:
            with autocast():
                x, y = batch
                x, y_a, y_b, lam = mixup(x, y)
                
                output = self.model(x)
                loss = criterion(output, y_a) * lam + criterion(output, y_b) * (1 - lam)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step() 