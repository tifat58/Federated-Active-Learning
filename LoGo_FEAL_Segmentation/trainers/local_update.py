import torch
from torch.utils.data import DataLoader
from utils.evidential_loss import edl_mse_loss, evidential_segmentation_loss

class LocalUpdate:
    def __init__(self, model, dataset, idxs, device, lr=5e-4, local_epochs=1, batch_size=32, kl_weight=1e-2, annealing_step=10, num_classes=4, cid =None):
        self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.idxs = list(idxs)
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step
        self.num_classes = num_classes
        self.cid = cid

        self.ldr_train = DataLoader(
            torch.utils.data.Subset(self.dataset, self.idxs),
            batch_size=self.batch_size,
            shuffle=True
        )

    def train(self, task='classification', lam=1e-3):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        epoch_loss = 0

        for epoch in range(self.local_epochs):
            for batch_idx, batch in enumerate(self.ldr_train):
                optimizer.zero_grad()

                if task == 'classification':
                    images, labels, _ = batch
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
                    annealed_kl_weight = min(1.0, epoch / self.annealing_step) * self.kl_weight

                    loss = edl_mse_loss(outputs, labels_onehot, epoch, self.num_classes, annealed_kl_weight)

                elif task == 'segmentation':
                    images, masks, _ = batch
                    images, masks = images.to(self.device), masks.to(self.device)

                    if masks.dim() == 3:  # Convert to one-hot
                        masks = torch.nn.functional.one_hot(masks.long(), num_classes=self.num_classes)
                        masks = masks.permute(0, 3, 1, 2).float()

                    alpha = self.model(images)
                    loss = evidential_segmentation_loss(alpha, masks, lam=lam)

                else:
                    raise ValueError(f"Unsupported task: {task}")

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        return self.model.state_dict(), epoch_loss / len(self.ldr_train)