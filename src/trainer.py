import torch


class ModelTrainer:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(images)

            loss = self.criterion(predictions, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted_classes = torch.max(predictions, dim=1)

            correct_predictions += (predicted_classes == labels).sum().item()

            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)

        accuracy = (correct_predictions / total_samples) * 100

        return avg_loss, accuracy


    def train(self, train_loader, epochs=10):
        print(f"Start of the training session on the device : {self.device}...")

        for epoch in range(epochs):
            loss, acc = self.train_epoch(train_loader)

            print(f"Epoch {epoch+1}/{epochs} | Loss : {loss:.4f} | Accuracy : {acc :.4f}% " )


        print("Training done !")
