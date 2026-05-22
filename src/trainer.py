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

        
        return {"loss" : avg_loss, "accuracy" : accuracy}



    def test(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():

            for images, labels in test_loader: 
                images = images.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(images)
                

                loss = self.criterion(predictions, labels)
                running_loss += loss.item() * images.size(0)

                _, predicted_classes = torch.max(predictions, dim=1)

                correct_predictions += (predicted_classes == labels).sum().item()
                total_samples += labels.size(0)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_predictions.extend(predictions.cpu().numpy().tolist())
                

        test_acc = (correct_predictions / total_samples) * 100
        test_loss = running_loss / total_samples

        return {
            "loss" : test_loss, "accuracy" : test_acc,
            "y_true" : all_labels, "y_pred" : all_predictions
            }


    def save_model(self, filepath="mon_modele_v1.pth"):
        torch.save(self.model.state_dict(), filepath)
        print(f"Modèle sauvegardé avec succès dans {filepath}")