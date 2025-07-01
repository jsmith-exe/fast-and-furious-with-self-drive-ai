import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch2trt import torch2trt

class ModelTrainer:
    def __init__(
        self,
        data_dir: str = "dataset",
        batch_size: int = 32,
        num_epochs: int = 30,
        lr: float = 1e-4,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # transforms & loader
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(data_dir, transform=self.transform)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # model, loss, optimizer
        model = models.resnet18(pretrained=True)
        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, len(dataset.classes))
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for imgs, labels in self.loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * imgs.size(0)
            epoch_loss = running_loss / len(self.loader.dataset)
            print(f"Epoch [{epoch+1}/{self.num_epochs}]  Loss: {epoch_loss:.4f}")

    def save_model(self, out_path: str = "resnet18_3class.pth"):
        torch.save(self.model.state_dict(), out_path)
        print(f"Saved trained model → {out_path}")

if __name__ == "__main__":
    # 1) Train
    trainer = ModelTrainer(data_dir="dataset", batch_size=32, num_epochs=30)
    trainer.train()
    trainer.save_model("resnet18_3class.pth")