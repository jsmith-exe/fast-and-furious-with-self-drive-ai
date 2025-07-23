import torch
from torch2trt import TRTModule
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path


class CollisionAvoidanceEvaluator:
    def __init__(self,
                 trt_model_path: str,
                 data_dir: str,
                 batch_size: int = 1,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.half):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        self.trt_model_path = trt_model_path
        self.batch_size = batch_size

        self.model = self._load_trt_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dataset = ImageFolder(data_dir, transform=self.transform)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.class_names = self.dataset.classes

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).to(self.dtype)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).to(self.dtype)

    def _load_trt_model(self):
        path = Path(self.trt_model_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        print(f"Loading TRT model from {path}...")
        module = TRTModule()
        state = torch.load(str(path), map_location=self.device)
        module.load_state_dict(state)
        module.to(self.device).eval()
        print("Model loaded.")
        return module

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        t = img.to(self.device).to(self.dtype)
        t = (t - self.mean[:, None, None]) / self.std[:, None, None]
        return t.unsqueeze(0)

    def evaluate(self) -> float:
        correct, total = 0, len(self.dataset)
        print("Starting evaluation...\n")
        for imgs, labels in self.loader:
            inp = self._preprocess(imgs[0])
            labels = labels.to(self.device)
            with torch.no_grad():
                out = self.model(inp)
                _, pred = torch.max(out, 1)
            pc = self.class_names[pred.item()]
            tc = self.class_names[labels.item()]
            print(f"Predicted: {pc.ljust(8)} | Ground Truth: {tc}")
            if pred == labels:
                correct += 1
        acc = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} = {acc:.2f}%")
        return acc


if __name__ == "__main__":
    evaluator = CollisionAvoidanceEvaluator(
        trt_model_path="automated_data_box_trt.pth",
        data_dir="automated_data",
        batch_size=1
    )
    evaluator.evaluate()