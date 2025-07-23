import torch
import torchvision
from torch2trt import torch2trt
from pathlib import Path

class TRTModelConverter:
    def __init__(self, num_classes: int = 3,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.half):
        """
        Initialize the TensorRT converter for a ResNet model.

        Args:
            num_classes (int): Number of output classes.
            device (torch.device, optional): Device for model and data. Defaults to CUDA if available.
            dtype (torch.dtype, optional): Data type for model and inputs. Defaults to torch.half.
        """
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        self.num_classes = num_classes
        self.model: torch.nn.Module = None
        self.model_trt: torch.nn.Module = None

    def setup_model(self) -> None:
        """
        Builds the ResNet-18 architecture, replaces the final layer,
        and moves it to the target device and dtype.
        """
        print("Setting up the base model...")
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        model = model.to(self.device).eval().to(self.dtype)
        self.model = model
        print(f"Model initialized on {self.device} with dtype {self.dtype}")

    def load_model(self, model_path: str,
                   map_location: torch.device = None) -> None:
        """
        Loads pretrained weights into the model from a .pth file.

        Args:
            model_path (str): Path to the .pth file.
            map_location (torch.device, optional): Device for loading. Defaults to self.device.

        Raises:
            FileNotFoundError: If the weights file does not exist.
            RuntimeError: If loading fails.
        """
        filepath = Path(model_path)
        if not filepath.is_file():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        print(f"Loading weights from {filepath}...")
        loc = map_location or self.device
        state_dict = torch.load(str(filepath), map_location=loc)
        self.model.load_state_dict(state_dict)
        print("Weights loaded successfully.")

    def convert_to_trt(self,
                       input_size: tuple = (1, 3, 224, 224),
                       fp16_mode: bool = True) -> None:
        """
        Converts the loaded PyTorch model into a TensorRT engine.

        Args:
            input_size (tuple): Shape of the sample input tensor.
            fp16_mode (bool): Enable FP16 mode for TensorRT conversion.

        Raises:
            RuntimeError: If model is not set up or loaded.
        """
        if self.model is None:
            raise RuntimeError("Call setup_model() and load_model() before conversion.")
        print("Creating sample input tensor...")
        sample = torch.zeros(input_size, device=self.device, dtype=self.dtype)
        print("Running TensorRT conversion...")
        self.model_trt = torch2trt(self.model, [sample], fp16_mode=fp16_mode)
        print("Conversion to TensorRT complete.")

    def save_trt_model(self, output_path: str) -> None:
        """
        Saves the TensorRT engine state dictionary.

        Args:
            output_path (str): Path to save the TRT .pth file.

        Raises:
            RuntimeError: If conversion has not been run.
        """
        if self.model_trt is None:
            raise RuntimeError("TensorRT model not available. Run convert_to_trt() first.")
        outfile = Path(output_path)
        torch.save(self.model_trt.state_dict(), str(outfile))
        print(f"TensorRT model saved to {outfile}")

if __name__ == "__main__":
    converter = TRTModelConverter(num_classes=3)
    converter.setup_model()
    converter.load_model("combined_data_model.pth")
    converter.convert_to_trt()
    converter.save_trt_model("combined_data_model_trt.pth")