import torchvision
import torch

CATEGORIES = ['apex']

device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
model = model.cuda().eval().half()

model.load_state_dict(torch.load('road_following_model.pth'))
