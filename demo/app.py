import torch
import torch.nn as nn
import cv2
import numpy as np
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = SRCNN().to(device)
model.eval()

def super_resolve(img):
    img = np.array(img)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)

    y = y.squeeze().permute(1, 2, 0).cpu().numpy()
    y = (y * 255).astype(np.uint8)
    return y

demo = gr.Interface(
    fn=super_resolve,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="numpy"),
    title="Satellite Image Super-Resolution"
)

demo.launch(share=True)
