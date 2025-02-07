import os

from PIL import Image
import click
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.utils import save_image

from models import GeneratorResNet
from utils import tensor_from_path


hr_shape = (256, 256)
device = "cuda" if torch.cuda.is_available() else "cpu"


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


def norm_range(t, value_range=None):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))


def infer(filepath: str) -> None:
    os.makedirs("outputs", exist_ok=True)
    fname, ext = os.path.splitext(os.path.basename(filepath))

    model = GeneratorResNet()
    checkpoint = torch.load("./saved_models/G_best.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device=device)
    model.eval()

    img = Image.open(filepath)
    t = transforms.ToTensor()
    img = t(img).unsqueeze(0)

    x = tensor_from_path(filepath)
    x = x.to(device=device)
    with torch.no_grad():
        y_hat = model(x).to("cpu")

    lr = nn.functional.interpolate(img, size=y_hat.shape[-2:])
    norm_range(lr)
    norm_range(y_hat)
    out = torch.cat([lr, y_hat], -1)
    save_image(out[0], f"outputs/{fname}_sr{ext}", normalize=False)


@click.command()
@click.option("--img_path", type=str, help="path to input image")
def main(img_path: str):
    infer(img_path)


if __name__ == "__main__":
    main()
