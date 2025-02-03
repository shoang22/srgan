import os

import click
from torchvision.utils import save_image
import torch

from models import GeneratorResNet
from utils import tensor2img, tensor_from_path


hr_shape = (256, 256)
device = "cuda" if torch.cuda.is_available() else "cpu"


# def tensor_from_path(img_path) -> torch.Tensor:
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.transpose(img, (2, 0, 1))[None, :, :, :] / 255.0
#     img = img * 2 - 1
#     img = torch.from_numpy(img).float()
#     if len(img.shape) == 3:
#         img = img.unsqueeze(0)
#
#     return img
#


def infer(filepath: str) -> None:
    model = GeneratorResNet()
    checkpoint = torch.load("./saved_models/generator_best.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device=device)
    model.eval()

    x = tensor_from_path(filepath)
    with torch.no_grad():
        y_hat = model(x.to(device=device))

    os.makedirs("outputs", exist_ok=True)
    fname, ext = os.path.splitext(os.path.basename(filepath))

    save_image(y_hat[0], f"outputs/{fname}_sr{ext}", normalize=True)


@click.command()
@click.option("--img_path", type=str, help="path to input image")
def main(img_path: str):
    infer(img_path)


if __name__ == "__main__":
    main()
