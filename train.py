import argparse
import os

import yaml
import click
from torchvision.models.vision_transformer import math
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
import torch

from models import Discriminator, GeneratorResNet, FeatureExtractor
from datasets import ImageDataset
from logger import logger


def make_gen_real_grid(gen_hr, imgs_lr, n: int):
    imgs_lr = nn.functional.interpolate(imgs_lr[:n], scale_factor=4)
    gen_hr = make_grid(gen_hr[:n], nrow=1, normalize=True)
    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
    img_grid = torch.cat((imgs_lr, gen_hr), -1)
    return img_grid


os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument(
    "--n_epochs", type=int, default=200, help="number of epochs of training"
)
parser.add_argument(
    "--dataset_name", type=str, default="img_align_celeba", help="name of the dataset"
)
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--decay_epoch", type=int, default=100, help="epoch from which to start lr decay"
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval",
    type=int,
    default=100,
    help="interval between saving image samples",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=-1,
    help="interval between model checkpoints",
)
opt = parser.parse_args()
logger.info(opt)

device = "cuda" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter(log_dir="./runs")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

generator = generator.to(device=device)
discriminator = discriminator.to(device=device)
feature_extractor = feature_extractor.to(device=device)
criterion_GAN = criterion_GAN.to(device=device)
criterion_content = criterion_content.to(device=device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)

dataset = ImageDataset("../data/%s" % opt.dataset_name, hr_shape=hr_shape)

train_length = int(len(dataset) * 0.7)
test_length = len(dataset) - train_length
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset=dataset, lengths=(train_length, test_length)
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)


# ----------
#  Training
# ----------


global_step = 0
for epoch in range(opt.epoch, opt.n_epochs):
    generator.train()
    discriminator.train()
    for i, imgs in enumerate(train_dataloader):
        # Configure model input
        imgs_lr = imgs["lr"].to(device=device)
        imgs_hr = imgs["hr"].to(device=device)

        # Adversarial ground truths
        valid = torch.ones(
            (imgs_lr.size(0), *discriminator.output_shape), requires_grad=False
        ).to(device=device)
        fake = torch.zeros(
            (imgs_lr.size(0), *discriminator.output_shape), requires_grad=False
        ).to(device=device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        logger.info(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_dataloader),
                loss_D.item(),
                loss_G.item(),
            )
        )

        writer.add_scalar("/train/loss/discriminator", loss_D, global_step)
        writer.add_scalar("/train/loss/generator", loss_G, global_step)
        writer.add_image(
            "/train/outputs",
            make_gen_real_grid(gen_hr=gen_hr, imgs_lr=imgs_lr, n=1),
            global_step=global_step,
        )
        global_step += 1

        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            img_grid = make_gen_real_grid(gen_hr=gen_hr, imgs_lr=imgs_lr, n=4)
            # Normalize=False because we already normalize in make_gen_real_grid
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(
            discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch
        )

    # ----------
    # Evaluation
    # ----------

    generator.eval()
    discriminator.eval()

    running_vloss_D = 0.0
    running_vloss_G = 0.0

    best_vloss_G = math.inf

    with torch.no_grad():
        for i, imgs in enumerate(test_dataloader):
            imgs_lr = imgs["lr"].to(device=device)
            imgs_hr = imgs["hr"].to(device=device)

            # Adversarial ground truths
            valid = torch.ones(
                (imgs_lr.size(0), *discriminator.output_shape), requires_grad=False
            ).to(device=device)
            fake = torch.zeros(
                (imgs_lr.size(0), *discriminator.output_shape), requires_grad=False
            ).to(device=device)

            gen_hr = generator(imgs_lr)

            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features)

            loss_G = loss_content + 1e-3 * loss_GAN
            running_vloss_G += loss_G

            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr), fake)

            loss_D = (loss_real + loss_fake) / 2
            running_vloss_D += loss_D

            writer.add_image(
                "/test/outputs",
                make_gen_real_grid(gen_hr=gen_hr, imgs_lr=imgs_lr, n=1),
                global_step=epoch,
            )

    avg_vloss_D = running_vloss_D / len(test_dataloader)
    avg_vloss_G = running_vloss_G / len(test_dataloader)
    writer.add_scalar("/test/loss/discriminator", avg_vloss_D, epoch)
    writer.add_scalar("/test/loss/generator", avg_vloss_G, epoch)
    writer.flush()

    if best_vloss_G > avg_vloss_G:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_best.pth")
        best_vloss_G = avg_vloss_G

    logger.info(
        "[Epoch %d/%d] [Eval] [Avg D loss: %f] [Avg G loss: %f]"
        % (
            epoch,
            opt.n_epochs,
            avg_vloss_D,
            avg_vloss_G,
        )
    )
