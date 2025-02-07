from glob import glob
import os
import re

import yaml
import click
from torchvision.models.vision_transformer import math
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch

from models import Discriminator, GeneratorResNet, FeatureExtractor
from datasets import ImageDataset
from utils import make_gen_real_grid
from logger import logger


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hr_shape = (cfg["hr_height"], cfg["hr_width"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = GeneratorResNet()
        self.discriminator = Discriminator(
            input_shape=(cfg["channels"], *self.hr_shape)
        )
        self.feature_extractor = FeatureExtractor()
        self.generator = self.generator.to(device=self.device)
        self.discriminator = self.discriminator.to(device=self.device)
        self.feature_extractor = self.feature_extractor.to(device=self.device)

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_content = torch.nn.L1Loss()
        self.criterion_GAN = self.criterion_GAN.to(device=self.device)
        self.criterion_content = self.criterion_content.to(device=self.device)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=cfg["lr"], betas=(cfg["b1"], cfg["b2"])
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg["lr"], betas=(cfg["b1"], cfg["b2"])
        )

        self.writer = SummaryWriter(log_dir=self.cfg["path"]["writer"])

        for _, p in self.cfg["path"].items():
            os.makedirs(p, exist_ok=True)

        dataset = ImageDataset(
            os.path.join(self.cfg["path"]["data"], cfg["dataset_name"]),
            hr_shape=self.hr_shape,
        )
        train_length = int(len(dataset) * 0.7)
        test_length = len(dataset) - train_length
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset=dataset, lengths=(train_length, test_length)
        )
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["n_cpu"],
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["n_cpu"],
        )

    def resume_training(self):
        self.global_step = 1
        self.epoch = 1

        r = re.compile(r"(\d+)")
        mfp = glob(os.path.join(self.cfg["path"]["models"], "*_*[0-9].pth"))
        sfp = glob(os.path.join(self.cfg["path"]["training_state"], "*[0-9].state"))
        if (resume_epoch := self.cfg["resume_epoch"]) and len(mfp) > 0 and len(sfp) > 0:
            if resume_epoch == -1:
                mepochs = set([int(r.findall(i)[0]) for i in mfp])
                sepochs = set([int(r.findall(i)[0]) for i in sfp])
                epoch = max(mepochs.intersection(sepochs))
            else:
                epoch = resume_epoch

            resume_state = torch.load(
                os.path.join(self.cfg["path"]["training_state"], f"{epoch}.state")
            )
            self.load_training_state(state=resume_state)
            path_G = os.path.join(self.cfg["path"]["models"], f"G_{epoch}.pth")
            path_D = os.path.join(self.cfg["path"]["models"], f"D_{epoch}.pth")
            self.load_network(load_path_G=path_G, load_path_D=path_D)

            self.global_step = epoch * len(self.train_dataloader)
            self.epoch = epoch

        self.train()

    def train(self):
        logger.info(f"Starting training at epoch: {self.epoch}")
        while self.epoch <= self.cfg["n_epochs"]:
            self.train_loop(n_steps=self.cfg["n_train_steps_per_epoch"])
            self.eval_loop(n_steps=self.cfg["n_eval_steps_per_epoch"])
            self.epoch += 1

    def train_loop(self, n_steps: int | None = None):
        self.generator.train()
        self.discriminator.train()

        for i, imgs in enumerate(self.train_dataloader):
            if n_steps and i >= n_steps:
                break

            # Configure model input
            imgs_lr = imgs["lr"].to(device=self.device)
            imgs_hr = imgs["hr"].to(device=self.device)

            # Adversarial ground truths
            valid = torch.ones(
                (imgs_lr.size(0), *self.discriminator.output_shape), requires_grad=False
            ).to(device=self.device)
            fake = torch.zeros(
                (imgs_lr.size(0), *self.discriminator.output_shape), requires_grad=False
            ).to(device=self.device)

            # ------------------
            #  Train Generators
            # ------------------

            self.optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = self.generator(imgs_lr)
            # Adversarial loss
            loss_GAN = self.criterion_GAN(self.discriminator(gen_hr), valid)

            # Content loss
            gen_features = self.feature_extractor(gen_hr)
            real_features = self.feature_extractor(imgs_hr)
            loss_content = self.criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            loss_G.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            # Loss of real and fake images
            loss_real = self.criterion_GAN(self.discriminator(imgs_hr), valid)
            loss_fake = self.criterion_GAN(self.discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            self.optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            logger.info(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    self.epoch,
                    self.cfg["n_epochs"],
                    i + 1,
                    len(self.train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                )
            )

            self.writer.add_scalar(
                "/train/loss/discriminator", loss_D, self.global_step
            )
            self.writer.add_scalar("/train/loss/generator", loss_G, self.global_step)
            self.writer.add_image(
                "/train/outputs",
                make_gen_real_grid(gen_hr=gen_hr, imgs_lr=imgs_lr, n=1),
                global_step=self.global_step,
            )
            self.global_step += 1

            if self.global_step % self.cfg["sample_interval"] == 0:
                # Save image grid with upsampled inputs and SRGAN outputs
                img_grid = make_gen_real_grid(gen_hr=gen_hr, imgs_lr=imgs_lr, n=4)
                # Normalize=False because we already normalize in make_gen_real_grid
                save_image(
                    img_grid,
                    os.path.join(self.cfg["path"]["images"], f"{self.global_step}.png"),
                    normalize=False,
                )

        if (
            self.cfg["checkpoint_interval"] != -1
            and self.epoch % self.cfg["checkpoint_interval"] == 0
        ):
            self.save_network(epoch=self.epoch)
            self.save_training_state(epoch=self.epoch)

    def eval_loop(self, n_steps: int | None = None):
        self.generator.eval()
        self.discriminator.eval()

        running_vloss_D = 0.0
        running_vloss_G = 0.0

        best_vloss_G = math.inf

        with torch.no_grad():
            for i, imgs in enumerate(self.test_dataloader):
                if n_steps is not None and i >= n_steps:
                    break

                imgs_lr = imgs["lr"].to(device=self.device)
                imgs_hr = imgs["hr"].to(device=self.device)

                # Adversarial ground truths
                valid = torch.ones(
                    (imgs_lr.size(0), *self.discriminator.output_shape),
                    requires_grad=False,
                ).to(device=self.device)
                fake = torch.zeros(
                    (imgs_lr.size(0), *self.discriminator.output_shape),
                    requires_grad=False,
                ).to(device=self.device)

                gen_hr = self.generator(imgs_lr)

                loss_GAN = self.criterion_GAN(self.discriminator(gen_hr), valid)

                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr)
                loss_content = self.criterion_content(gen_features, real_features)

                loss_G = loss_content + 1e-3 * loss_GAN
                running_vloss_G += loss_G

                loss_real = self.criterion_GAN(self.discriminator(imgs_hr), valid)
                loss_fake = self.criterion_GAN(self.discriminator(gen_hr), fake)

                loss_D = (loss_real + loss_fake) / 2
                running_vloss_D += loss_D

                self.writer.add_image(
                    "/test/outputs",
                    make_gen_real_grid(gen_hr=gen_hr, imgs_lr=imgs_lr, n=1),
                    global_step=self.global_step,
                )

        avg_vloss_D = running_vloss_D / len(self.test_dataloader)
        avg_vloss_G = running_vloss_G / len(self.test_dataloader)
        self.writer.add_scalar(
            "/test/loss/discriminator", avg_vloss_D, self.global_step
        )
        self.writer.add_scalar("/test/loss/generator", avg_vloss_G, self.global_step)
        self.writer.flush()

        if best_vloss_G > avg_vloss_G:
            # Save model checkpoints
            torch.save(
                self.generator.state_dict(),
                os.path.join(self.cfg["path"]["models"], "G_best.pth"),
            )
            best_vloss_G = avg_vloss_G

        logger.info(
            "[Epoch %d/%d] [Eval] [Avg D loss: %f] [Avg G loss: %f]"
            % (
                self.epoch,
                self.cfg["n_epochs"],
                avg_vloss_D,
                avg_vloss_G,
            )
        )

    def save_training_state(self, epoch):
        """Save training state and optimizers so we can resume training"""
        state = {"epoch": epoch}
        state["optimizer_G"] = self.optimizer_G.state_dict()
        state["optimizer_D"] = self.optimizer_D.state_dict()
        save_fname = f"{epoch}.state"
        save_path = os.path.join(self.cfg["path"]["training_state"], save_fname)
        torch.save(state, save_path)

    def load_training_state(self, state):
        """Load training state and optimizers so we can resume training"""
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])

    def save_network(self, epoch):
        torch.save(
            self.generator.state_dict(),
            os.path.join(self.cfg["path"]["models"], f"G_{epoch}.pth"),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(self.cfg["path"]["models"], f"D_{epoch}.pth"),
        )

    def load_network(self, load_path_G, load_path_D, strict=True):
        self.generator.load_state_dict(torch.load(load_path_G), strict=strict)
        self.discriminator.load_state_dict(torch.load(load_path_D), strict=strict)


@click.command
@click.option("--cfg_file", type=str, help="Path of config file.")
def main(cfg_file: str):
    with open(cfg_file, "r") as file:
        cfg = yaml.safe_load(file)

    trainer = Trainer(cfg=cfg)
    trainer.resume_training()


if __name__ == "__main__":
    main()
