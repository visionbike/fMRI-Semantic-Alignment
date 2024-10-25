import random
from argparse import Namespace, ArgumentParser
from pathlib import Path
import torch
import numpy as np
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from dataset import get_dataloaders
from graph import *


class ModelMRI:
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed(0)
            self.device_ids = args.gpu_ids.split(",")
            self.device = torch.device(f"cuda:{self.device_ids[0]}")
        else:
            self.device = torch.device("cpu")

        self.loader_train, self.loader_val, self.loader_test = get_dataloaders(args)
        self.net_mri, _ = get_network(args.net, args.num_classes)
        self.net_mri = self.net_mri.to(self.device)
        self.net_clip, self.tokenizer = get_network(args.encoder, args.num_classes)
        self.optimizer, self.scheduler = get_optimizer(args.optimizer, args.scheduler, self.net_mri.parameters(), lr=args.lr)
        self.cri = get_loss("combine")
        path_log = Path("runs")
        path_log.mkdir(parents=True, exist_ok=True)
        path_log_train = path_log / f"./runs/train_{args.run}"
        path_log_train.mkdir(parents=True, exist_ok=True)
        path_log_val = path_log / f"./runs/val_{args.run}"
        path_log_val.mkdir(parents=True, exist_ok=True)
        self.writer_train = SummaryWriter(log_dir=path_log_train.__str__())
        self.writer_val = SummaryWriter(log_dir=path_log_val.__str__())

    def train(self):
        loss_best = 1e4
        time_start = time.time()

        for epoch in range(self.args.epochs):
            loss, loss_co, loss_ce = self.train_epoch(epoch)
            print(f"Train loss: {loss.data.item()} || @ epoch {epoch}.")
            time_end = time.time()
            print(f"Training time: {time_end - time_start}")
            self.writer_train.add_scalar("LOSS", loss.data.item(), epoch)
            self.writer_train.add_scalar("LOSS_CE", loss_ce.item(), epoch)
            self.writer_train.add_scalar("LOSS_CO", loss_co.item(), epoch)
            if epoch % self.args.val_freq == 0 or epoch == self.args.epochs - 1:
                loss, loss_co, loss_ce = self.validate_epoch(epoch)
                print(f"Val loss: {loss.data.item()}, Constrastive loss: {loss_co}, CE loss: {loss_ce} || @ epoch {epoch}.")
                time_end = time.time()
                print(f"Training time: {time_end - time_start}")
                self.writer_val.add_scalar("LOSS", loss.data.item(), epoch)
                self.writer_val.add_scalar("LOSS_CE", loss_ce.item(), epoch)
                self.writer_val.add_scalar("LOSS_CO", loss_co.item(), epoch)
                if loss_best > loss:
                    loss_best = loss.item()
        print("best loss: ", loss_best)
        self.writer_train.close()
        self.writer_val.close()

    def validate_epoch(self, epoch):
        self.net_mri.eval()
        loss_epoch = 0.
        loss1_epoch = 0.
        loss2_epoch = 0.
        num_val = len(self.loader_val)
        with tqdm(total=num_val, desc=f'Epoch {epoch}', unit="image", leave=False) as pbar:
            for i, data in enumerate(self.loader_train):
                mri = data["mri"].to(dtype=torch.float32, device=self.device)
                tgt = data["tgt"].to(dtype=torch.long, device=self.device)
                img = data["img"].to(dtype=torch.float32, device="cpu")
                txt = self.tokenizer([f"a photo of {data['cls_name']}"])
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    feat_mri_img, feat_mri_txt, out = self.net_mri(mri)
                    feat_img = self.net_clip.encode_image(img)
                    feat_txt = self.net_clip.encode_text(txt)
                    out = out.softmax(-1)
                    loss, loss1, loss2 = self.cri(feat_mri_img, feat_img.to(self.device), feat_mri_txt, feat_txt.to(self.device), out, tgt, data["contrastive"].to(self.device))
                    loss_epoch += loss
                    loss1_epoch += loss1
                    loss2_epoch += loss2
            pbar.update()
        return loss_epoch / num_val, loss1_epoch / num_val, loss2_epoch / num_val

    def train_epoch(self, epoch):
        self.net_mri.train()
        self.optimizer.zero_grad()
        loss_epoch = 0.
        loss1_epoch = 0.
        loss2_epoch = 0.
        num_train = len(self.loader_train)
        with tqdm(total=num_train, desc=f'Epoch {epoch}', unit="image", leave=False) as pbar:
            for i, data in enumerate(self.loader_train):
                mri = data["mri"].to(dtype=torch.float32, device=self.device)
                tgt = data["tgt"].to(dtype=torch.long, device=self.device)
                img = data["img"].to(dtype=torch.float32, device="cpu")
                txt = self.tokenizer([f"a photo of {data['cls_name']}"])
                feat_mri_img, feat_mri_txt, out = self.net_mri(mri)
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    feat_img = self.net_clip.encode_image(img)
                    feat_txt = self.net_clip.encode_text(txt)
                out = out.softmax(-1)
                loss, loss1, loss2 = self.cri(feat_mri_img, feat_img.to(self.device), feat_mri_txt, feat_txt.to(self.device), out, tgt, data["contrastive"].to(self.device))
                loss_epoch += loss
                loss1_epoch += loss1
                loss2_epoch += loss2
                loss.backward()
                self.optimizer.step()
            pbar.update()
        return loss_epoch / num_train, loss1_epoch / num_train, loss2_epoch / num_train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-data_path", type=str, default="./data/processed_fmri")
    parser.add_argument("-dataset", type=str, default="BOLD5000")
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-gpu_ids", type=str, default="0")
    parser.add_argument("-net", type=str, default="mri")
    parser.add_argument("-num_classes", type=int, default=4)
    parser.add_argument("-encoder", type=str, default="clip")
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-optimizer", type=str, default="adam")
    parser.add_argument("-scheduler", type=str, default="step")
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-val_freq", type=int, default=1)
    parser.add_argument("-run", type=int, default=1)
    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    trainer = ModelMRI(args)
    trainer.train()
