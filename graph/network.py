import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import open_clip

__all__ = ["CLIPTuner"]


class CLIPTuner(nn.Module):
    def __init__(self, model_name: str = "ViT-B/32"):
        super().__init__()
        self.clip_model_pretrain = open_clip.create_model(model_name, pretrained="laion2b_s34b_b79k")
        # self.clip_model_mri, _ = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.proj_img = nn.Sequential(
            nn.Linear(64292,150528, bias=True),
            nn.LayerNorm(150528),
            nn.Mish(inplace=True),
            Rearrange("b (c h w) -> b c h w", c=3, h=224, w=224)
        )
        self.proj_text = nn.Sequential(
            nn.Linear(64292, 256, bias=True),
            nn.LayerNorm(256),
            nn.Mish(inplace=True)
        )



    def forward(self, mri: torch.Tensor, img: torch.Tensor, class_name: str):
        with torch.no_grad(), torch.cuda.amp.autocast():
            feat_img = self.clip_model_pretrain.encode_image(img)
            txt = self.clip_tokenizer([f"a photo of {class_name}"])
            feat_txt = self.clip_model_pretrain.encode_text(txt)
            feat_img /= feat_img.norm(dim=-1, keepdim=True)
            feat_txt /= feat_txt.norm(dim=-1, keepdim=True)
        # feat_mri_img = self.proj_img(mri)
        # feat_mri_img = self.clip_model_mri.encode_image(feat_mri_img)
        # feat_mri_txt= self.proj_txt(mri)
        # feat_mri_txt = self.clip_model_mri.encode_text(feat_mri_txt)
        # print(feat_mri_txt.shape, feat_mri_img.shape)
        print(feat_txt.shape, feat_img.shape)
        return None
        # return feat_mri_txt.shape, feat_mri_img.shape


if __name__ == "__main__":
    model = CLIPTuner()
    mri = torch.randn((1, 64292))
    img = torch.randn((1, 3, 224, 224))
    txt = "animal"
    model(mri, img, txt)
