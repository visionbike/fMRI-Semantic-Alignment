import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import open_clip


__all__ = ["CLIPTuner"]


class CLIPTuner(nn.Module):
    def __init__(
            self,
            num_classes: int = 4
    ) -> None:
        super().__init__()
        self.clip_model_mri, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr', reparameterize=False)
        self.proj_img = nn.Sequential(
            nn.Linear(64292,150528, bias=True),
            nn.LayerNorm(150528),
            nn.Mish(inplace=True),
            Rearrange("b (c h w) -> b c h w", c=3, h=224, w=224)
        )
        self.proj_text = nn.Sequential(
            nn.Linear(64292, 77, bias=True),
            nn.LayerNorm(77),
            nn.Mish(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_model_mri.projection_dim * 2, self.clip_model_mri.projection_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(self.clip_model_mri.projection_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.clip_model_mri.projection_dim, num_classes)
        )

    def forward(self, mri: torch.Tensor):
        feat_mri_img = self.proj_img(mri)
        feat_mri_img = self.clip_model_mri.encode_image(feat_mri_img)
        feat_mri_txt= self.proj_txt(mri)
        feat_mri_txt = self.clip_model_mri.encode_text(feat_mri_txt)
        feat_mri_fuse = torch.cat([feat_mri_img, feat_mri_txt], dim=-1)
        out = self.classifier(feat_mri_fuse)
        return feat_mri_img, feat_mri_txt, out


if __name__ == "__main__":
    clip_model_pretrain, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr')
    cls_name = "animal"
    model = CLIPTuner()
    clip_tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
    mri = torch.randn((1, 64292))
    img = torch.randn((1, 3, 224, 224))
    with torch.no_grad(), torch.cuda.amp.autocast():
        feat_img = clip_model_pretrain.encode(img)
        txt = clip_tokenizer([f"a photo of {cls_name}"])
        feat_txt = clip_model_pretrain.encode_text(txt)

    print(txt.shape)
    mri_img, mri_txt, out = model(mri)
    print(mri_img.shape, mri_txt.shape)
    print(feat_img.shape, feat_txt.shape)
