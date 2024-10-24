import torch
import torch.nn as nn
import open_clip


__all__ = ["MRIModel"]


class MRIModel(nn.Module):
    def __init__(
            self,
            num_classes: int = 4
    ) -> None:
        super().__init__()
        self.proj_img = nn.Sequential(
            nn.Linear(64292,512, bias=True),
            nn.LayerNorm(512),
            nn.Mish(inplace=True),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=8),
                num_layers=4,
                norm=nn.LayerNorm(512),
            )
        )
        self.proj_txt = nn.Sequential(
            nn.Linear(64292, 512, bias=True),
            nn.LayerNorm(512),
            nn.Mish(inplace=True),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=8),
                num_layers=4,
                norm=nn.LayerNorm(512),
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Mish(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, mri: torch.Tensor):
        feat_mri_img = self.proj_img(mri)
        feat_mri_txt= self.proj_txt(mri)
        feat_mri_fuse = torch.cat([feat_mri_img, feat_mri_txt], dim=-1)
        out = self.classifier(feat_mri_fuse)
        return feat_mri_img, feat_mri_txt, out


# if __name__ == "__main__":
#     clip_model_pretrain, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr')
#     cls_name = "animal"
#     model = MRIModel().cuda()
#     clip_tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
#     mri = torch.randn((1, 64292)).cuda()
#     img = torch.randn((1, 3, 224, 224))
#     with torch.no_grad(), torch.amp.autocast("cuda"):
#          feat_img = clip_model_pretrain.encode_image(img)
#          txt = clip_tokenizer([f"a photo of {cls_name}"])
#          feat_txt = clip_model_pretrain.encode_text(txt)
#
#     print(txt.shape)
#     mri_img, mri_txt, out = model(mri)
#     print(mri_img.shape, mri_txt.shape)
#     print(feat_img.shape, feat_txt.shape)
