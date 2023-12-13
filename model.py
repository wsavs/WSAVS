import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import torch.distributed as dist

import resnets
from avfusion import *


EMBED_DIM = {'resnet50': 2048}


class WSAVS(nn.Module):
    def __init__(self, tau, latent_dim, dropout_img=0.9, dropout_aud=0., imgnet_type='resnet50', audnet_type='resnet50', avfusion_stages=[0,1,2,3]):
        super(WSAVS, self).__init__()
        self.tau = tau

        # Vision model
        self.img_enc = self.build_resnet_encoder(imgnet_type, in_chans=3, pretrained='supervised')
        self.img_proj1 = nn.Sequential(
            nn.LayerNorm(EMBED_DIM[imgnet_type]//8),
            nn.Dropout(p=dropout_img),
            nn.Linear(EMBED_DIM[imgnet_type]//8, latent_dim)
        )
        self.img_proj2 = nn.Sequential(
            nn.LayerNorm(EMBED_DIM[imgnet_type]//4),
            nn.Dropout(p=dropout_img),
            nn.Linear(EMBED_DIM[imgnet_type]//4, latent_dim)
        )
        self.img_proj3 = nn.Sequential(
            nn.LayerNorm(EMBED_DIM[imgnet_type]//2),
            nn.Dropout(p=dropout_img),
            nn.Linear(EMBED_DIM[imgnet_type]//2, latent_dim)
        )
        self.img_proj4 = nn.Sequential(
            nn.LayerNorm(EMBED_DIM[imgnet_type]),
            nn.Dropout(p=dropout_img),
            nn.Linear(EMBED_DIM[imgnet_type], latent_dim)
        )

        # Audio model
        self.aud_enc = self.build_resnet_encoder(audnet_type, in_chans=1, pretrained='supervised')
        self.aud_proj = nn.Sequential(
            nn.LayerNorm(EMBED_DIM[audnet_type]),
            nn.Dropout(p=dropout_aud),
            nn.Linear(EMBED_DIM[audnet_type], latent_dim)
        )

        # Audio-visual fusion
        self.avfusion_stages = avfusion_stages
        for i in [0, 1, 2, 3]:
            setattr(self, f"avfusion_b{i+1}", AVFusionModule(in_channels=latent_dim, mode='dot'))

        # Decoder blocks
        self.path4 = FeatureFusionBlock(latent_dim)
        self.path3 = FeatureFusionBlock(latent_dim)
        self.path2 = FeatureFusionBlock(latent_dim)
        self.path1 = FeatureFusionBlock(latent_dim)

        # output conv for predict the final map
        self.output_conv = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        # Initialize weights (except pretrained visual model)
        for net in [self.img_proj1, self.img_proj2, self.img_proj3, self.img_proj4, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    w = m.weight.data
                    torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                    nn.init.constant_(m.bias, 0)
    
    def build_resnet_encoder(self, resnet_type, in_chans, pretrained=''):
        model = resnets.__dict__[resnet_type](in_chans=in_chans, pretrained=pretrained)
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()
        return model

    def forward_img_features(self, img_enc, img_projs, image):
        _, imgs = img_enc(image, return_embs=True)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].flatten(2).permute(0, 2, 1)
            imgs[i] = img_projs[i](imgs[i])                      # B x HW x C
        return imgs

    def forward_aud_features(self, aud_enc, aud_proj, audio):
        aud = aud_enc(audio)
        aud = aud.flatten(2).max(dim=2)[0]       # Max pool
        aud = aud_proj(aud)                      # B x C
        return aud

    def pre_reshape_for_avfusion(self, x):
        # x: [B, C, H, W]
        _, C, H, W = x.shape
        x = x.reshape(-1, 1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, 1, H, W]
        return x

    def post_reshape_for_avfusion(self, x):
        # x: [B, C, 1, H, W]
        # return: [B, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, 1, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def avfusion_va(self, x, audio, stage):
        # x: visual, [B*1, C=256, H, W]
        # audio: [B*1, 128]
        avfusion_b = getattr(self, f'avfusion_b{stage+1}')
        audio = audio.view(-1, 1, audio.shape[-1]) # [B, 1, 128]
        x = self.pre_reshape_for_avfusion(x) # [B, C, 1, H, W]
        x, a = avfusion_b(x, audio) # [B, C, 1, H, W], [B, 1, C]
        x = self.post_reshape_for_avfusion(x) # [B*1, C, H, W]
        return x, a

    def msmil_loss(self, img, aud, i):
        img = nn.functional.normalize(img, dim=-1)
        aud = nn.functional.normalize(aud, dim=-1)
        logits = torch.einsum('ntc,mc->nmt', img, aud)            # Bv x Ba x Tv
        max_logits = logits.max(dim=-1)[0] / self.tau             # Bv x Ba
        labels = torch.arange(img.shape[0]).long().to(img.device)
        loss = self.tau * (F.cross_entropy(max_logits, labels) + F.cross_entropy(max_logits.permute(1, 0), labels))
        with torch.no_grad():
            pred_maps = torch.einsum('ntc,mc->nmt', img, aud).view(img.shape[0], aud.shape[0], 7*(2**i), 7*(2**i))            # B x B x 7 x 7
        return loss, pred_maps

    def forward(self, image, audio, pseudo_mask=None, mode='train'):
        # Image
        imgs = self.forward_img_features(self.img_enc, [self.img_proj1, self.img_proj2, self.img_proj3, self.img_proj4], image)
        # print('imgs:', imgs[0].shape)     # [B, HW, C]

        # Audio
        aud = self.forward_aud_features(self.aud_enc, self.aud_proj, audio)
        # print('aud:', aud.shape)          # [B, C]

        feature_map_list = [None] * 4
        a_fea_list = [None] * 4

        for i in [0, 1, 2, 3]:
            avfusion_count = 0
            img_feat = imgs[i].view(imgs[i].shape[0], imgs[i].shape[-1], 7*(2**(3-i)), 7*(2**(3-i)))       
            conv_feat = torch.zeros_like(img_feat).cuda()
            # print('stage:', i)  
            # print('imgs[i]:', imgs[i].shape)      # [B, HW, C]
            # print('aud:', aud.shape)              # [B, C]
            # print('img_feat:', img_feat.shape)    # [B, C, H, W]
            conv_feat_va, a_fea = self.avfusion_va(img_feat, aud, stage=i)
            conv_feat += conv_feat_va
            avfusion_count += 1
            a_fea_list[i] = a_fea
            conv_feat /= avfusion_count
            feature_map_list[i] = conv_feat # update features of stage-i which conduct TPAVI

        conv4_feat = self.path4(feature_map_list[3])            # B x 512 x 14 x 14
        conv43 = self.path3(conv4_feat, feature_map_list[2])    # B x 512 x 28 x 28
        conv432 = self.path2(conv43, feature_map_list[1])       # B x 512 x 56 x 56
        conv4321 = self.path1(conv432, feature_map_list[0])     # B x 512 x 112 x 112

        pred = self.output_conv(conv4321)   # B x 1 x 224 x 224
        pred = torch.sigmoid(pred).squeeze(1)

        if mode == 'train':
            # Compute msmil loss
            msmil_loss = 0
            for i in range(len(imgs)):
                if i in self.avfusion_stages:
                    loss, _ = self.msmil_loss(imgs[i], aud, 3-i)
                    msmil_loss += loss

            # Compute pixel loss
            pixel_loss = torch.nn.BCELoss()(pred, pseudo_mask.float())

            return msmil_loss, pixel_loss, pred
        
        elif mode == 'test':

            return pred


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return tensor
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output