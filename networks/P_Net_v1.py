from .unet_part import *

class Strcutre_Extraction_Network(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Strcutre_Extraction_Network, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        # unet, seg mask
        x1 = self.inc(x)                # 256,256,64
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        seg_latent_feat = self.down4(x4)

        x = self.up1(seg_latent_feat, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        seg_structure_feat = self.up4(x, x1)
        seg_mask = self.outc(seg_structure_feat)

        # only the seg_mask used in the paper
        return seg_mask, seg_structure_feat, seg_latent_feat


class Image_Reconstruction_Network(nn.Module):
    def __init__(self, in_ch, modality, ablation_mode):
        super(Image_Reconstruction_Network, self).__init__()
        self.modality = modality
        self.ablation_mode = ablation_mode
        # input: 224 * 224 * in_ch
        """
        ablation study mode
        """
        # 0: output_structure                       (1 feature)
        # 2: image, i.e. auto-encoder               (1 feature)
        # 4: output_structure + image               (2 features)

        self.image_encoder = nn.Sequential(
            inconv(in_ch, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512),
            down(512, 512)
        )

        self.inc = inconv(1, 64)
        self.seg_encoder_down1 = down(64, 128)
        self.seg_encoder_down2 = down(128, 256)
        self.seg_encoder_down3 = down(256, 512)
        self.seg_encoder_down4 = down(512, 512)

        """
        Difference: number channel of decoder
        """
        if ablation_mode == 0:
            # output_structure
            self.up1 = up(1024, 256)
            self.up2 = up(512, 128)
            self.up3 = up(256, 64)
            self.up4 = up(128, 64)
        elif ablation_mode == 2:
            # 1 feature
            self.up1 = up_wo_skip(512, 512)
            self.up2 = up_wo_skip(512, 256)
            self.up3 = up_wo_skip(256, 128)
            self.up4 = up_wo_skip(128, 64)
        else:
            # sugar net
            self.up1 = up(1536, 256)
            self.up2 = up(512, 128)
            self.up3 = up(256, 96)
            self.up4 = up_wo_skip(96, 64)

        self.outc = outconv(64, in_ch)

    def forward(self, image, seg_mask, seg_latent_feat=None):
        ### old version: seg_latent_feat=None
        ### in new version, we don't use seg_latent_feat

        # image
        rec_latent_feat_image = self.image_encoder(image)

        if self.modality == 'oct':
            seg_mask = torch.argmax(seg_mask, dim=1).unsqueeze(dim=1)
            seg_mask = (seg_mask / 11).clamp(0, 1).float()

        # structure
        x0 = self.inc(seg_mask)
        x1 = self.seg_encoder_down1(x0)  # 112 * 112 * 128
        x2 = self.seg_encoder_down2(x1)  # 56 * 56 * 256
        x3 = self.seg_encoder_down3(x2)  # 28 * 28 * 512
        x4 = self.seg_encoder_down4(x3)  # 14 * 14 * 512

        rec_latent_feat_struc = x4
        # 14 * 14 * 1536

        """
        feature fusion with different mode;
        encoder is different
        """
        if self.ablation_mode == 0:
            # only output_structure
            rec_latent_feat = rec_latent_feat_struc
            x = self.up1(rec_latent_feat, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.up4(x, x0)
        elif self.ablation_mode == 2:
            # 1 feature without skip
            rec_latent_feat = rec_latent_feat_image
            x = self.up1(rec_latent_feat)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
        elif self.ablation_mode == 4:
            # 4: output_structure + image
            # 512 + 512
            rec_latent_feat = torch.cat([rec_latent_feat_struc, rec_latent_feat_image], dim=1)
            x = self.up1(rec_latent_feat, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.up4(x)
        else:
            # ablation study for different skip-connections
            raise NotImplementedError('error')

        x = self.outc(x)

        return x
