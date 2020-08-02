from .unet_part import *


class UNet_4mp(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_4mp, self).__init__()
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet_5mp(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_5mp, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 1024)

        self.up0 = up(2048, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)

        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class RecAE_4mp(nn.Module):
    def __init__(self, n_channels, out_channels=1):
        super(RecAE_4mp, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

        self.up_ae_1 = up_wo_skip(1024, 512)
        self.up_ae_2 = up_wo_skip(512, 256)
        self.up_ae_3 = up_wo_skip(256, 128)
        self.up_ae_4 = up_wo_skip(128, 64)

        self.outc_ae = outconv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)        # 64
        x2 = self.down1(x1)     # 128
        x3 = self.down2(x2)     # 256
        x4 = self.down3(x3)     # 512
        x5 = self.down4(x4)     # 1024

        x_rec = self.up_ae_1(x5)     # 512
        x_rec = self.up_ae_2(x_rec)     # 256
        x_rec = self.up_ae_3(x_rec)     # 128
        x_rec = self.up_ae_4(x_rec)
        x_rec = self.outc_ae(x_rec)

        return x_rec


class RecAE_5mp(nn.Module):
    def __init__(self, n_channels, out_channels):
        super(RecAE_5mp, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 2048)

        self.up_ae_0 = up_wo_skip(2048, 1024)
        self.up_ae_1 = up_wo_skip(1024, 512)
        self.up_ae_2 = up_wo_skip(512, 256)
        self.up_ae_3 = up_wo_skip(256, 128)
        self.up_ae_4 = up_wo_skip(128, 64)

        self.outc_ae = outconv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)        # 64
        x2 = self.down1(x1)     # 128
        x3 = self.down2(x2)     # 256
        x4 = self.down3(x3)     # 512
        x5 = self.down4(x4)     # 1024
        x6 = self.down5(x5)     # 1024

        x5_de = self.up_ae_0(x6)        # 1024
        x4_de = self.up_ae_1(x5_de)     # 512
        x3_de = self.up_ae_2(x4_de)     # 256
        x2_de = self.up_ae_3(x3_de)     # 128
        x_rec = self.up_ae_4(x2_de)
        x_rec = self.outc_ae(x_rec)

        return x_rec

# TO_Delete
class Seg_UNet_Rec_AE(nn.Module):
    def __init__(self, mode, n_channels=1, n_classes=1):
        super(Seg_UNet_Rec_AE, self).__init__()
        assert mode in ['seg', 'rec', 'seg_rec'], 'mode error'
        self.mode = mode
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)

        self.up_ae_1 = up_wo_skip(512, 256)
        self.up_ae_2 = up_wo_skip(256, 128)
        self.up_ae_3 = up_wo_skip(128, 64)
        self.up_ae_4 = up_wo_skip(64, 64)

        self.outc = outconv(64, n_classes)
        self.outc_ae = outconv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.mode == 'seg':
            x_seg = self.up1(x5, x4)
            x_seg = self.up2(x_seg, x3)
            x_seg = self.up3(x_seg, x2)
            x_seg = self.up4(x_seg, x1)
            x_seg = self.outc(x_seg)
            return x_seg
        elif self.mode == 'rec':
            x_rec = self.up_ae_1(x5)
            x_rec = self.up_ae_2(x_rec)
            x_rec = self.up_ae_3(x_rec)
            x_rec = self.up_ae_4(x_rec)
            x_rec = self.outc_ae(x_rec)
            return x_rec
        else:
            x_seg = self.up1(x5, x4)
            x_seg = self.up2(x_seg, x3)
            x_seg = self.up3(x_seg, x2)
            x_seg = self.up4(x_seg, x1)
            x_seg = self.outc(x_seg)

            x_rec = self.up_ae_1(x5)
            x_rec = self.up_ae_2(x_rec)
            x_rec = self.up_ae_3(x_rec)
            x_rec = self.up_ae_4(x_rec)
            x_rec = self.outc_ae(x_rec)
            return x_seg, x_rec


def main():
    pass


if __name__ == '__main__':
    main()
