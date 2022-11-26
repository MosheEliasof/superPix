import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from skimage.segmentation._slic import _enforce_label_connectivity_cython
import kornia
from utils import ContourLoss, TV_smoothness, imageFromSpix, mutual_information, reconstruction, smoothness

class ConvInRelu(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride

        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(self.in_c, self.out_c, self.kernel_size,
                      stride=self.stride, bias=False),
            nn.InstanceNorm2d(self.out_c, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class UpConvInRelu(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride

        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.ConvTranspose2d(self.in_c, self.out_c, self.kernel_size, stride=self.stride, bias=False, padding=self.kernel_size // 2,
                               output_padding=self.kernel_size // 2),
            nn.InstanceNorm2d(self.out_c, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=100):
        super(ASPP, self).__init__()
        self.out_channels = out_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels,
                      3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Encoder(nn.Module):
    def __init__(self, in_c=5, n_filters=32, n_layers=5):
        super().__init__()
        self.original_in_c = in_c
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(ConvInRelu(in_c, n_filters << i))
            in_c = n_filters << i

        self.layers = nn.Sequential(*self.layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return x


class ASPP_SuperPix(nn.Module):
    def __init__(self, 
                    atrous_rates=[2, 4, 8], in_c=5, 
                    n_spix=100, n_filters=32, n_layers=4,
                    use_recons=True, use_last_inorm=True,
                    image_size=512, useTV=True, useSuperPixRecons=True,
                    use_spix_recons=True):
        super(ASPP_SuperPix, self).__init__()
        self.in_channels = in_c
        self.atrous_rates = atrous_rates
        self.n_spix = n_spix
        self.enc_out_channel = n_filters * (2 ** (n_layers - 2))
        self.out_channels = n_spix
        self.use_last_inorm = use_last_inorm
        self.use_recons = use_recons
        self.use_spix_recons = use_spix_recons
        self.useSuperPixRecons = useSuperPixRecons
        self.image_size = image_size
        self.useTV = useTV

        if use_last_inorm:
            self.norm = nn.InstanceNorm2d(n_spix, affine=True)

        if use_recons:
            self.out_channels += 3

        self.encoder = Encoder(in_c, n_filters, n_layers)
        self.decoder = ASPP(in_channels=2 * self.enc_out_channel, atrous_rates=atrous_rates,
                            out_channels=self.enc_out_channel)
        self.final = nn.Sequential(
            nn.Conv2d(self.enc_out_channel, self.enc_out_channel,
                      3, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.enc_out_channel, self.out_channels,
                      1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.encoder(x)
        lap_x = kornia.filters.laplacian(x, kernel_size=3)
        x = torch.cat([x, lap_x], dim=1)
        x = self.decoder(x)
        x = self.final(x)
        if self.use_recons:
            recons, spix = x[:, :3], x[:, 3:]
        else:
            recons = None

        if self.use_last_inorm:
            spix = self.norm(spix)
        return spix, recons

    def __preprocess(self, image, device="cuda"):
        image = image.permute(2, 0, 1).float()[None]
        h, w = image.shape[-2:]
        coord = torch.stack(torch.meshgrid(
            torch.arange(h), torch.arange(w),indexing='ij')).float()[None]
        input = torch.cat([image, coord], 1).to(device)
        data_mean = input.mean((2, 3), keepdim=False)
        data_std = input.std((2, 3), keepdim=False)
        input = (input - input.mean((2, 3), keepdim=True)) / \
            input.std((2, 3), keepdim=True)
        return input, data_mean, data_std

    def optimize(self, image, n_iter=500, lr=1e-2, lam=2, alpha=2, beta=2, device="cuda", usecontourLoss=True):
        input, data_mean, data_std = self.__preprocess(image, device)
        optimizer = optim.Adam(self.parameters(), lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.1)
        input_image = input[:, :3].clone()
        for i in range(n_iter):
            optimizer.zero_grad()
            spix, recons = self.forward(input)
            mean_img = imageFromSpix(spix, input)
            loss_mi = mutual_information(spix, lam)
            if self.useTV:
                loss_smooth = TV_smoothness(spix, input)
            else:
                loss_smooth = smoothness(spix, input)
                if usecontourLoss:
                    if self.useSuperPixRecons:
                        loss_smooth += 0.5 * (
                            ContourLoss(mean_img, input_image) + ContourLoss(recons, input_image))
                    else:
                        loss_smooth += 0.5 * \
                            self.ContourLoss(recons, input_image)
            loss = loss_mi + alpha * loss_smooth
            if self.use_recons:
                if self.useSuperPixRecons:
                    loss_recon = reconstruction(mean_img,
                                                     input[:, :3]) + reconstruction(recons, input[:, :3])
                else:
                    loss_recon = reconstruction(recons, input[:, :3])
                loss = loss + beta * loss_recon
            loss.backward()
            optimizer.step()
            scheduler.step()

            print(
                f"[{i + 1}/{n_iter}] loss {loss.item()}, loss_mi {loss_mi.item()},"
                f"loss_smooth {loss_smooth.item()}, loss_recon {loss_recon.item()}",
                flush=True)

        return self.calc_spixel(image, device), mean_img.detach().cpu(), \
            input_image.detach().cpu(), data_mean[0, :3].cpu(), data_std[0,
                                                                         :3].cpu(), recons.detach().cpu()

    def calc_spixel(self, image, device="cuda"):
        input, _, _ = self.__preprocess(image, device)
        spix, recons = self.forward(input)

        spix = spix.argmax(1).squeeze().to("cpu").detach().numpy()

        segment_size = spix.size / self.n_spix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        spix = _enforce_label_connectivity_cython(
            spix[None], min_size, max_size)[0]

        return spix
