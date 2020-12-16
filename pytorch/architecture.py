import math
import torchvision
import borrowed.spectral_norm as SN
from collections import OrderedDict
import torch
import torch.nn as nn
import copy


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            "normalization layer [{:s}] is not found".format(norm_type)
        )
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            "padding layer [{:s}] is not implemented".format(pad_type)
        )
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = "Identity .. \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device("cuda"))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    mode="CNA",
):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ["CNA", "NAC", "CNAC"], "Wong conv mode [{:s}]".format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = act(act_type) if act_type else None
    if "CNA" in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == "NAC":
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class minibatch_std_concat_layer(nn.Module):
    def __init__(self, averaging="all"):
        super(minibatch_std_concat_layer, self).__init__()
        self.averaging = averaging.lower()
        if "group" in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ["all", "flat", "spatial", "none", "gpool"], (
                "Invalid averaging mode" % self.averaging
            )
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(
            torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8
        )

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == "all":
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == "spatial":
            if len(shape) == 4:
                vals = mean(vals, axis=[2, 3], keepdim=True)
        elif self.averaging == "none":
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == "gpool":
            if len(shape) == 4:
                vals = mean(x, [0, 2, 3], keepdim=True)
        elif self.averaging == "flat":
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:
            target_shape[1] = self.n
            vals = vals.view(
                self.n, self.shape[1] / self.n, self.shape[2], self.shape[3]
            )
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(
        self,
        in_nc,
        mid_nc,
        out_nc,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="relu",
        mode="CNA",
        res_scale=1,
    ):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(
            in_nc,
            mid_nc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
        )
        if mode == "CNA":
            act_type = None
        if mode == "CNAC":
            act_type = None
            norm_type = None
        conv1 = conv_block(
            mid_nc,
            out_nc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
        )
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(
        self,
        nc,
        kernel_size=3,
        gc=32,
        stride=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        gaussian_noise=True,
    ):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.noise = GaussianNoise() if gaussian_noise else None
        self.conv1x1 = conv1x1(nc, gc)
        self.conv1 = conv_block(
            nc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        self.conv2 = conv_block(
            nc + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        self.conv3 = conv_block(
            nc + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        self.conv4 = conv_block(
            nc + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        if mode == "CNA":
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(
            nc + 4 * gc,
            nc,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return self.noise(x5.mul(0.2) + x)


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        nc,
        kernel_size=3,
        gc=32,
        stride=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
    ):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode
        )

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


def pixelshuffle_block(
    in_nc,
    out_nc,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(
        in_nc,
        out_nc * (upscale_factor ** 2),
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=None,
        act_type=None,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(
    in_nc,
    out_nc,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    mode="nearest",
):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=norm_type,
        act_type=act_type,
    )
    return sequential(upsample, conv)


class SRResNet(nn.Module):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        upscale=4,
        norm_type="batch",
        act_type="relu",
        mode="NAC",
        res_scale=1,
        upsample_mode="upconv",
    ):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [
            ResNetBlock(
                nf,
                nf,
                nf,
                norm_type=norm_type,
                act_type=act_type,
                mode=mode,
                res_scale=res_scale,
            )
            for _ in range(nb)
        ]
        LR_conv = conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )

        if upsample_mode == "upconv":
            upsample_block = upconv_blcok
        elif upsample_mode == "pixelshuffle":
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]
        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*resnet_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1
        )

    def forward(self, x):
        x = self.model(x)
        return x


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
    ):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [
            RRDB(
                nf,
                kernel_size=3,
                gc=32,
                stride=1,
                bias=True,
                pad_type="zero",
                norm_type=norm_type,
                act_type=act_type,
                mode="CNA",
            )
            for _ in range(nb)
        ]
        LR_conv = conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )

        if upsample_mode == "upconv":
            upsample_block = upconv_blcok
        elif upsample_mode == "pixelshuffle":
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]
        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*rb_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1
        )

    def forward(self, x):
        x = self.model(x)
        return x


# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(
        self, in_nc, base_nf, norm_type="batch", act_type="leakyrelu", mode="CNA"
    ):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = conv_block(
            in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode
        )
        conv1 = conv_block(
            base_nf,
            base_nf,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 64, 64
        conv2 = conv_block(
            base_nf,
            base_nf * 2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv3 = conv_block(
            base_nf * 2,
            base_nf * 2,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 32, 128
        conv4 = conv_block(
            base_nf * 2,
            base_nf * 4,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv5 = conv_block(
            base_nf * 4,
            base_nf * 4,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 16, 256
        conv6 = conv_block(
            base_nf * 4,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv7 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 8, 512
        conv8 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv9 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 4, 512
        self.features = sequential(
            conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(
        self, in_nc, base_nf, norm_type="batch", act_type="leakyrelu", mode="CNA"
    ):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = conv_block(
            in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode
        )
        conv1 = conv_block(
            base_nf,
            base_nf,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 48, 64
        conv2 = conv_block(
            base_nf,
            base_nf * 2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv3 = conv_block(
            base_nf * 2,
            base_nf * 2,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 24, 128
        conv4 = conv_block(
            base_nf * 2,
            base_nf * 4,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv5 = conv_block(
            base_nf * 4,
            base_nf * 4,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 12, 256
        conv6 = conv_block(
            base_nf * 4,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv7 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 6, 512
        conv8 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv9 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 3, 512
        self.features = sequential(
            conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(
        self, in_nc, base_nf, norm_type="batch", act_type="leakyrelu", mode="CNA"
    ):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = conv_block(
            in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode
        )
        conv1 = conv_block(
            base_nf,
            base_nf,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 96, 64
        conv2 = conv_block(
            base_nf,
            base_nf * 2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv3 = conv_block(
            base_nf * 2,
            base_nf * 2,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 48, 128
        conv4 = conv_block(
            base_nf * 2,
            base_nf * 4,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv5 = conv_block(
            base_nf * 4,
            base_nf * 4,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 24, 256
        conv6 = conv_block(
            base_nf * 4,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv7 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 12, 512
        conv8 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv9 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 6, 512
        conv10 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv11 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 3, 512
        self.features = sequential(
            conv0,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv7,
            conv8,
            conv9,
            conv10,
            conv11,
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_layer=34,
        use_bn=False,
        use_input_norm=True,
        device=torch.device("cpu"),
    ):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        self.features = nn.Sequential(
            *list(model.features.children())[: (feature_layer + 1)]
        )
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device("cpu")):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        self.features = nn.Sequential(*list(model.children())[:8])
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_layer=34,
        use_bn=False,
        use_input_norm=True,
        device=torch.device("cpu"),
    ):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load("../experiments/pretrained_models/VGG16minc_53.pth"), strict=True
        )
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output
