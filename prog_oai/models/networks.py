import logging
import math

import coloredlogs
import torch
import torch.nn as nn
from einops import rearrange

coloredlogs.install()

__all__ = ['make_model', 'make_global_pool']


def make_model(model_name, backbone_name, n_classes, use_bn, pretrained, drop_rate, n_out_features,
               global_pool_name='gap', ssp_start=None, use_ssp=False, n_input_imgs=1,
               max_depth=5, gp_list=None, extra_classes=5, input_3x3=False):
    logging.info(f'Create model {model_name} with global pooling {global_pool_name}, use_bn = {use_bn}')
    if model_name == "kneeimage":
        return KneeImageNet(n_out_features=n_out_features, drop_rate=drop_rate, backbone_name=backbone_name,
                            use_bn=use_bn, pretrained=pretrained, n_input_imgs=n_input_imgs,
                            global_pool_name=global_pool_name, max_depth=max_depth, input_3x3=input_3x3,
                            n_classes=n_classes, n_extra_classes=extra_classes)
    elif model_name == "sspsam":
        return SSPSAMNet(drop_rate=drop_rate, backbone_name=backbone_name, use_bn=use_bn, pretrained=pretrained,
                         extra_classes=extra_classes,
                         global_pool_name=global_pool_name, ssp_start=ssp_start, max_depth=max_depth, use_ssp=use_ssp,
                         gp_list=gp_list, input_3x3=input_3x3)
    elif model_name == "ensamnet":
        return ENSAMNet(drop_rate=drop_rate, backbone_name=backbone_name, use_bn=use_bn, pretrained=pretrained,
                        global_pool_name=global_pool_name, ssp_start=ssp_start, max_depth=max_depth, use_ssp=use_ssp,
                        gp_list=gp_list, input_3x3=input_3x3)
    elif model_name == "mustnet":
        return MUSTNet(drop_rate=drop_rate, backbone_name=backbone_name, use_bn=use_bn, pretrained=pretrained,
                       n_classes=n_classes, extra_classes=extra_classes, gp_list=gp_list, input_3x3=input_3x3)
    else:
        logging.fatal(f'Not support model name {model_name}')
        assert False


def make_global_pool(pool_name, channels, sz, use_bn=False):
    if pool_name == "samh":
        return SAMH(channels, sz, use_bn)
    elif pool_name == "samv":
        return SAMV(channels, sz, use_bn)
    elif pool_name == "saml":
        return SAML(channels, sz, use_bn)
    elif pool_name == "samm":
        return SAMM(channels, sz, use_bn)
    elif pool_name == "sama":
        return SAMA(channels, sz, use_bn)
    elif pool_name == "samc":
        return SAMC(channels, sz, use_bn)
    elif pool_name == "gap":
        return nn.AvgPool2d(sz, sz)
    elif pool_name is None:
        return None
    else:
        logging.fatal(f'Not support SAM {pool_name}')
        assert False


def output_maxpool(input_shape, padding=0, dilation=1, ks=2, stride=2):
    return math.floor((input_shape + 2 * padding - dilation * (ks - 1) - 1) / stride + 1)


class SAMV(nn.Module):
    def __init__(self, channels, sz, use_bn=False):
        super().__init__()
        self.channels = channels
        self.output_shape = None
        # Pool VH
        self._sam_vh = nn.Sequential(nn.MaxPool2d((sz, 1), (sz, 1), 0),
                                     nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.MaxPool2d((1, sz), sz, 0))  # state size ndf * 4 x 1 x 1

    @property
    def output_channels(self):
        return self.channels

    def forward(self, x):
        return self._sam_vh(x)


class SAMH(nn.Module):
    def __init__(self, channels, sz, use_bn=False):
        super().__init__()
        self.channels = channels
        self.output_shape = None

        # Pool HV
        self._sam_hv = nn.Sequential(nn.MaxPool2d((1, sz), (1, sz), 0),
                                     nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.MaxPool2d((sz, 1), sz, 0))  # state size ndf * 4 x 1 x 1

    @property
    def output_channels(self):
        return self.channels

    def forward(self, x):
        return self._sam_hv(x)


class SAMC(nn.Module):
    def __init__(self, channels, sz, use_bn=False):
        super().__init__()
        self.channels = channels
        self.output_shape = None
        # Pool VH
        self._sam_vh = nn.Sequential(nn.MaxPool2d((sz, 1), (sz, 1), 0),
                                     nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.MaxPool2d((1, sz), sz, 0))  # state size ndf * 4 x 1 x 1

        # Pool HV
        self._sam_hv = nn.Sequential(nn.MaxPool2d((1, sz), (1, sz), 0),
                                     nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.MaxPool2d((sz, 1), sz, 0))  # state size ndf * 4 x 1 x 1

    @property
    def output_channels(self):
        return self.channels * 2

    def forward(self, x):
        x1 = self._sam_hv(x)
        x2 = self._sam_vh(x)
        x12 = torch.cat((x1, x2), dim=1)
        self.output_shape = x12.shape
        return x12


class SAMM(nn.Module):
    def __init__(self, channels, sz, drop_rate, use_bn=False):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=drop_rate)
        self.channels = channels
        # Pool VH
        self._sam_v = nn.Sequential(nn.MaxPool2d((sz, 1), (sz, 1), 0),
                                    nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                    nn.ReLU())  # state size B x C x H x 1

        # Pool HV
        self._sam_h = nn.Sequential(nn.MaxPool2d((1, sz), (1, sz), 0),
                                    nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                    nn.ReLU())  # state size B x C x 1 X W

    @property
    def output_channels(self):
        return self.channels

    def forward(self, x):
        v = self._sam_v(x)
        h = self._sam_h(x)

        o = torch.matmul(v, h)

        return o


class SAMA(nn.Module):
    def __init__(self, channels, sz, drop_rate, use_bn=False):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=drop_rate)
        self.channels = channels
        # Pool VH
        self._sam_v = nn.Sequential(nn.MaxPool2d((sz, 3), (sz, 1), padding=1),
                                    nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                    nn.ReLU())  # state size B x C x H x 1

        # Pool HV
        self._sam_h = nn.Sequential(nn.MaxPool2d((3, sz), (1, sz), padding=1),
                                    nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                    nn.ReLU())  # state size B x C x 1 X W

    @property
    def output_channels(self):
        return self.channels

    def forward(self, x):
        v = self._sam_v(x)
        h = self._sam_h(x)

        o = torch.matmul(h, v)
        return o


class SAML(nn.Module):
    def __init__(self, channels, sz, drop_rate, use_bn=False):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=drop_rate)
        self.channels = channels
        # Pool VH
        self._sam_vh = nn.Sequential(nn.MaxPool2d((sz, 1), (sz, 1), 0),
                                     nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.MaxPool2d((1, sz), sz, 0))  # state size ndf * 4 x 1 x 1

        # Pool HV
        self._sam_hv = nn.Sequential(nn.MaxPool2d((1, sz), (1, sz), 0),
                                     nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(channels) if use_bn else nn.InstanceNorm2d(channels),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.MaxPool2d((sz, 1), sz, 0))  # state size ndf * 4 x 1 x 1

        # Weight of HV and VH
        self._sam_w1 = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(channels * 2) if use_bn else nn.InstanceNorm2d(channels * 2),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size 8 x 8

        self._sam_w2 = nn.Sequential(nn.Conv2d(channels * 2, channels * 4, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(channels * 4) if use_bn else nn.InstanceNorm2d(channels * 4),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size 4 x 4

        wn_sz = output_maxpool(sz, ks=2, stride=2)
        wn_sz = output_maxpool(wn_sz, ks=2, stride=2)
        wn_sz = output_maxpool(wn_sz, ks=2, stride=2)

        self._sam_wn = nn.Sequential(nn.AvgPool2d(wn_sz, wn_sz),
                                     nn.Conv2d(channels * 4, channels, 1, 1, 0, bias=False),
                                     nn.Sigmoid())

    @property
    def output_channels(self):
        return self.channels

    def forward(self, x):
        x0 = self._sam_vh(x)

        x1 = self._sam_hv(x)

        x2 = self.drop(self.pool(x))
        x2 = self._sam_w1(x2)
        x2 = self.drop(self.pool(x2))
        x2 = self._sam_w2(x2)
        x2 = self.drop(self.pool(x2))
        x2 = self._sam_wn(x2)

        o = x0 * x2 + (1.0 - x2) * x1
        return o


def get_output_channels(seq, n_blocks):
    if n_blocks > 1:
        last_seq = seq[-1]
    else:
        last_seq = seq

    last_conv = get_last_layer_name(last_seq)
    feature_channels = getattr(last_seq, last_conv).out_channels
    return feature_channels


def get_last_layer_name(layer, prefix="conv", max_layer=10):
    last_conv_name = None
    for i in range(max_layer, 1, -1):
        if hasattr(layer, f"{prefix}{i}"):
            last_conv_name = f"{prefix}{i}"
            break
    return last_conv_name


class KneeImageNet(nn.Module):
    def __init__(self, n_out_features, drop_rate, backbone_name='se_resnext50_32x4d', global_pool_name=None,
                 use_bn=False, pretrained=True, max_depth=5, n_classes=3, n_extra_classes=4, input_3x3=False,
                 n_input_imgs=2):
        super().__init__()

        if max_depth < 1 or max_depth > 5:
            logging.fatal('Max depth must be in [1, 5].')
            assert False

        self.extra_classes = n_extra_classes
        self.n_out_features = n_out_features
        self.n_input_imgs = n_input_imgs

        if use_bn:
            from common.models.networks_bn import make_network
        else:
            from common.models.networks_in import make_network
        self.feature_extractor = make_network(name=backbone_name, pretrained=pretrained, input_3x3=input_3x3)

        self.blocks = []
        for i in range(max_depth):
            if hasattr(self.feature_extractor, f"layer{i}"):
                self.blocks.append(getattr(self.feature_extractor, f"layer{i}"))
            elif i == 0 and 'resnet' in backbone_name:
                self.blocks.append([self.feature_extractor.conv1, self.feature_extractor.bn1,
                                    self.feature_extractor.relu, self.feature_extractor.maxpool])

        self._feature_channels = get_output_channels(self.blocks[-1], max_depth)

        self.n_channel_list = [64, 256, 512, 1024, None]
        # self.sz_list = [75, 75, 38, 19, 10] # 300x300
        self.sz_list = [64, 64, 32, 16, 8]  # 256x256

        self.pool = make_global_pool(pool_name=global_pool_name, channels=self._feature_channels, use_bn=use_bn,
                                     sz=self.sz_list[max_depth - 1])

        use_sam = "sam" in global_pool_name

        if use_sam:
            self.n_features = self.pool.output_channels  # self._feature_channels * 2
        else:
            self.n_features = self._feature_channels

        self.dropout = nn.Dropout(p=drop_rate) if drop_rate is not None else None
        # 4 KL-grades
        if self.extra_classes > 0:
            self.classifier_extra = nn.Linear(self.n_features, n_extra_classes)

        # 3 progression sub-types
        self.classifier_prog = nn.Linear(self.n_features, n_classes)

        self.dual_img_ft = nn.Sequential(nn.Linear(self.n_input_imgs * self.n_features, 2 * self.n_features),
                                         nn.ReLU(),
                                         nn.Linear(2 * self.n_features, self.n_out_features),
                                         nn.ReLU())

        logging.info(f'[INFO] Num of blocks: {len(self.blocks)}')

    def forward_img(self, x):
        for block in self.blocks:
            if isinstance(block, list) or isinstance(block, tuple):
                for sub_block in block:
                    x = sub_block(x)
            else:
                x = block(x)
        x = self.pool(x)

        ft = x.view(x.size(0), -1)

        x = self.dropout(ft)
        if self.extra_classes > 0:
            extra_output = self.classifier_extra(x)
        else:
            extra_output = None
        prog = self.classifier_prog(x)
        return ft, prog, extra_output

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            img_ft, _, _ = self.forward_img(x)
            img_ft = self.dual_img_ft(img_ft)
            return img_ft, None, None
        else:
            device = next(self.parameters()).device
            if isinstance(x, dict):
                img1 = x['img1'].to(device) if x['img1'].device != device else x['img1']
                img2 = x['img2'].to(device) if x['img2'].device != device else x['img2']
            elif isinstance(x, tuple) or isinstance(x, list):
                img1 = x[0].to(device) if x[0].device != device else x[0]
                img2 = x[1].to(device) if x[1].device != device else x[1]
            else:
                raise ValueError(f"Not support input type {type(x)}.")
            img_ft1, _, _ = self.forward_img(img1)
            img_ft2, _, _ = self.forward_img(img2)
            img_ft12 = torch.cat((img_ft1, img_ft2), dim=-1)
            img_ft12 = self.dual_img_ft(img_ft12)

            return img_ft12, None, None


class SSPSAMNet(nn.Module):
    def __init__(self, drop_rate, global_pool_name, backbone_name='se_resnext50_32x4d', max_depth=5, use_bn=False,
                 pretrained=True, ssp_start=3, use_ssp=True, gp_list=None, extra_classes=5, n_classes=2,
                 input_3x3=False):
        super().__init__()

        if max_depth < 1 or max_depth > 5:
            logging.fatal(f'Max depth {max_depth} must be in [1, 5].')
            assert False

        if use_ssp and (ssp_start < 0 or ssp_start > max_depth - 1):
            logging.fatal(
                f'SSP connections must start out of the network scope, but found {ssp_start} not in [0, {max_depth - 1}].')
            assert False

        if gp_list is not None and len(gp_list) != max_depth:
            logging.fatal(f'The length of global pooling list {len(gp_list)} must match max_depth {max_depth}')
            assert False

        if use_bn:
            from common.models.networks_bn import make_network
        else:
            from common.models.networks_in import make_network
        self.feature_extractor = make_network(name=backbone_name, pretrained=pretrained, input_3x3=input_3x3)

        self.extra_classes = extra_classes

        # self.blocks = [getattr(self.feature_extractor, f"layer{i}") for i in range(max_depth)]
        self.n_block_count = len([getattr(self.feature_extractor, f"layer{i}") for i in range(max_depth)])

        self.last_block_channels = get_output_channels(
            getattr(self.feature_extractor, f"layer{self.n_block_count - 1}"), max_depth)

        self.dropout = nn.Dropout(p=drop_rate) if drop_rate is not None else None

        self.ssp_start = ssp_start
        self.use_sam = "sam" in global_pool_name.lower()
        self.max_depth = max_depth
        # self.pools = [None] * max_depth
        self.n_features = 0
        self.extra_classes = extra_classes
        self.n_classes = n_classes

        n_layer4_features = get_output_channels(self.feature_extractor.layer4, max_depth)
        self.n_channel_list = [64, 256, 512, 1024, n_layer4_features]
        self.sz_list = [75, 75, 38, 19, 10]

        n_gp_count = 0

        for i in range(max_depth):
            if gp_list is None:
                if (use_ssp and ssp_start <= i) or (i == max_depth - 1 and max_depth < 5):
                    logging.info(f'Global pooling connection at block {i}')
                    setattr(self, f'global_pool_{i}',
                            make_global_pool(pool_name=global_pool_name, channels=self.n_channel_list[i],
                                             use_bn=use_bn, sz=self.sz_list[i]))
                    self.n_features += getattr(self, f'global_pool_{i}').output_channels if self.use_sam else \
                        self.n_channel_list[i]

                    n_gp_count += 1
                    # self.pools[i] = getattr(self, f'global_pool_{i}')
                elif i == 4:  # max_depth = 5
                    logging.info('Global pooling connection at block 4')
                    self.pool = nn.AvgPool2d(kernel_size=self.sz_list[max_depth - 1],
                                             stride=self.sz_list[max_depth - 1])
                    self.n_features += self.last_block_channels

                    n_gp_count += 1
                    # self.pools[4] = self.pool
            else:
                logging.info(f'Global pooling connection at block {i}')
                setattr(self, f'global_pool_{i}',
                        make_global_pool(pool_name=gp_list[i], channels=self.n_channel_list[i],
                                         use_bn=use_bn, sz=self.sz_list[i]))
                n_input_channels = getattr(self, f'global_pool_{i}').output_channels if "sam" in gp_list[i] else \
                    self.n_channel_list[i]
                self.n_features += n_input_channels

                n_gp_count += 1

        n_gp_count -= 1

        # 4 KL-grades
        self.classifier_kl_4 = nn.Linear(self.n_features, self.extra_classes)
        # 3 progression sub-types
        self.classifier_prog_4 = nn.Linear(self.n_features, self.n_classes)

        # self.feature_mixer = nn.Sequential(nn.Linear(self.n_features, self.n_features), nn.ReLU())

        logging.info(f'Num of blocks: {self.n_block_count}')
        logging.info(f'Num of SSP connections: {n_gp_count}, SSP starts at block {ssp_start}.')

    def forward(self, x):
        gps = []
        for i in range(self.n_block_count):
            x = getattr(self.feature_extractor, f"layer{i}")(x)
            if hasattr(self, f"global_pool_{i}"):
                gps.append(getattr(self, f"global_pool_{i}")(x))

        if len(gps) > 0:
            gps = torch.cat(gps, dim=1)
        else:
            gps = x

        gps = gps.view(gps.size(0), -1)

        # f = self.dropout(f)
        # f = self.feature_mixer(f)

        gps = self.dropout(gps)

        if self.extra_classes > 0:
            extra_output = self.classifier_kl_4(gps)
        else:
            extra_output = None
        prog = self.classifier_prog_4(gps)
        return extra_output, prog


class ENSAMNet(nn.Module):
    def __init__(self, drop_rate, global_pool_name, backbone_name='se_resnext50_32x4d', max_depth=5, use_bn=False,
                 pretrained=True, ssp_start=3, use_ssp=True, gp_list=None, extra_classes=5, n_classes=2,
                 input_3x3=False):
        super().__init__()

        if max_depth < 1 or max_depth > 5:
            logging.fatal(f'Max depth {max_depth} must be in [1, 5].')
            assert False

        if use_ssp and (ssp_start < 0 or ssp_start > max_depth - 1):
            logging.fatal(
                f'SSP connections must start out of the network scope, but found {ssp_start} not in [0, {max_depth - 1}].')
            assert False

        if gp_list is not None and len(gp_list) != max_depth:
            logging.fatal(f'The length of global pooling list {len(gp_list)} must match max_depth {max_depth}')
            assert False

        if use_bn:
            from common.models.networks_bn import make_network
        else:
            from common.models.networks_in import make_network
        self.feature_extractor = make_network(name=backbone_name, pretrained=pretrained, input_3x3=input_3x3)

        self.extra_classes = extra_classes
        self.n_classes = n_classes

        # self.blocks = [getattr(self.feature_extractor, f"layer{i}") for i in range(max_depth)]
        self.n_block_count = len([getattr(self.feature_extractor, f"layer{i}") for i in range(max_depth)])

        self.last_block_channels = get_output_channels(
            getattr(self.feature_extractor, f"layer{self.n_block_count - 1}"), max_depth)

        self.dropout = nn.Dropout(p=drop_rate) if drop_rate is not None else None

        self.ssp_start = ssp_start
        self.use_sam = "sam" in global_pool_name.lower()
        self.max_depth = max_depth
        # self.pools = [None] * max_depth
        self.n_features = 0

        n_layer4_features = get_output_channels(self.feature_extractor.layer4, max_depth)
        self.n_channel_list = [64, 256, 512, 1024, n_layer4_features]
        self.sz_list = [75, 75, 38, 19, 10]

        n_gp_count = 0

        for i in range(max_depth):
            if gp_list is None:
                if (use_ssp and ssp_start <= i) or (i == max_depth - 1 and max_depth < 5):
                    logging.info(f'Global pooling connection at block {i}')
                    setattr(self, f'global_pool_{i}',
                            make_global_pool(pool_name=global_pool_name, channels=self.n_channel_list[i],
                                             use_bn=use_bn, sz=self.sz_list[i]))
                    n_input_channels = getattr(self, f'global_pool_{i}').output_channels if hasattr(
                        getattr(self, f'global_pool_{i}'), 'output_channels') else \
                        self.n_channel_list[i]
                    self.n_features += n_input_channels

                    self._add_prediction(i, n_input_channels)

                    n_gp_count += 1
                    # self.pools[i] = getattr(self, f'global_pool_{i}')
                elif i == 4:  # max_depth = 5
                    logging.info('Global pooling connection at block 4')
                    self.pool = nn.AvgPool2d(kernel_size=self.sz_list[max_depth - 1],
                                             stride=self.sz_list[max_depth - 1])
                    self.n_features += self.last_block_channels

                    self._add_prediction(i, self.last_block_channels)

                    n_gp_count += 1
                    # self.pools[4] = self.pool
            else:
                logging.info(f'Global pooling connection at block {i}')
                setattr(self, f'global_pool_{i}',
                        make_global_pool(pool_name=gp_list[i], channels=self.n_channel_list[i],
                                         use_bn=use_bn, sz=self.sz_list[i]))
                n_input_channels = getattr(self, f'global_pool_{i}').output_channels if "sam" in gp_list[i] else \
                    self.n_channel_list[i]
                self.n_features += n_input_channels

                self._add_prediction(i, n_input_channels)

                n_gp_count += 1

        n_gp_count -= 1

        logging.info(f'Num of blocks: {self.n_block_count}')
        logging.info(f'Num of SSP connections: {n_gp_count}, SSP starts at block {ssp_start}.')

    def _add_prediction(self, i, input_channels):
        setattr(self, f'classifier_kl_{i}', nn.Linear(input_channels, self.extra_classes))
        setattr(self, f'classifier_prog_{i}', nn.Linear(input_channels, self.n_classes))

    def forward(self, x):
        gps = []
        preds_kl = []
        preds_prog = []
        for i in range(self.n_block_count):
            x = getattr(self.feature_extractor, f"layer{i}")(x)
            if hasattr(self, f"global_pool_{i}"):
                gps.append(getattr(self, f"global_pool_{i}")(x))
                x_drop = self.dropout(gps[-1]).view(gps[-1].shape[0], -1)
                if hasattr(self, f"classifier_kl_{i}"):
                    preds_kl.append(getattr(self, f"classifier_kl_{i}")(x_drop))
                if hasattr(self, f"classifier_prog_{i}"):
                    preds_prog.append(getattr(self, f"classifier_prog_{i}")(x_drop))

        kl = torch.mean(torch.stack(preds_kl, dim=1), dim=1)
        prog = torch.mean(torch.stack(preds_prog, dim=1), dim=1)

        return kl, prog


class MUSTNet(nn.Module):
    def __init__(self, drop_rate, backbone_name='se_resnext50_32x4d', use_bn=False,
                 pretrained=True, gp_list=None, extra_classes=5, n_classes=2, input_3x3=False):
        super().__init__()

        max_depth = len(gp_list)
        self.max_depth = max_depth
        if max_depth < 1 or max_depth > 5:
            logging.fatal(f'Max depth {max_depth} must be in [1, 5].')
            assert False

        if gp_list is not None and len(gp_list) != max_depth:
            logging.fatal(f'The length of global pooling list {len(gp_list)} must match max_depth {max_depth}')
            assert False

        if use_bn:
            from common.models.networks_bn import make_network
        else:
            from common.models.networks_in import make_network
        self.feature_extractor = make_network(name=backbone_name, pretrained=pretrained, input_3x3=input_3x3)

        self.extra_classes = extra_classes
        self.n_classes = n_classes

        self.n_block_count = len([getattr(self.feature_extractor, f"layer{i}") for i in range(max_depth)])

        self.last_block_channels = get_output_channels(
            getattr(self.feature_extractor, f"layer{self.n_block_count - 1}"), max_depth)

        self.dropout = nn.Dropout(p=drop_rate) if drop_rate is not None else None

        self.n_features = 0

        n_layer4_features = get_output_channels(self.feature_extractor.layer4, max_depth)
        if backbone_name == 'custom_resnet':
            self.n_channel_list = [64, 64, 128, 256, n_layer4_features]
            self.sz_list = [None, None, 150, 75, 38]
        else:
            self.n_channel_list = [64, 256, 512, 1024, n_layer4_features]
            self.sz_list = [75, 75, 38, 19, 10]

        n_gp_count = 0

        for i in range(max_depth):
            logging.info(f'Global pooling connection at block {i}')
            gp = make_global_pool(pool_name=gp_list[i], channels=self.n_channel_list[i],
                                  use_bn=use_bn, sz=self.sz_list[i])
            if gp is not None:
                setattr(self, f'global_pool_{i}', gp)
                n_input_features = getattr(self, f'global_pool_{i}').output_channels if hasattr(
                    getattr(self, f'global_pool_{i}'), 'output_channels') else \
                    self.n_channel_list[i]
                self.n_features += n_input_features
                self._add_prediction(i, n_input_features)
                n_gp_count += 1

        n_gp_count -= 1

        self.ssp_start = None
        for i, gp in enumerate(gp_list):
            if gp is not None:
                self.ssp_start = i
                break

        logging.info(f'Num of blocks: {self.n_block_count}')
        logging.info(f'Num of SSP connections: {n_gp_count}, SSP starts at block {self.ssp_start}.')

    def _add_prediction(self, i, input_channels):
        setattr(self, f'classifier_kl_{i}', nn.Linear(input_channels, self.extra_classes))
        setattr(self, f'classifier_prog_{i}', nn.Linear(input_channels, self.n_classes))

    def forward(self, x):
        gps = []
        preds_kl = {}
        preds_prog = {}

        for i in range(self.n_block_count):
            x = getattr(self.feature_extractor, f"layer{i}")(x)
            if hasattr(self, f"global_pool_{i}"):
                gps.append(getattr(self, f"global_pool_{i}")(x))
                x_drop = self.dropout(gps[-1]).view(gps[-1].shape[0], -1)
                if hasattr(self, f"classifier_kl_{i}"):
                    preds_kl[i] = getattr(self, f"classifier_kl_{i}")(x_drop)
                if hasattr(self, f"classifier_prog_{i}"):
                    preds_prog[i] = getattr(self, f"classifier_prog_{i}")(x_drop)

        kl = torch.mean(torch.stack([torch.softmax(v, 1) for v in preds_kl.values()], dim=1), dim=1)
        prog = torch.mean(torch.stack([torch.softmax(v, 1) for v in preds_prog.values()], dim=1), dim=1)

        return preds_kl, preds_prog, kl, prog


class SPPNet(nn.Module):
    def __init__(self, cfg, gp_list=None):
        super().__init__()

        max_depth = len(gp_list)
        self.max_depth = max_depth
        if max_depth < 1 or max_depth > 5:
            logging.fatal(f'Max depth {max_depth} must be in [1, 5].')
            assert False

        if gp_list is not None and len(gp_list) != max_depth:
            logging.fatal(f'The length of global pooling list {len(gp_list)} must match max_depth {max_depth}')
            assert False

        if cfg.use_bn:
            from common.models.networks_bn import make_network
        else:
            from common.models.networks_in import make_network
        self.feature_extractor = make_network(name=cfg.backbone_name, pretrained=cfg.pretrained,
                                              input_3x3=cfg.input_3x3)

        self.n_pn_classes = cfg.n_pn_classes
        self.n_pr_classes = cfg.n_pr_classes

        self.n_block_count = len([getattr(self.feature_extractor, f"layer{i}") for i in range(max_depth)])

        self.last_block_channels = get_output_channels(
            getattr(self.feature_extractor, f"layer{self.n_block_count - 1}"), max_depth)

        self.dropout = nn.Dropout(p=cfg.drop_rate) if cfg.drop_rate is not None else None

        self.n_features = 0

        n_layer4_features = get_output_channels(self.feature_extractor.layer4, max_depth)
        if cfg.backbone_name == 'custom_resnet':
            self.n_channel_list = [64, 64, 128, 256, n_layer4_features]
            self.sz_list = [None, None, 150, 75, 38]
        else:
            self.n_channel_list = [64, 256, 512, 1024, n_layer4_features]
            # self.sz_list = [75, 75, 38, 19, 10] # 300x300
            self.sz_list = [64, 64, 32, 16, 8]  # 256x256

        n_gp_count = 0

        self.patch_size = 2
        self.n_patches = 0
        self.patch_dim = self.n_channel_list[-1] * self.patch_size ** 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.vit.dim))
        self.transformer = Transformer(cfg.vit.dim, cfg.vit.depth, cfg.vit.heads, cfg.vit.mlp_dim)
        for i in range(max_depth):
            logging.info(f'Global pooling connection at block {i}')
            if gp_list[i]:
                n_patches = (self.sz_list[i] // self.patch_size) ** 2
                self.n_patches += n_patches
                patch_dim = self.n_channel_list[i] * self.patch_size ** 2
                setattr(self, f"patch_to_embedding_{i}", nn.Linear(patch_dim, cfg.vit.dim))
                # rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
                self.n_features += cfg.vit.dim
                n_gp_count += 1

        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, cfg.vit.dim))
        self.mlp_head = nn.Sequential(
            nn.Linear(cfg.vit.dim, cfg.vit.mlp_dim),
            nn.GELU(),
            nn.Linear(cfg.vit.mlp_dim, self.n_pr_classes + self.n_pn_classes)
        )

        n_gp_count -= 1

        self.ssp_start = None
        for i, gp in enumerate(gp_list):
            if gp is not None:
                self.ssp_start = i
                break

        logging.info(f'Num of blocks: {self.n_block_count}')
        logging.info(f'Num of SSP connections: {n_gp_count}, SSP starts at block {self.ssp_start}.')

    def _add_prediction(self, i, input_channels):
        setattr(self, f'classifier_kl_{i}', nn.Linear(input_channels, self.extra_classes))
        setattr(self, f'classifier_prog_{i}', nn.Linear(input_channels, self.n_classes))

    def forward(self, x):
        # preds_kl = {}
        # preds_prog = {}

        bs = x.shape[0]
        patches = [self.cls_token.expand(bs, -1, -1)]

        for i in range(self.n_block_count):
            x = getattr(self.feature_extractor, f"layer{i}")(x)
            if hasattr(self, f"patch_to_embedding_{i}"):
                emb = getattr(self, f"patch_to_embedding_{i}")(x)
                patches.append(emb)

        seq_ft = torch.cat(patches, dim=1)
        seq_ft += self.pos_embedding
        seq_ft = self.transformer(seq_ft)

        target_ft = seq_ft[:, 0]
        logits = self.mlp_head(target_ft)
        # kl = torch.mean(torch.stack([torch.softmax(v, 1) for v in preds_kl.values()], dim=1), dim=1)
        # prog = torch.mean(torch.stack([torch.softmax(v, 1) for v in preds_prog.values()], dim=1), dim=1)

        return logits


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
