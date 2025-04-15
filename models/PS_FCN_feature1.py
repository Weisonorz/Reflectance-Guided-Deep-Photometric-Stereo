
"""
Complete replica of PS_FCN as a baseline model
"""

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from . import cbn

class BRDF_EmbedBlock(nn.Module):
    # LSTM*
    def __init__(self, brdf_input_size, brdf_embed_size):
        super(BRDF_EmbedBlock, self).__init__()
        self.brdf_input_size = brdf_input_size
        self.brdf_embed_size = brdf_embed_size

        self.embed_transform = nn.Linear(self.brdf_input_size, self.brdf_embed_size)
    
    def forward(self, x):
        x = self.embed_transform(x)

        return x

class Conditional_Conv(nn.Module):
    def __init__(self, cin, cout, batch_size, brdf_embed_size, k=3, stride=1, pad=-1):
        super(Conditional_Conv, self).__init__()

        self.regular_conv = nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False)
        
        self.cBatchNorm = cbn.CBN(brdf_embed_size, brdf_embed_size, cout, batch_size, cout)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    
    def forward(self, x, embeding_vector):
        x = self.regular_conv(x)
        x, _ = self.cBatchNorm(x, embeding_vector)
        x = self.activation(x)

        return x


class FeatExtractor_CBN(nn.Module):
    '''
    Remember to pass in batch_size, height, width
    '''
    def __init__(self, batch_size, c_in, batchNorm=False, other={}, brdf_embed_size=512):
        
        super(FeatExtractor_CBN, self).__init__()

        # find brdf embedding first
        self.other = other

        # downsampling
        self.conv1 = Conditional_Conv(c_in, 64, batch_size, brdf_embed_size, k=3, stride=1, pad=1)
        self.conv2 = Conditional_Conv(64, 128, batch_size, brdf_embed_size, k=3, stride=2, pad=1)
        self.conv3 = Conditional_Conv(128, 128, batch_size, brdf_embed_size, k=3, stride=1, pad=1)
        self.conv4 = Conditional_Conv(128,  256, batch_size, brdf_embed_size, k=3, stride=2, pad=1)
        self.conv5 = Conditional_Conv(256,  256, batch_size, brdf_embed_size, k=3, stride=1, pad=1)

        # upsampling
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x, brdf_embed_vector):
        out = self.conv1(x, brdf_embed_vector)
        out = self.conv2(out, brdf_embed_vector)
        out = self.conv3(out, brdf_embed_vector)
        out = self.conv4(out, brdf_embed_vector)
        out = self.conv5(out, brdf_embed_vector)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.size()
        out_feat = out_feat.view(n, -1)
        return out_feat, [n, c, h, w]

class Regressor_CBN(nn.Module):
    def __init__(self, batchNorm=False, other={}): 
        super(Regressor_CBN, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class PS_FCN_CBN(nn.Module):
    def __init__(self, 
                batch_size, 
                fuse_type='max', 
                batchNorm=True, 
                c_in=3, 
                other={},
                brdf_input_size=14, 
                brdf_embed_size=512):
        super(PS_FCN_CBN, self).__init__()

        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        self.brdf_embed_block = BRDF_EmbedBlock(brdf_input_size, brdf_embed_size)
        c_in_concatenated = self.c_in*2
        self.extractor = FeatExtractor_CBN(batch_size, c_in_concatenated, batchNorm=False, other=other, brdf_embed_size=brdf_embed_size)
        self.regressor = Regressor_CBN(batchNorm, other)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img, light=None, brdf=None):
        # print("DEBUG: img.shape = ", img.shape)
        # print("DEBUG: light.shape = ", light.shape)
        # print("DEBUG: brdf.shape = ", brdf.shape)
        img_split = torch.split(img, self.c_in, 1)
        light_split = torch.split(light, self.c_in, 1) if light is not None else [None] * len(img_split)

        brdf_embedding_vector = self.brdf_embed_block(brdf)

        feats = []
        for i in range(len(img_split)):
            if light is not None:
                net_in = torch.cat([img_split[i], light_split[i]], 1)
            else:
                net_in = img_split[i]
            # print("DEBUG: ", img_split[i].shape)
            # print("DEBUG: ", light_split[i].shape)
            # print("DEBUG: ", net_in.shape)
            feat, shape = self.extractor(net_in, brdf_embedding_vector)
            feats.append(feat)


        if self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        else:
            raise ValueError(f"Unsupported fuse_type: {self.fuse_type}")

        normal = self.regressor(feat_fused, shape)
        return normal
