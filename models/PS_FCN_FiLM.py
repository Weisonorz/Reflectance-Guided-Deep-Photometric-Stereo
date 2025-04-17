import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from .FiLM import FiLMConv

class BRDF_EmbedBlock(nn.Module):
    def __init__(self, brdf_input_size, brdf_embed_size):
        super(BRDF_EmbedBlock, self).__init__()
        self.brdf_input_size = brdf_input_size
        self.brdf_embed_size = brdf_embed_size

        self.embed_transform = nn.Sequential(nn.Linear(brdf_input_size, brdf_embed_size),
                                             nn.LeakyReLU(0.1))
    
    def forward(self, x):
        x = self.embed_transform(x)

        return x
    

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, brdf_emb_dim=512, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = FiLMConv(batchNorm, c_in, 64,  k=3, stride=1, pad=1, brdf_emb_dim=brdf_emb_dim)
        self.conv2 = FiLMConv(batchNorm, 64,   128, k=3, stride=2, pad=1, brdf_emb_dim=brdf_emb_dim)
        self.conv3 = FiLMConv(batchNorm, 128,  128, k=3, stride=1, pad=1, brdf_emb_dim=brdf_emb_dim)
        self.conv4 = FiLMConv(batchNorm, 128,  256, k=3, stride=2, pad=1, brdf_emb_dim=brdf_emb_dim)
        self.conv5 = FiLMConv(batchNorm, 256,  256, k=3, stride=1, pad=1, brdf_emb_dim=brdf_emb_dim)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x, brdf_emb_vector):
        out = self.conv1(x, brdf_emb_vector)
        out = self.conv2(out, brdf_emb_vector)
        out = self.conv3(out, brdf_emb_vector)
        out = self.conv4(out, brdf_emb_vector)
        out = self.conv5(out, brdf_emb_vector)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat   = out_feat.view(-1)
        return out_feat, [n, c, h, w]

class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}): 
        super(Regressor, self).__init__()
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

class PS_FCN_FiLM(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, brdf_input_size=14, brdf_embed_size=512, other={}):
        super(PS_FCN_FiLM, self).__init__()
        self.brdf_embed_block = BRDF_EmbedBlock(brdf_input_size, brdf_embed_size)
        self.extractor = FeatExtractor(batchNorm, c_in, brdf_emb_dim=brdf_embed_size)
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        self.brdf_embed_block = BRDF_EmbedBlock(brdf_input_size, brdf_embed_size)
        self.extractor = FeatExtractor(batchNorm, c_in, other=other, brdf_emb_dim=brdf_embed_size)
        self.regressor = Regressor(batchNorm, other)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img   = x[0]
        brdf = x[2]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1: # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)

        brdf_embedding_vector = self.brdf_embed_block(brdf)

        feats = []
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
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
