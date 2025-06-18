
import torch.nn.functional as F

from einops import rearrange

import torch
import torch.nn as nn

def stride_generator(N, reverse=False):
    strides = [1, 1,1,1,1]
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, channels,out_channels, kernel_size=3, stride=1, padding=1, transpose=False, act_norm=False,groups = 1):
        super(BasicConv3d, self).__init__()
        self.act_norm = act_norm
        # self.conv_in = nn.Conv3d(in_channels, channels, kernel_size=1, stride=1, padding=0,groups = groups)
        self.conv_in = nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1, groups=groups)
        if not transpose:
            self.conv = nn.Conv3d(channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups = groups)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv_in(x)
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, (1,2,2), stride=(1,2,2))

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class DenseUNet(nn.Module):
    """B-DenseUNets"""
    def __init__(self, in_nc=12, out_nc=12,channel = 24):
        super(DenseUNet, self).__init__()
        lay=2
        feat = channel
        self.inc = nn.Sequential(
            single_conv(in_nc, feat*2),
            single_conv(feat*2, feat*2),
        )
        self.down1 = nn.Conv3d(feat*2, feat*2,kernel_size=(1,2,2),stride=(1,2,2),padding=0)
        self.conv1 = nn.Sequential(
            single_conv(feat*2, feat*4),
            RDB(feat*4, lay, feat),
        )
        self.down2 = nn.Conv3d(feat*4, feat*4,kernel_size=(1,2,2),stride=(1,2,2),padding=0)
        self.conv2 = nn.Sequential(
            single_conv(feat*4, feat*8),
            RDB(feat*8, lay+1, feat),
        )
        self.up1 = up(feat*8)
        self.conv3 = nn.Sequential(
            RDB(feat*4, lay+1, feat),
        )


        self.up2 = up(feat*4)
        self.conv4 = nn.Sequential(
            RDB(feat*2, lay, feat),
        )

        self.outc = outconv(feat*2, out_nc)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out
 
class LKA(nn.Module):
    def __init__(self, C_hid, C_out, act_norm=True):
        super().__init__()
        self.act_norm = act_norm
        dim = C_hid
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=3, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, C_out, 1)
        self.norm = nn.GroupNorm(2, C_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        if self.act_norm:
            attn = self.act(self.norm(attn))
        return attn

class Amplitude_Phase_Block(nn.Module):
    def __init__(self, C_in, channels, stride):
        super(Amplitude_Phase_Block, self).__init__()
        self.stride = stride
        self.fpre = nn.Conv2d(C_in, channels, 1, 1, 0)
        self.amp_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))
        self.pha_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))

        self.down = nn.Conv2d(channels,channels,3,stride,1)
    
    def forward(self, x):
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_res = self.amp_fuse_1(msF_amp)
        pha_res = self.pha_fuse_1(msF_pha)

        amp_fuse = amp_res + msF_amp
        pha_fuse = pha_res + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.down(out)
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return out

class Bidomain_Cross_Attention(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(Bidomain_Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = LKA(dim * 2, dim * 2)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = LKA(dim, dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Spectral_DenseUNet(nn.Module):
    def __init__(self,in_nc=1,out_nc=1,channel =20):
        super(Spectral_DenseUNet, self).__init__()
        self.main = DenseUNet(in_nc, out_nc,channel=channel)
        self.out = nn.Conv3d(out_nc*2, out_nc, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        input = x
        out1 = self.main(input) + x
        cat1 = torch.cat([x, out1], dim=1)
        return self.out(cat1) + x
    
class Frequency_Extractor(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Frequency_Extractor, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            Amplitude_Phase_Block(C_in, C_hid, stride=strides[0]),
            *[Amplitude_Phase_Block(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        feat = []
        latent = self.enc[0](x)
        feat.append(latent)

        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            feat.append(latent)

        return latent, feat

class Spatial_Extractor(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Spatial_Extractor, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        feat = []
        latent = self.enc[0](x)
        feat.append(latent)

        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            feat.append(latent)

        return latent, feat

class Frequency_Spatial_Fusion(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(Frequency_Spatial_Fusion, self).__init__()
        self.att1 = Bidomain_Cross_Attention(dim=dim, num_heads=num_heads, bias=bias)
        self.att2 = Bidomain_Cross_Attention(dim=dim, num_heads=num_heads, bias=bias)
        self.out = nn.Conv2d(dim*2,dim,3,1,1)

    def forward(self, x, y):
        out1 = self.att1(x,y)
        out2 = self.att2(y,x)
        out = torch.cat((x+out1,y+out2),dim=1)
        out = self.out(out)
        return out

class Spectral_Evolution(nn.Module):
    def __init__(self,in_nc=1,out_nc=1,channels = 20):
        super(Spectral_Evolution, self).__init__()
        self.net = Spectral_DenseUNet(in_nc=in_nc,out_nc=out_nc,channel= channels)

    def forward(self, x):
        x = torch.transpose(x,1,2)
        z = self.net(x)
        y = torch.transpose(z,1,2)
        return y

class Restorer(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Restorer, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            ConvSC(C_hid, C_hid, stride=strides[0], transpose=True),
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[1:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None,enc2 = None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class fourierNet(nn.Module):
    def __init__(self,
                 hid_S,
                 hid_T):
        super(fourierNet, self).__init__()
        T, C = 31,1
        N_S = 2
        self.up = nn.Conv2d(C,hid_S,3,1,1)
        self.enc = Spatial_Extractor(hid_S, hid_S, N_S)
        self.enc_fre = Frequency_Extractor(hid_S, hid_S, N_S)
        self.fusion = Frequency_Spatial_Fusion(dim=hid_S)
        self.hid = Spectral_Evolution(in_nc=hid_S,out_nc=hid_S,channels=hid_T )
        self.dec = Restorer(hid_S, C, N_S)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        x= self.up(x)
        embed, feat = self.enc(x)
        skip = feat[0]

        embed_fre, feat_fre = self.enc_fre(x)
        embed = self.fusion(embed,embed_fre)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        # z = self.down(z)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y

class FIDNet(nn.Module):
    def __init__(self,
                 hid_S,
                 hid_T,
                 opt = None):
        super(FIDNet, self).__init__()
        self.fourierNet = fourierNet(hid_S = hid_S,
                                hid_T = hid_T)
    def forward(self,HSI):
        device = HSI.device
        input = torch.unsqueeze(HSI, dim=2)
        final_denoised = self.fourierNet(input)
        final_denoised = torch.squeeze(final_denoised, dim=2)

        return final_denoised
