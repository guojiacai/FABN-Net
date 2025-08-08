import torch.nn as nn
from torch.nn import functional as F
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import math
from transformers import CLIPProcessor, CLIPModel



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexConv2d, self).__init__()
        # define the real and imaginary parts of the convolutional kernels
        self.real_conv = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1)
        self.imag_conv = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1)

    def forward(self, x):
        # extract the real and imaginary parts of the input
        real = x.real
        imag = x.imag

        # forward the real and imaginary parts through the real and imaginary kernels
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)

        # combine the real and imaginary parts
        return torch.complex(real_out, imag_out)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        flag = 0
        if x.dim() == 5:
            B, C, _, w, h = x.shape
            x = x.view(B, C * 3, w, h).clone()
            flag = 1
        else:
            B, C, _, _ = x.shape

        avg_out = torch.mean(x, dim=1, keepdim=True)  # 
        #avg_out = avg_out.squeeze(1)
        max_out, nonthing = torch.max(x, dim=1, keepdim=True) # 
        #max_out = max_out.squeeze(1)
        
        spatial_weights = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

        x = x * spatial_weights
        if flag == 1:
            x = x.view(B, C, 3, w, h).clone()
        return x 




class NEWMODEL(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4], num_classes=1, zero_init_residual=False):
        super(NEWMODEL, self).__init__()

        self.dwt=DWTForward(J=1,mode='zero',wave='db3')
        self.idwt=DWTInverse(mode='zero',wave='db3')
        self.weightdw1 = nn.Parameter(torch.randn((64, 3, 1, 1)).cuda())
        self.biasdw1   = nn.Parameter(torch.randn((64,)).cuda())



        self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)).cuda())
        self.bias1   = nn.Parameter(torch.randn((64,)).cuda())
        self.complexconv1 = ComplexConv2d(64, 64)



        self.weightdw2 = nn.Parameter(torch.randn((64, 64, 1, 1)).cuda())
        self.biasdw2   = nn.Parameter(torch.randn((64,)).cuda())


        self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)).cuda())
        self.bias2   = nn.Parameter(torch.randn((64,)).cuda())
        self.complexconv2 = ComplexConv2d(64, 64)



        self.weightdw3 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.biasdw3  = nn.Parameter(torch.randn((256,)).cuda())

        self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.bias3   = nn.Parameter(torch.randn((256,)).cuda())
        self.complexconv3 = ComplexConv2d(256, 256)


        self.weightdw4 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.biasdw4   = nn.Parameter(torch.randn((256,)).cuda())

        self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.bias4   = nn.Parameter(torch.randn((256,)).cuda())
        self.complexconv4 = ComplexConv2d(256, 256)

        
        
        self.inplanes = 64 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])            
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)
        
        self.SP_attn1 = SpatialAttention()
        self.SP_attn2 = SpatialAttention()
        self.SP_attn3 = SpatialAttention()
        self.SP_attn4 = SpatialAttention()
        
        self.SP_attnyhD1 = SpatialAttention()
        self.SP_attnyhD2 = SpatialAttention()
        self.SP_attnyhD3 = SpatialAttention()
        self.SP_attnyhD4 = SpatialAttention()
        
        self.SP_attnylD1 = SpatialAttention()
        self.SP_attnylD2 = SpatialAttention()
        self.SP_attnylD3 = SpatialAttention()
        self.SP_attnylD4 = SpatialAttention()


        
        self.drop1 = AdaptiveChannelDropoutV2(64, (0.1, 0.3))
        self.drop2 = AdaptiveChannelDropoutV2(64, (0.1, 0.3))
        self.drop3 = AdaptiveChannelDropoutV2(256, (0.1, 0.3))
        self.drop4 = AdaptiveChannelDropoutV2(256, (0.1, 0.3))
        self.freq_attn1 = FrequencyAttention(gamma_low=0.8, gamma_high=1.2, threshold_ratio=0.5)
        self.freq_attn2 = FrequencyAttention(gamma_low=0.8, gamma_high=1.2, threshold_ratio=0.5)
        self.freq_attn3 = FrequencyAttention(gamma_low=0.8, gamma_high=1.2, threshold_ratio=0.5)
        self.freq_attn4 = FrequencyAttention(gamma_low=0.8, gamma_high=1.2, threshold_ratio=0.5)
        
        self.dwtEnhancer = AdaptiveFeatureEnhancer()

        self.proj_768 = nn.Linear(768, 512)
        self.weights = nn.Parameter(torch.tensor([0.2, 0.8])) 



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)



    def forward(self, x):

        scale = 4
        

        yl,yh=self.dwt(x)
        yh = self.dwtEnhancer(yh)
        yh = self.SP_attnyhD1(yh)  
        yh = [yh]
        yl = self.SP_attnylD1(yl)   

        dwtx=self.idwt((yl,yh))
        dwtx=F.conv2d(dwtx, self.weightdw1, self.biasdw1, stride=1, padding=0)
        dwtx=F.relu(dwtx, inplace=True)

        
        x=dwtx + x
        


        yl,yh=self.dwt(x)
        yh = self.dwtEnhancer(yh) 
        yh = self.SP_attnyhD2(yh)   
        yh = [yh]
        yl = self.SP_attnylD2(yl)   # 

        # 
        dwtx=self.idwt((yl,yh))
        dwtx=F.conv2d(dwtx, self.weightdw2, self.biasdw2, stride=2, padding=0)
        dwtx=F.relu(dwtx, inplace=True)
 

        x=dwtx + x

        x = self.maxpool(x)
        x = self.layer1(x)  # 
        

        #
        yl,yh=self.dwt(x)
        yh = self.dwtEnhancer(yh) #  
        yh = self.SP_attnyhD3(yh)   # 
        yh = [yh]
        yl = self.SP_attnylD3(yl)   # 

        # 
        dwtx=self.idwt((yl,yh))
        dwtx=F.conv2d(dwtx, self.weightdw3, self.biasdw3, stride=1, padding=0)
        dwtx=F.relu(dwtx, inplace=True)

        
        x=dwtx + x



        # 
        yl,yh=self.dwt(x)
        yh = self.dwtEnhancer(yh) #  
        yh = self.SP_attnyhD4(yh)   # 
        yh = [yh]
        yl = self.SP_attnylD4(yl)   # 

        # 
        dwtx=self.idwt((yl,yh))
        dwtx=F.conv2d(dwtx, self.weightdw4, self.biasdw4, stride=2, padding=0)
        dwtx=F.relu(dwtx, inplace=True)

        
        x=dwtx + x

        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1).clone()
        w = torch.softmax(self.weights, dim=0)
        x = self.fc1(x)
        return x


class AdaptiveFeatureEnhancer(nn.Module):
    def __init__(self, alpha=0.2, gamma=0.7, sigma=8, clip_range=(0, 255)):
        """

        """
        super(AdaptiveFeatureEnhancer, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.clip_min, self.clip_max = clip_range
    # 
    def batch_normalize(self, tensor):
        """"""
        original_shape = tensor.shape
        reshaped = tensor.view(-1, original_shape[-2] * original_shape[-1]).clone()
    
        min_vals = reshaped.min(dim=1, keepdim=True)[0]
        max_vals = reshaped.max(dim=1, keepdim=True)[0]
        normalized = (reshaped - min_vals) / (max_vals - min_vals + 1e-8)
    
        return normalized.view(original_shape).clone()
    
    def forward(self, feature_map):
        """

        """
        # 
        # 
        feature_map = feature_map[0]
        is_single = feature_map.dim() == 3 # 
        if is_single:
            feature_map = feature_map.unsqueeze(0).unsqueeze(2)
            


        noise_template = torch.sign(feature_map) * (torch.abs(feature_map)+1e-6).pow(self.gamma)
        noise_template = torch.sign(feature_map) * (torch.abs(feature_map)+1e-6).pow(self.gamma)
        normalized_template = self.batch_normalize(noise_template)
            # 
        enhanced_feature = torch.clamp(
            2.0 * feature_map + 
            0.001*self.alpha * normalized_template * torch.randn_like(feature_map) * self.sigma,
            self.clip_min, self.clip_max
        )
    
        return enhanced_feature.squeeze(0).squeeze(1) if is_single else enhanced_feature
        

class AdaptiveChannelDropoutV2(nn.Module):
    def __init__(self, channels, drop_prob_range=(0.05, 0.2), temperature=0.5):
        super().__init__()
        self.channel_importance = nn.Parameter(torch.ones(channels))
        self.drop_min, self.drop_max = drop_prob_range
        self.temperature = temperature

    def forward(self, x):
        if not self.training:
            return x

        # 
        channel_probs = torch.sigmoid(self.channel_importance)
        
        # 
        dropout_rates = self.drop_min + (self.drop_max - self.drop_min) * (1 - channel_probs)
        
        #
        retain_probs = 1 - dropout_rates
        logits = torch.log(retain_probs.clamp(min=1e-8)) - torch.log((1 - retain_probs).clamp(min=1e-8))
        
        # 
        uniform_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
        
        # 
        mask_soft = torch.sigmoid((logits + gumbel_noise) / self.temperature)
        
        # 
        mask_hard = (mask_soft > 0.5).float()
        mask = mask_hard - mask_soft.detach() + mask_soft
        
        # 
        mask = mask.view(1, -1, 1, 1).clone().to(x.device)
        scale = 1.0 / (1 - dropout_rates.mean().clamp(min=1e-8))
        return x * mask * scale
        
class FrequencyAttention(nn.Module):
    """

    """
    def __init__(self, gamma_low=0.8, gamma_high=1.2, threshold_ratio=0.5, eps=1e-6):
        super(FrequencyAttention, self).__init__()
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high
        self.threshold_ratio = threshold_ratio
        self.eps = eps

    def forward(self, x):
        """
        
        """
        B, C, H, W = x.size()
        device = x.device

        fft = torch.fft.fft2(x, dim=(-2, -1))  # FFT 
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)  # 

        #  (0, 0)
        y_coords = torch.arange(H, device=device) - H // 2
        x_coords = torch.arange(W, device=device) - W // 2
        # torch.meshgrid  indexing='ij'）
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # 
        radius = torch.sqrt(grid_y.float() ** 2 + grid_x.float() ** 2)
        max_radius = radius.max()
        # 
        freq_threshold = self.threshold_ratio * max_radius

        low_mask = (radius <= freq_threshold).float().unsqueeze(0).unsqueeze(0)
        high_mask = (radius > freq_threshold).float().unsqueeze(0).unsqueeze(0)

        low_energy = torch.sum(magnitude * low_mask, dim=(-2, -1))
        high_energy = torch.sum(magnitude * high_mask, dim=(-2, -1))

        ratio = high_energy / (low_energy + high_energy + self.eps)  # shape: [B, C]

        # 6. 
        scale = self.gamma_low + (self.gamma_high - self.gamma_low) * ratio  # shape: [B, C]
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # 

        out = x * scale

        return out

def newmodel(**kwargs):

    return NEWMODEL()
    
    


    


