import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import lerp
import numpy

class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class WScaleLayer(nn.Module):
    def __init__(self, size, gain=numpy.sqrt(2),fan_in=None):
        super(WScaleLayer, self).__init__()
        if fan_in is None:
            fan_in=numpy.prod(size[:-1])
        self.scale = gain / numpy.sqrt(fan_in) # No longer a parameter
        #print(self.scale)
        self.b = nn.Parameter(torch.zeros(size[-1]))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
            x_size[0], self.size[-1], x_size[2], x_size[3])
        return x

class WScaleLinearLayer(nn.Module):
    def __init__(self, size, gain=numpy.sqrt(2),fan_in=None):
        super(WScaleLinearLayer, self).__init__()
        if fan_in is None:
            fan_in=numpy.prod(size[:-1])
        self.scale = gain / numpy.sqrt(fan_in) # No longer a parameter
        self.b = nn.Parameter(torch.zeros(size[-1]))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1).expand(
            x_size[0], self.size[-1])
        return x

class LinearBlock(nn.Module):
    def __init__(self, in_channels,out_channels,gain=numpy.sqrt(2),use_relu=False):
        super(LinearBlock,self).__init__()
        self.linear=nn.Linear(in_channels,out_channels,bias=False)
        self.wscale = WScaleLinearLayer([in_channels,out_channels], gain=gain)
        self.relu=nn.LeakyReLU(negative_slope=0.2)
        self.use_relu=use_relu
        #print(self.wscale.scale)
        # init
        self.linear.weight.data.normal_()
        # ---

    def forward(self, x):
        x = self.linear(x)
        x = self.wscale(x)
        if self.use_relu:
            x=self.relu(x)
        return x

class NormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer([kernel_size,kernel_size,in_channels,out_channels],
                gain=numpy.sqrt(2))
        #print(self.wscale.scale)#0.0208333333333
        self.relu = nn.LeakyReLU( negative_slope=0.2)
        # init
        self.conv.weight.data.normal_()
        # ---

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        x = self.norm(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer([kernel_size,kernel_size,in_channels,out_channels],gain=numpy.sqrt(2))
        #init
        self.conv.weight.data.normal_()
        #---
        self.relu = nn.LeakyReLU( negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x


class NormUpscaleBilinearConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleBilinearConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.wscale = WScaleLayer([kernel_size,kernel_size,in_channels,out_channels],
                gain=numpy.sqrt(2))
        #print(self.wscale.scale)#0.0208333333333
        # init
        self.conv.weight.data.normal_()
        # ---
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = F.interpolate(x,scale_factor=2,mode="nearest")
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        x = self.norm(x)
        return x

class NormUpscaleTransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleTransposeConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, bias=False)
        self.wscale = WScaleLayer([kernel_size,kernel_size,in_channels,out_channels],
                gain=numpy.sqrt(2),fan_in=kernel_size*kernel_size*in_channels)
        #print(self.wscale.scale)#0.0208333333333

        # init
        weight=self.deconv.weight.data.normal_()
        weight=F.pad(weight,(1,1,1,1))
        weight=weight[:,:,1:,1:]+weight[:,:,:-1,:-1]+weight[:,:,:-1,1:]+weight[:,:,1:,:-1]
        self.deconv=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False)
        self.deconv.weight.data.copy_(weight)
        del weight
        # ---
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.deconv(x)
        x = self.relu(self.wscale(x))
        x = self.norm(x)
        return x

class DownscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownscaleConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,stride=2,padding=1, bias=False)
        self.wscale = WScaleLayer([kernel_size,kernel_size,in_channels,out_channels],
                gain=numpy.sqrt(2))
        # init
        self.conv.weight.data.normal_()
        #self.conv.weight.data *= self.wscale.scale
        # ---
        self.relu = nn.LeakyReLU( negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x

class DownscalePoolConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownscalePoolConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,padding=1, bias=False)
        self.wscale = WScaleLayer([kernel_size,kernel_size,in_channels,out_channels],
                gain=numpy.sqrt(2))
        # init
        self.conv.weight.data.normal_()
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        x=F.avg_pool2d(x,2)
        return x

class OutputConvBlock(nn.Module):
    def __init__(self, in_channels,out_channels=3,kernel_size=1):
        super(OutputConvBlock,self).__init__()
        self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=0, bias=False)
        self.wscale = WScaleLayer([kernel_size,kernel_size,in_channels,out_channels], gain=1)
        #print(self.wscale.scale)#0.0441941738242
        # init
        self.conv.weight.data.normal_()

    def forward(self, x):
        x = self.conv(x)
        x = self.wscale(x)
        return x

class InputLatentBlock(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3):
        super(InputLatentBlock,self).__init__()
        self.norm = PixelNormLayer()
        self.linear = nn.Linear(in_channels, out_channels*4*4, bias=False)
        self.Lwscale = WScaleLinearLayer([in_channels,out_channels*4*4], gain=numpy.sqrt(2)/4)
        #print(self.Lwscale.scale)#0.015625
        self.relu = nn.LeakyReLU( negative_slope=0.2)
        # init
        self.linear.weight.data.normal_()

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        x = self.Lwscale(x)
        x=x.reshape(-1,512,4,4)
        x = self.relu(x)
        x=self.norm(x)
        return x

class InputImgBlock(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=1):
        super(InputImgBlock,self).__init__()
        self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.wscale = WScaleLayer([kernel_size,kernel_size,in_channels,out_channels], gain=numpy.sqrt(2))
        #print(self.wscale.scale)#0.816496580928
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        # init
        self.conv.weight.data.normal_()

    def forward(self, x):
        x = self.conv(x)
        x = self.wscale(x)
        x = self.relu(x)
        return x
#--------------------------------------------------------------------------------------------------------------------
class D_reshape_module(nn.Module):
    def __init__(self):
        super(D_reshape_module, self).__init__()

    def forward(self, x):
        return x.reshape((x.shape[0], -1))

class minibatch_std_layer(nn.Module):
    def __init__(self,group_size=4):
        super(minibatch_std_layer, self).__init__()
        self.group_size=group_size

    def forward(self, x):
        size = x.size()
        subGroupSize = min(size[0], self.group_size)
        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]
        G = int(size[0] / subGroupSize)
        if subGroupSize > 1:
            y = x.view(subGroupSize, -1, size[1], size[2], size[3])
            y = torch.var(y, 0)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, size[2] * size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return torch.cat([x, y], dim=1)

class G_paper(nn.Module):

    def block(self,res):
        name="%d*%d_block"%(2**res,2**res)
        if res==2:
            model=nn.Sequential()
            model.add_module("InputLatent",InputLatentBlock(self.res_channels.get(res),self.res_channels.get(res)))
            model.add_module("Conv",NormConvBlock(self.res_channels.get(res),self.res_channels.get(res),kernel_size=3,padding=1))
        else:
            model=nn.Sequential()
            model.add_module("UpscaleConv",NormUpscaleTransposeConvBlock(self.res_channels.get(res-1),self.res_channels.get(res),kernel_size=3,padding=1))
            model.add_module("Conv",NormConvBlock(self.res_channels.get(res),self.res_channels.get(res),kernel_size=3,padding=1))
        return name,model

    def Add_To_RGB_Layer(self,res):
        name="%d*%d_toRGB"%(2**res,2**res)
        self.to_RGBlayer.add_module(name,OutputConvBlock(self.res_channels.get(res)))
        #if res>=4:
            #self.to_RGBlayer = nn.Sequential(*list(self.to_RGBlayer.children())[-2:]).cuda()

    def Add_To_RGB_Layer_copy_weight(self,res,G):
        name="%d*%d_toRGB"%(2**res,2**res)
        self.to_RGBlayer.add_module(name,OutputConvBlock(self.res_channels.get(res)))
        self.to_RGBlayer[-1].load_state_dict(G.to_RGBlayer[-1].state_dict().copy())

    def update_Gs(self,G,decay=0.999):
        with torch.no_grad():
            G_param_dict=dict(G.named_parameters())
            for name,param in self.named_parameters():
                param_G=G_param_dict[name]
                param.copy_(decay * param + (1. - decay) * param_G.detach())

    def create_graph(self,max_res):
        self.to_RGBlayer=nn.Sequential()
        self.Add_To_RGB_Layer(res=2)
        Generator=nn.Sequential()
        for res in range(2,max_res+1):
            name, model = self.block(res)
            Generator.add_module(name, model)
        return Generator

    def __init__(self,max_resolution):
        super(G_paper, self).__init__()
        self.res_channels={2:512,3:512,4:512,5:512,6:256,7:128,8:64,9:32,10:16}
        self.max_resolution=max_resolution
        self.G=self.create_graph(max_res=max_resolution)
        print(self.G)

    def forward(self, x,lod_in,phase,TRANSITION):
        if phase==2:
            x=self.G[0](x)
            return self.to_RGBlayer[-1](x)
        else:
            if TRANSITION is True:
                alpha=lod_in-phase

                for res in range(2,phase):
                    x=self.G[res-2](x)
                return lerp(F.interpolate(self.to_RGBlayer[-2](x),scale_factor=2,mode="nearest"),self.to_RGBlayer[-1](self.G[phase-2](x)),alpha)
            else:
                for res in range(2,phase+1):
                    x=self.G[res-2](x)
                return self.to_RGBlayer[-1](x)


class D_paper(nn.Module):
    def block(self,res):
        name="%d*%d_block"%(2**res,2**res)
        if res==2:
            model=nn.Sequential()
            model.add_module("minibatch_stddev",minibatch_std_layer())
            model.add_module("conv",ConvBlock(self.res_channels.get(res)+1,self.res_channels.get(res),kernel_size=3,padding=1))
            model.add_module("D_reshape",D_reshape_module())
            model.add_module("Linear1",LinearBlock(self.res_channels.get(res)*4*4,self.res_channels.get(res),use_relu=True))
            model.add_module("Linear2", LinearBlock(self.res_channels.get(res), 1,gain=1,use_relu=False))
        else:
            model=nn.Sequential()
            model.add_module("conv",
                             ConvBlock(self.res_channels.get(res),
                                                self.res_channels.get(res),
                                              kernel_size=3,padding=1))
            model.add_module("downscale",DownscaleConvBlock(self.res_channels.get(res),
                                                self.res_channels.get(res-1),
                                              kernel_size=3))
        return name,model

    def Add_From_RGB_Layer(self,res):
        name = "%d*%d_fromRGB" % (2 ** res, 2 ** res)
        self.from_RGBlayer.add_module(name,InputImgBlock(3,self.res_channels.get(res)))
        #if res>=4:
            #self.from_RGBlayer = nn.Sequential(*list(self.from_RGBlayer.children())[-2:]).cuda()

    def create_graph(self,max_res):
        self.from_RGBlayer = nn.Sequential()
        self.Add_From_RGB_Layer(res=2)
        Discriminator = nn.Sequential()
        for res in range(2,max_res+1):
            name, model = self.block(res)
            Discriminator.add_module(name, model)
        return Discriminator

    def __init__(self,max_resolution):
        super(D_paper, self).__init__()
        self.res_channels = {2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}
        self.max_resolution = max_resolution
        self.D = self.create_graph(max_res=max_resolution)
        print(self.D)

    def forward(self, x,lod_in,phase,TRANSITION):
        if phase==2:
            x=self.from_RGBlayer[-1](x)
            return self.D[0](x)
        else:
            if TRANSITION is True:
                alpha=lod_in-phase
                x=lerp(self.from_RGBlayer[-2](F.avg_pool2d(x,2)),self.D[phase-2](self.from_RGBlayer[-1](x)),alpha)
                for res in range(phase-1,1,-1):
                    x=self.D[res-2](x)
                return x
            else:
                x=self.from_RGBlayer[-1](x)
                for res in range(phase,1,-1):
                    x=self.D[res-2](x)
                return x

if __name__=="__main__":
    G=G_paper(8)
    block=G.G[4]
    a=torch.zeros([24, 512, 32, 32])
    print(block(a).shape)
