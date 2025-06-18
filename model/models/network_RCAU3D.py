"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
# from torchsummary import summary
import torch
import time

from network_RCAN3D import ResidualGroup,default_conv

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super(Conv3DBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=(3, 3, 3),
                               padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        self.conv2 = nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=(3, 3, 3),
                               padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)

        self.relu = nn.ReLU()

        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (
                    last_layer == True and num_classes != None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2),
                                          stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels + res_channels, out_channels=in_channels // 2,
                               kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=(3, 3, 3),
                               padding=(1, 1, 1))

        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels // 2, out_channels=num_classes, kernel_size=(1, 1, 1))

    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual != None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out


class RCAU_test3(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """

    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512,
                 kernel_size=3,reduction=16,act=nn.ReLU(True),res_scale = 1,conv = default_conv,n_resgroups=3,n_resblocks=5) -> None:
        super(RCAU_test3, self).__init__()

        #n_feats = bottleneck_channel

        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)

        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck=True)
        modules_body = [ResidualGroup(conv, bottleneck_channel, kernel_size, reduction, act=act, res_scale=res_scale,
                                        n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        self.main_body = nn.Sequential(*modules_body)

        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

        # modules_body1 = [ResidualGroup(conv, level_1_chnls, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        # modules_body1.append(conv(level_1_chnls, level_1_chnls, kernel_size))
        # self.body1 = nn.Sequential(*modules_body1)
        #
        # modules_body2 = [ResidualGroup(conv, level_2_chnls, kernel_size, reduction, act=act, res_scale=res_scale,
        #                                n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        # modules_body2.append(conv(level_2_chnls, level_2_chnls, kernel_size))
        # self.body2 = nn.Sequential(*modules_body2)
        #
        # modules_body3 = [ResidualGroup(conv, level_3_chnls, kernel_size, reduction, act=act, res_scale=res_scale,
        #                                n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        # modules_body3.append(conv(level_3_chnls, level_3_chnls, kernel_size))
        # self.body3 = nn.Sequential(*modules_body3)

    def forward(self, input):
        # Analysis path forward feed

        out, residual_level1 = self.a_block1(input)
        # out = self.body1(out)
        out, residual_level2 = self.a_block2(out)
        # out = self.body2(out)
        out, residual_level3 = self.a_block3(out)
        # out = self.body3(out)

        out, _ = self.bottleNeck(out)
        out = self.main_body(out)
        # Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out

if __name__ == '__main__':
#     #Configurations according to the Xenopus kidney dataset
    model = RCAU_test3(in_channels=1, num_classes=1,n_resgroups=3,n_resblocks=5)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
#     start_time = time.time()
#     summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")
#     print("--- %s seconds ---" % (time.time() - start_time))
    x = torch.rand(1, 1, 8, 128, 128)
    model.eval()
    y=model(x)
    print(y.shape)