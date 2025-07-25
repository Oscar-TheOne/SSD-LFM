import functools
import torch
from torch.nn import init


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']
    # ----------------------------------------
    # UNet + DFCAN
    # ----------------------------------------
    if net_type == 'bf_unet':
        from models.network_bf_unet import BF_UNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'])
    elif net_type == 'bf_unet3d':
        from models.network_bf_unet3D import BF_UNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'])
    elif net_type == 'bf_unet3d2': 
        from models.network_bf_unet3D2 import BF_UNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'])
    elif net_type == 'rcan':
        from models.network_rcan import RCAN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'])
    elif net_type == 'rdn':
        from models.network_rdn import RDN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'])
    elif net_type == 'uformer':
        from models.network_uformer import Uformer as net
        netG = net(in_chans=opt_net['in_nc'],
                   img_size=opt_net['patch_size']) 
    elif net_type == 'wf_unet':
        from models.network_wf_unet import BF_UNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'])
    elif net_type == 'RCAN3D': 
        from models.network_RCAN3D import RCAN3D as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'])
    elif net_type == 'RCAU3D': 
        from models.network_RCAU3D import RCAU_test3 as net
        netG = net(in_channels=opt_net['in_nc'],num_classes=opt_net['out_nc'])
    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG

"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """
    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or (classname.find('Linear') != -1 and classname.find('Projection')==-1):

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
    net.apply(fn)
