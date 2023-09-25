from .DSVDD_224 import Deep_SVDD_224, pretrain_autoencoder_224
from .DSVDD_96 import Deep_SVDD_96, pretrain_autoencoder_96
from .DSVDD_96_100 import Deep_SVDD_96_100, pretrain_autoencoder_96_100
from .DSVDD_1000 import Deep_SVDD_1024, pretrain_autoencoder_1024


def Deep_SVDD(args):
    
    """Builds the neural network."""

    implemented_networks = (224, 96, 96_100, 1024)
    assert args.net_name in implemented_networks

    net = None

    if args.net_name == 224:
        net = Deep_SVDD_224(args)

    if args.net_name == 96:
        net = Deep_SVDD_96(args)

    if args.net_name == 96_100:
        net = Deep_SVDD_96_100(args)

    if args.net_name == 1024:
        net = Deep_SVDD_1024(args)

    return net


def Autoencoder(args):
    """Builds the corresponding autoencoder network."""

    implemented_networks = (224, 96, 96_100, 1024)
    assert args.net_name in implemented_networks

    ae_net = None

    if args.net_name == 224:
        ae_net = pretrain_autoencoder_224(args)

    if args.net_name == 96:
        ae_net = pretrain_autoencoder_96(args)

    if args.net_name == 96_100:
        ae_net = pretrain_autoencoder_96_100(args)

    if args.net_name == 1024:
        ae_net = pretrain_autoencoder_1024(args)

    return ae_net