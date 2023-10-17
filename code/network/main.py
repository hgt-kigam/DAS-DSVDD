from .DSVDD_rectangle import Deep_SVDD, pretrain_autoencoder


def DeepSVDD(args):
    """Builds the neural network."""

    implemented_networks = (406)
    assert args.net_name in implemented_networks

    net = None

    if args.net_name == 406:
        net = Deep_SVDD(args)

    return net


def Autoencoder(args):
    """Builds the corresponding autoencoder network."""

    implemented_networks = (406)
    assert args.net_name in implemented_networks

    ae_net = None

    if args.net_name == 406:
        ae_net = pretrain_autoencoder(args)

    return ae_net
