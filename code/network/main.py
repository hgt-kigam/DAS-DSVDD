from .DSVDD_CNN import Deep_SVDD, pretrain_autoencoder
from .DSVDD_LSTM import Deep_SVDD_LSTM, pretrain_autoencoder_LSTM
from .DSVDD_MNIST import Deep_SVDD_MNIST, pretrain_autoencoder_MNIST


def DeepSVDD(args):
    """Builds the neural network."""

    implemented_networks = ('CNN', 'LSTM', 'MNIST')
    assert args.net_name in implemented_networks

    net = None

    if args.net_name == 'CNN':
        net = Deep_SVDD(args)

    if args.net_name == 'LSTM':
        net = Deep_SVDD_LSTM(args)

    if args.net_name == 'MNIST':
        net = Deep_SVDD_MNIST(args)

    return net


def Autoencoder(args):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('CNN', 'LSTM', 'MNIST')
    assert args.net_name in implemented_networks

    ae_net = None

    if args.net_name == 'CNN':
        ae_net = pretrain_autoencoder(args)

    if args.net_name == 'LSTM':
        ae_net = pretrain_autoencoder_LSTM(args)

    if args.net_name == 'MNIST':
        ae_net = pretrain_autoencoder_MNIST(args)

    return ae_net
