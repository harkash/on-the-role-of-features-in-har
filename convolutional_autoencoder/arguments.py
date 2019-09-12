import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for the Convolutional Autoencoder models')
    # Data loading and setup
    parser.add_argument('-w', '--window', type=int, default=30, help='Window size')
    parser.add_argument('-op', '--overlap', type=int, default=15)

    # Training parameters
    parser.add_argument('-b', '--batch_size', type=int, default=129)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--pretrain_num_epochs', type=int, default=150, help='number of epochs to train the autoencoder')
    parser.add_argument('--num_mlp_epochs', type=int, default=300, help='number of epochs to train the MLP')
    parser.add_argument('--model', type=str, required=True, help='picking the model')
    parser.add_argument('--num_gpu', type=str, default=0, help='Selecting the GPU to execute it with')
    parser.add_argument('--mlp_norm', type=bool, default=True, help='Choosing whether to perform normalization on the '
                                                                    'features extracted from the CAE')
    parser.add_argument('--dataset', type=str, default='opportunity', help='Choosing the dataset to perform the '
                                                                           'training on')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation function for obtaining the '
                                                                       'attention part of the network')
    parser.add_argument('--train_data_fraction', type=float, default=1.0, help='Fraction of the training set used')

    # Setting the latent size of the autoencoder
    parser.add_argument('--latent_size', type=int, default=100, help='Embedding size')

    args = parser.parse_args()

    # Setting the input size based on the dataset
    if args.dataset == 'opportunity':
        args.input_size = 77
        args.num_classes = 18
        args.root_dir = '/coc/pcba1/hharesamudram3/afresh/data/opportunity'
        args.data_file = 'opportunity_hammerla.mat'
        args.conv_size = 512 * 1 * 4
    elif args.dataset == 'skoda':
        args.input_size = 60
        args.num_classes = 11
        args.num_clusters = 11
        args.root_dir = '/coc/pcba1/hharesamudram3/afresh/data/skoda'
        args.data_file = 'Skoda.mat'
        args.conv_size = 512 * 1 * 3
    elif args.dataset == 'pamap2':
        args.input_size = 52
        args.num_classes = 12
        args.root_dir = '/coc/pcba1/hharesamudram3/afresh/data/pamap2'
        args.data_file = 'PAMAP2.mat'
        args.conv_size = 512 * 1 * 3
        # TODO: Figure out what to do with USC-HAD and Daphnet
    elif args.dataset == 'usc_had':
        args.input_size = 6
        args.num_classes = 12
        args.root_dir = '/coc/pcba1/hharesamudram3/afresh/data/usc-had'
        args.data_file = 'usc-had.mat'
        args.conv_size = 512 * 7 * 1
    elif args.dataset == 'daphnet':
        args.input_size = 9
        args.num_classes = 3
        args.root_dir = '/coc/pcba1/hharesamudram3/afresh/data/daphnet/dataset_fog_release'
        args.data_file = 'daphnet_gait.mat'
        args.conv_size = 512 * 3 * 1

    args.device = torch.device("cuda:" + str(args.num_gpu) if torch.cuda.is_available() else "cpu")
    return args
