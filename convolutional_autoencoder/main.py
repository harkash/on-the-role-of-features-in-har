import os
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from arguments import parse_args
from data_loader import HAR_Dataset, MLP_Dataset
from model import MLP_Classifier, Convolutional_Autoencoder_Flat
from trainer import train_ae, train_mlp, perform_inference

# Setting the seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Parsing the arguments
    args = parse_args()
    print(args)

    # Setting the GPU to run the training process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.num_gpu)
    torch.cuda.set_device(int(args.num_gpu))

    if args.model == 'conv_ae_flat':
        model = Convolutional_Autoencoder_Flat(dataset=args.dataset, conv_size=args.conv_size,
                                               latent_size=args.latent_size)
    model.cuda()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of trainable parameters is {}'.format(num_parameters))
    # exit(0)

    # Loading the data for the autoencoder training
    full_model_name = 'models/' + args.dataset + '/' + args.model + '_pretrain_epochs_' + \
                      str(args.pretrain_num_epochs) + '_latent_size_' + str(args.latent_size) + \
                      '_train_data_fraction_' + str(args.train_data_fraction) + '_full.pkl'

    datasets = {x: HAR_Dataset(args=args, phase=x) for x in ['train', 'val', 'test']}
    data_loaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                                  num_workers=1, pin_memory=True) for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    # Training the model if necessary
    if os.path.exists(full_model_name):
        print('Loading the already trained model')
        model.load_state_dict(torch.load(full_model_name))
        model.cuda()
    else:
        print('Training the model')
        model = train_ae(model, data_loaders=data_loaders, dataset_sizes=dataset_sizes, args=args,
                         num_epochs=args.pretrain_num_epochs)
        torch.save(model.state_dict(), full_model_name)

    # Next, we perform the forward pass on the model and save the encodings
    encoded_data_loc = 'encoded_data/' + args.dataset + '/' + args.model + '_pretrain_epochs_' + \
                      str(args.pretrain_num_epochs) + '_latent_size_' + str(args.latent_size) + \
                      '_train_data_fraction_' + str(args.train_data_fraction) + '_data.pkl'

    encoded_data = perform_inference(model, data_loaders, dataset_sizes, args.batch_size, args.latent_size, args.mlp_norm)
    with open(encoded_data_loc, 'wb') as f:
        pickle.dump(encoded_data, f, pickle.HIGHEST_PROTOCOL)
    print('Saved the encoded data into a pickle file')

    # Then we proceed to perform classification using a MLP
    mlp_datasets = {x: MLP_Dataset(loc=encoded_data_loc, phase=x) for x in ['train', 'val', 'test']}
    mlp_data_loaders = {x: DataLoader(mlp_datasets[x], batch_size=args.batch_size,
                                      shuffle=True if x == 'train' else False, num_workers=1,
                                      pin_memory=True) for x in ['train', 'val', 'test']}

    mlp_model = MLP_Classifier(input_size=args.latent_size)
    mlp_model.cuda()

    optimizer = optim.Adam(mlp_model.parameters(), lr=args.learning_rate)
    exp_lr_scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    best_mlp_model = train_mlp(mlp_model, mlp_data_loaders, criterion, optimizer, exp_lr_scheduler,
                                 args.num_mlp_epochs, dataset_sizes)

    print('Training complete!')
    exit(0)