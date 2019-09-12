# Convolutional autoencoder for human activity recognition
This folder contains the code for training convolutional autoencoder based representation for human activity recognition (HAR). 

Please create the following folders: 
`models` and `encoded_data`. The `models` folder stores the autoencoder models after they are trained. On a  similar vein, the `encoded_data` directory stores the representations extracted from the trained autoencoder. 

In order to train the model for Opportunity: ``python main.py --model conv_ae_flat --num_gpu 0 --dataset opportunity --mlp_norm True  --num_epochs 150``
