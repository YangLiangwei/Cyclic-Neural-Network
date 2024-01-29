import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'mnist', type = str,
                        help = 'Dataset to use')
    parser.add_argument('--model', default = 'mlp', type = str,
                        help = 'model within each block')
    parser.add_argument('--epochs', default = 9999999, type = int,
                        help = 'Total epochs')
    parser.add_argument('--batch_size', default = 100, type = int,
                        help = 'Batch size')
    parser.add_argument('--T', default = 3, type = int,
                        help = 'Time step')
    parser.add_argument('--lr', default = 1e-3, type = float,
                        help  = 'Learning rate for each neuron')
    parser.add_argument('--weight_decay', default = 3e-4, type = float,
                        help = 'weight decay for optimizer')
    parser.add_argument('--momentum', default = 0.9, type = float,
                        help = 'momentum for optimizer')
    parser.add_argument('--readout_lr', default = 1e-3, type = float,
                        help = 'Learning rate for readout layer')
    parser.add_argument('--readout_weight_decay', default = 1e-6, type = float,
                        help = 'weight decay for readout layer')
    parser.add_argument('--neurons', default = 4, type = int,
                        help = 'Computation neuron number')
    parser.add_argument('--threshold', default = 2.0, type = float,
                        help = 'Threshold for neuron')
    parser.add_argument('--connect_rate', default = 1.0, type = float,
                        help = 'Neuron connect rate with feature')
    parser.add_argument('--h_feat', default = 32, type = int,
                        help = "Hidden dimension for each single neuron")
    parser.add_argument('--type', default = 'single', type = str,
                        help = "Net type single, line or cycle")
    parser.add_argument('--patience', default = 10, type = int,
                        help = "Patience for early stop")
    parser.add_argument('--class_num', default = 10, type = int,
                        help = "Class number")
    parser.add_argument('--world_size', default = 8, type = int,
                        help = "World size")
    parser.add_argument('--seed', default = 42, type = int,
                        help = "random seed")
    parser.add_argument('--input_neurons', default = -1, type = int,
                        help = "input neurons")
    parser.add_argument('--gpu', default = 0, type = int,
                        help = "gpu")
    parser.add_argument('--weight', default = 1, type = float,
                        help = "weight for loss")
    parser.add_argument('--out_dim', default = 2000, type = int,
                        help = "output dimension for each neuron")
    parser.add_argument('--label', default = "nan", type = str,
                        help = 'label to classify logs')
    parser.add_argument('--goodness_threshold', default = 1.0, type = float,
                        help = 'goodness threshold for neuron')
    
    # hyparams for graph generator
    parser.add_argument('--synapses', default = 3, type = int,
                        help = 'synapses between neurons')
    parser.add_argument('--k', default = 4, type = int,
                        help = 'number of nearest neighbors connected for WSGraph')
    parser.add_argument('--p', default = 0.5, type = float,
                        help = 'probability of rewiring each edge in WS Graph')

    # hyparams for transformer
    parser.add_argument('--patch_size', default = 4, type = int, help = 'patch size')
    parser.add_argument('--in_dim', default = 64, type = int, help = 'in_dim')
    parser.add_argument('--image_size', default = 28, type = int, help = 'image size')
    parser.add_argument('--channels', default = 1, type = int, help = 'channels')
    parser.add_argument('--patch_num', default = 1, type = int, help = 'patch number')
    
    args = parser.parse_args()

    return args