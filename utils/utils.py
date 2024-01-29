import torch
import torchvision
from utils import DataLoader
import numpy as np
import random
from utils.EarlyStop import EarlyStoppingCriterion
import logging
import os
import pdb
from sklearn.datasets import load_svmlight_file
# import torchtext
# from torchtext.datasets import IMDB
from torch.utils.data import Dataset
import pickle


def get_MNIST_partition(args, partition):
    if partition in ["train", "val", "train_val"]:
        mnist = torchvision.datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        mnist = torchvision.datasets.MNIST(
            './data',
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    if partition == "train":
        mnist = torch.utils.data.Subset(mnist, range(50000))
    elif partition == "val":
        mnist = torchvision.datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        mnist = torch.utils.data.Subset(mnist, range(50000, 60000))

    return mnist


def get_FashionMNIST_partition(args, partition):
    if partition in ["train", "val", "train_val"]:
        fashionmnist = torchvision.datasets.FashionMNIST(
            './data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        fashionmnist = torchvision.datasets.FashionMNIST(
            './data',
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    if partition == "train":
        fashionmnist = torch.utils.data.Subset(fashionmnist, range(50000))
    elif partition == "val":
        fashionmnist = torchvision.datasets.FashionMNIST(
            './data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        fashionmnist = torch.utils.data.Subset(fashionmnist, range(50000, 60000))

    return fashionmnist

def get_cifar10_partition(args, partition):
    if partition in ["train", "val", "train_val"]:
        cifar10 = torchvision.datasets.CIFAR10(
            './data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        cifar10 = torchvision.datasets.CIFAR10(
            './data',
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    if partition == "train":
        cifar10 = torch.utils.data.Subset(cifar10, range(40000))
    elif partition == "val":
        cifar10 = torchvision.datasets.CIFAR10(
            './data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        cifar10 = torch.utils.data.Subset(cifar10, range(40000, 50000))

    return cifar10

def get_imdb_partition(args, partition):
    if partition in ["train"]:
        # imdb = torch.load('./data/aclImdb/imdb_train.pt')
        with open('./data/aclImdb/imdb_train.pkl', 'rb') as f:
            imdb = pickle.load(f)
    
    elif partition in ["val"]:
        # imdb = torch.load('./data/aclImdb/imdb_val.pt')
        with open('./data/aclImdb/imdb_valid.pkl', 'rb') as f:
            imdb = pickle.load(f)

    elif partition in ["test"]:
        # imdb = torch.load('./data/aclImdb/imdb_test.pt')
        with open('./data/aclImdb/imdb_test.pkl', 'rb') as f:
            imdb = pickle.load(f)

    else:
        raise NotImplementedError

    return imdb

def get_newsgroup_partition(args, partition):
    if partition in ["train"]:
        newsgroup = torch.load('./data/20newsgroups/20newsgroups_train.pt')
    
    elif partition in ["val"]:
        newsgroup = torch.load('./data/20newsgroups/20newsgroups_valid.pt')

    elif partition in ["test"]:
        newsgroup = torch.load('./data/20newsgroups/20newsgroups_test.pt')

    else:
        raise NotImplementedError

    return newsgroup


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(args, inputs, labels):
    if "cuda" in args.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data(opt, partition):

    if opt.dataset == 'mnist':
        dataset = DataLoader.FF_MNIST(opt, partition)
    elif opt.dataset == 'fashionmnist':
        dataset = DataLoader.FF_FashionMNIST(opt, partition)
    elif opt.dataset == 'cifar10':
        dataset = DataLoader.FF_CIFAR10(opt, partition)
    elif opt.dataset == 'imdb':
        dataset = DataLoader.FF_IMDB(opt, partition)
    elif opt.dataset == 'newsgroup':
        dataset = DataLoader.FF_Newsgroup(opt, partition)
    g = torch.Generator()
    g.manual_seed(opt.seed)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        persistent_workers=False
    )


def get_data_old(opt, partition):
    dataset = DataLoader.FF_MNIST(opt, partition)

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
        persistent_workers=False
    )

def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.epochs // 2):
        return lr * 2 * (1 + opt.epochs - epoch) / opt.epochs
    else:
        return lr

def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.lr
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.readout_lr
    )
    return optimizer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def config(args):
    setup_seed(args.seed)

    path = f"{args.dataset}_model_{args.model}_batch_size_{args.batch_size}_T_{args.T}_lr_{args.lr}_weight_decay_{args.weight_decay}_readout_lr_{args.readout_lr}_readout_weight_decay_{args.readout_weight_decay}_neurons_{args.neurons}_connect_rate_{args.connect_rate}_type_{args.type}_input_neurons_{args.input_neurons}_out_dim_{args.out_dim}_goodness_threshold_{args.goodness_threshold}_seed_{args.seed}_label_{args.label}"
    if os.path.exists('./logs/' + path + '.log'):
        os.remove('./logs/' + path + '.log')

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s  %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='./logs/' + path + '.log')
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    early_stop = EarlyStoppingCriterion(patience = args.patience, save_path = './best_models/' + path + '.pt')
    return early_stop