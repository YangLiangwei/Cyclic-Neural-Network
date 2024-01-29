import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import dgl
import pdb
from tqdm import tqdm
import argparse
import random
import numpy as np
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import concurrent.futures as futures
import torch.distributed as dist
from utils.DataLoader import FF_MNIST
from utils.utils import config, get_data, preprocess_inputs, update_learning_rate
from utils.parser import parse_args
import logging
import math
from models_ff import Transformer, ViT 
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

class Net(torch.nn.Module):

    def __init__(self, args, input_dim, label_dim, num_neuron, num_synapse):
        super().__init__()
        self.device = args.device
        self.count = 0
        self.label_dim = label_dim
        self.world_size = args.world_size
        self.epochs = args.epochs
        self.ff_loss = nn.BCEWithLogitsLoss()
        # self.graph = nx.gnm_random_graph(num_neuron, num_synapse)
        self.graph = nx.empty_graph(args.neurons)
        if args.type == 'complete':
            self.graph = nx.complete_graph(args.neurons)
        if args.type == 'WSGraph':
            self.graph = nx.watts_strogatz_graph(args.neurons, args.k, args.p)
        if args.type == 'BAGraph':
            self.graph = nx.barabasi_albert_graph(args.neurons, args.synapses)
        if args.type =='RandomGraph':
            self.graph = nx.gnm_random_graph(args.neurons, args.synapses)

        self.g = dgl.from_networkx(self.graph)
        if args.type == 'line':
            for i in range(len(self.g.nodes()) - 1):
                self.g.add_edges(i, i + 1)
        if args.type == 'cycle':
            for i in range(len(self.g.nodes()) - 1):
               self.g.add_edges(i, i + 1)
            self.g.add_edges(i + 1, 0)

        self.neurons = torch.nn.ModuleList()
        self.num_compute_neurons = len(self.g.nodes())
        self.num_neurons = self.num_compute_neurons
        self.classification_loss = nn.CrossEntropyLoss()

        self.input_neurons = args.input_neurons
        self.out_dim = args.out_dim
        self.linear_input_dim = 0
        # self.linear_input_dim = input_dim

        self.add_input_edges(args, input_dim, label_dim, list(range(self.input_neurons)))
        self.build_neurons(args)


        self.num_neurons = len(self.neurons)
        self.args = args
        self.T = args.T
        self.weight = args.weight
        self.threshold = args.threshold
        self.world_size = args.world_size
        self.batch_size = args.batch_size
        self.goodness_threshold = args.goodness_threshold
        self.g = self.g.to(self.device)

        if args.model == 'transformer':
            self.input_vit = ViT(args = args, image_size = args.image_size, patch_size = args.patch_size, dim = 64, heads = 4, pool = 'cls', channels = args.channels, dim_head = 8, dropout = 0., emb_dropout = 0.)
            self.input_vit.to(self.device)

        self.linear_classifier = nn.Sequential(nn.Linear(self.linear_input_dim, label_dim, bias = True)).to(self.device)

        params = []
        for neuron in self.neurons:
            if not neuron.input_neu:
                params += neuron.parameters()

        params_global = []  
        for param in self.linear_classifier.parameters():
            params_global += [param]

        self.optimizer = torch.optim.SGD([
            {"params": params,
             "lr": args.lr,
             "weight_decay": args.weight_decay,
             "momentum": args.momentum,
             },
             {"params": params_global,
              "lr": args.readout_lr,
              "weight_decay": args.readout_weight_decay,
              "momentum": args.momentum}])
        # self._init_weights()

        # new_weights = torch.load('./linear_weight.pth')
        # self.neurons[0].layer.load_state_dict(new_weights)

    def _init_weights(self):
        for neuron in self.neurons:
            for m in neuron.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(
                        m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                    )
                    torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def neuron_loss(self, z, labels):
        g_pos = z[labels == 1]
        g_neg = z[labels == 0]
        tensor_input = torch.cat([g_neg - self.threshold, -g_pos + self.threshold])
        loss = torch.log(1 + torch.exp(tensor_input)).mean()
        return loss, 0

    def layer_norm(self, x, eps = 1e-8):
        return x / torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim = True) + eps)

    def get_accuracy(self, logits, labels):
        return (torch.argmax(logits, dim=1) == labels).float().mean().item()

    def add_input_edges(self, args, input_dim, label_dim, target_nodes):
        src = [i + self.num_neurons for i in range(input_dim)] * len(target_nodes)
        dst = []
        for node in target_nodes:
            dst += [node] * input_dim

        mask = torch.rand(len(src)) < args.connect_rate

        src = np.array(src)[mask]
        dst = np.array(dst)[mask]
        self.g.add_edges(src, dst)
      
        # for i in range(label_dim):
        #     for node in target_nodes:
        #         self.g.add_edges(self.num_neurons, node)
        #     self.num_neurons += 1
        
    def build_neurons(self, args):
        self.num_neurons = 0
        for node in self.g.nodes():
            degree = len(self.g.in_edges(node)[0])
            if degree == 0:
                self.neurons.append(Neuron(args, self.num_neurons, input_neu = True))
            else:
                out_dim = self.out_dim
                self.neurons.append(Neuron(args, self.num_neurons, out_dim, device = self.device).to(self.device))
            self.num_neurons += 1

        for neuron in self.neurons:
            src_nodes = self.g.in_edges(neuron.id)[0]
            if len(src_nodes) == 0:
                pass
            else:
                out_dim = neuron.out_dim
                neuron.build_params(sum([self.neurons[src_node].out_dim for src_node in src_nodes]), out_dim)
                self.linear_input_dim += out_dim

    def _calc_ff_loss(self, z, labels):
        b, t, d = z.shape

        # pay attention to the max operation, it should be the same with readout function
        z = z.max(dim = 1)[0]
        sum_of_squares = torch.sum(z ** 2, dim=-1)
        # sum_of_squares = torch.sum(z, dim = -1)

        logits = sum_of_squares - self.goodness_threshold * z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def eval_neuron(self, neuron):
        sources = self.g.in_edges(neuron.id)[0]
        h_pos = neuron.model(self.feat_pos[:, sources])
        self.feat_pos_new[:, neuron.id] = h_pos.squeeze(1)
        return h_pos
        
        
    def neuron_test(self, partition, epoch = None):
        data_loader = get_data(self.args, partition)
        dataset = data_loader.dataset

        self.neurons.eval()
        accuracy = 0
        loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                batch_size = inputs['pos_images'].shape[0]

                goodness_label = []
                for label in range(self.label_dim):
                    goodness = 0
                    label = {'class_labels': torch.ones(batch_size) * label}
                    inputs, label = preprocess_inputs(self.args, inputs, label)
                    inputs_neu = inputs['neutral_sample']
                    inputs_with_label = dataset.batch_fuse_label(inputs_neu, label['class_labels'].long())

                    batch_size, dim = inputs["neutral_sample"].shape
                    if self.args.model == 'mlp':
                        neu = inputs_with_label.reshape(batch_size, 1, dim)
                    elif self.args.model == 'transformer':
                        neu = self.input_vit(inputs_with_label.reshape(batch_size, args.channels, -1))

                    self.init_features(neu)

                    for t in range(self.T):
                        for neuron in self.neurons:
                            if neuron.input_neu == False:
                                sources = self.g.in_edges(neuron.id)[0]
                                z = torch.cat([self.neurons[i].feat for i in sources], -1)
                                h, _ = neuron(z)
                                b, token, d = h.shape
                                if t == self.T-1:
                                    z = h.max(dim = 1)[0]
                                    sum_of_squares = torch.sum(z ** 2, dim=-1)
                                    goodness += sum_of_squares - z.shape[1]
                        self.renew_features()
                    goodness_label.append(goodness)
                goodness_label = torch.stack(goodness_label, dim = 1)
                labels_pred = torch.argmax(goodness_label, dim = 1)
                accuracy += (labels_pred == labels["class_labels"].to(self.device)).float().sum().item()

        accuracy = accuracy / len(data_loader.dataset)
        error_rate = (1 - accuracy) * 100
        logging.info("For epoch {}, {}, neurons error rate is {}".format(epoch, partition, error_rate))

        self.neurons.train()
        return error_rate


    def validate_or_test(self, partition, epoch = None):
        data_loader = get_data(self.args, partition)

        self.neurons.eval()
        self.linear_classifier.eval()
        accuracy = 0
        loss = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                batch_size = inputs['pos_images'].shape[0]
                inputs, labels = preprocess_inputs(self.args, inputs, labels)

                loss_batch, accuracy_batch = self.readout_model(
                    inputs, labels
                )
                loss += loss_batch * batch_size
                accuracy += accuracy_batch * batch_size
        loss = loss / len(data_loader.dataset)
        accuracy = accuracy / len(data_loader.dataset)
        error_rate = (1 - accuracy) * 100
        logging.info("For epoch {}, {}, loss is {}, error rate is {}".format(epoch, partition, loss, error_rate))

        self.neurons.train()
        self.linear_classifier.train()
        return loss, error_rate

    def init_features(self, input_feature):
        count = 0
        b, t, d = input_feature.shape
        for neuron in self.neurons:
            if neuron.input_neu == True:
                neuron.feat = input_feature[:, :, count].unsqueeze(2)
                count += 1
            else:
                neuron.feat = torch.zeros(b, t, neuron.out_dim, device = self.device)
    
    def renew_features(self):
        for neuron in self.neurons:
            if neuron.input_neu == False:
                neuron.feat = neuron.feat_new.clone().detach()
                # neuron.feat = neuron.feat_new

    def train_neuron(self, input_tuple):
        neuron, t = input_tuple
        # if t == self.T - 1:
        #     last = True
        # else:
        #     last = False
        last = True

        batch_size = self.batch_size
        sources = self.g.in_edges(neuron.id)[0]
        z = torch.cat([self.neurons[i].feat for i in sources], -1)
        z, hidden_mean = neuron(z)

        h_pos, h_neg = torch.split(z, batch_size, dim = 0)

        # self.feat_pos_new[:, neuron.id] = h_pos.squeeze(1).detach()
        # self.feat_neg_new[:, neuron.id] = h_neg.squeeze(1).detach()

        if last == False:
            return 0, 0
        else:
            posneg_labels = torch.zeros(z.shape[0]).to(self.device)
            posneg_labels[: batch_size] = 1
            loss, acc = self._calc_ff_loss(z, posneg_labels)
            # loss = self.neuron_loss(z, posneg_labels)[0]
            return loss, hidden_mean, acc
        
    def train(self):
        dataloader = get_data(self.args, 'train')

        for epoch in range(self.epochs):
            loss_ff_epoch = []
            loss_clf_epoch = []
            loss_epoch = []
            hidden_mean = []
            acc_neuron = []

            self.optimizer = update_learning_rate(self.optimizer, self.args, epoch)
            for inputs, labels in tqdm(dataloader):
                inputs, labels = preprocess_inputs(self.args, inputs, labels)

                batch_size, dim = inputs['pos_images'].shape
                if self.args.model == 'mlp':
                    pos = inputs['pos_images'].reshape(batch_size, 1, dim)
                    neg = inputs['neg_images'].reshape(batch_size, 1, dim)
                elif self.args.model == 'transformer':
                    pos = self.input_vit(inputs['pos_images'].reshape(batch_size, args.channels, -1))
                    neg = self.input_vit(inputs['neg_images'].reshape(batch_size, args.channels, -1))

                feature_tensor = torch.cat([pos, neg], 0).to(self.device)
                # pdb.set_trace()
                self.init_features(feature_tensor)

                self.optimizer.zero_grad()

                loss = torch.zeros(1).to(self.device)
                for t in range(self.T):
                    # with mp.Pool(self.world_size) as p:
                    #     res_batch = p.map(self.train_neuron, [(n,t) for n in self.neurons if n.input_neu == False])
                    #     loss_batch = [res[0] for res in res_batch]
                    #     hidden_mean_batch = [res[1] for res in res_batch]
                    
                    # if t == self.T - 1:
                    #     for i in range(len(loss_batch)):
                    #         loss += loss_batch[i]
                    #         hidden_mean.append(hidden_mean_batch[i].detach().float())

                    for neuron in self.neurons:
                        if neuron.input_neu == False:
                            loss_batch, hidden_mean_batch, acc_batch = self.train_neuron((neuron, t))
                            # if t == self.T - 1:
                                # Adding FF loss
                            loss += loss_batch
                            hidden_mean.append(hidden_mean_batch.detach().float())
                            acc_neuron.append(acc_batch)
                    self.renew_features()

                loss = loss / (self.num_compute_neurons * self.T)
                loss_ff_epoch.append(loss.detach().item())

                loss_clf, _ = self.readout_model(inputs, labels)
                loss_clf_epoch.append(loss_clf.detach().item())
                loss += self.weight * loss_clf

                loss_epoch.append(loss.item())
                loss.backward()
                self.optimizer.step()

            logging.info('hidden_mean')
            logging.info(sum(hidden_mean) / len(hidden_mean))
            logging.info('acc_neuron')
            logging.info(sum(acc_neuron) / len(acc_neuron))

            logging.info('ff_loss')
            logging.info(sum(loss_ff_epoch) / len(loss_ff_epoch))
            logging.info('clf_loss')
            logging.info(sum(loss_clf_epoch) / len(loss_clf_epoch))
            logging.info('loss')
            logging.info(sum(loss_epoch) / len(loss_epoch))
 
            loss_val, error_rate_val = self.validate_or_test('val', epoch)
            # self.neuron_test('test')
            early_stop(error_rate_val, self)
            # loss_test, accuracy_test = self.validate_or_test('test', epoch)
            if early_stop.early_stop:
                break
    
    def readout_model(self, inputs, labels):

        batch_size, dim = inputs["neutral_sample"].shape
        if self.args.model == 'mlp':
            neu = inputs['neutral_sample'].reshape(batch_size, 1, dim)
        elif self.args.model == 'transformer':
            neu = self.input_vit(inputs['neutral_sample'].reshape(batch_size, args.channels, -1))

        self.init_features(neu)

        linear_input = []
    
        # adding input to classifier
        # linear_input = [inputs['neutral_sample']]
        # linear_input = [self.layer_norm(inputs['neutral_sample'])]

        with torch.no_grad():
            for t in range(self.T):
                for neuron in self.neurons:
                    if neuron.input_neu == False:
                        sources = self.g.in_edges(neuron.id)[0]
                        z = torch.cat([self.neurons[i].feat for i in sources], -1)
                        h, _ = neuron(z)
                        b, token, d = h.shape
                        if t == self.T-1:
                            linear_input.append(self.layer_norm(h.max(dim = 1)[0].detach()))
                self.renew_features()

        output = self.linear_classifier(torch.cat(linear_input, dim = 1))
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = self.get_accuracy(output, labels["class_labels"])

        return classification_loss, classification_accuracy


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class Neuron(nn.Module):
    def __init__(self, args, id, out_features = 1, input_neu = False, device = None):
        super().__init__()
        # self.h_feat = [int(x) for x in args.h_feat.split(',')]
        self.h_feat = [args.h_feat]
        # self.activation = ReLU_full_grad()
        self.activation = nn.ReLU()
        self.id = id
        self.device = device
        self.input_neu = input_neu
        self.out_dim = out_features
        self.input_neu = input_neu
        self.model = args.model
    
    def build_params(self, in_features, out_features):
        if not self.input_neu:
            if self.model == 'mlp':
                self.layer = nn.Linear(in_features, out_features)
            elif self.model == 'transformer':
                self.layer = Transformer(in_dim = in_features, out_dim = out_features, depth = 1, heads = 3, dim_head = 256, mlp_dim = 2000)
            
            self.layer.to(self.device)
        
        # if not self.input_neu:
        #     self.layers = nn.ModuleList()
        #     self.layers.append(nn.Linear(in_features, self.h_feat[0]))
        #     for i in range(1, len(self.h_feat)):
        #         self.layers.append(nn.Linear(self.h_feat[i - 1], self.h_feat[i]))
        #     self.layers.append(nn.Linear(self.h_feat[-1], out_features))
        # self.layers.to(self.device)

    def init_output(self, outputs):
        self.feat = outputs
        self.feat_new = outputs.clone().detach()

    def forward(self, x):
        # x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-8)
        x_direction = x / (torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim = True)) + 1e-8)
        # for layer in self.layers[:-1]:
        #     x_direction = self.activation(layer(x_direction))
        # x_direction_linear = self.layers[-1](x_direction)
        # x_direction = self.activation(x_direction_linear)

        x_direction_linear = self.layer(x_direction)
        x_direction = self.activation(x_direction_linear)
        self.feat_new = x_direction
        return x_direction, x_direction_linear.mean()
   

if __name__ == "__main__":
    args = parse_args()
    early_stop = config(args)

    if args.input_neurons == -1:
        args.input_neurons = args.neurons
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(0)
    else:
        device = 'cpu'

    args.device = device

    # mp.set_start_method('spawn')
    if args.dataset == 'mnist': 
        input_dim = 784
    elif args.dataset == 'cifar10':
        input_dim = 3072
    elif args.dataset == 'imdb':
        input_dim = 770
    elif args.dataset == 'newsgroup':
        input_dim = 788

    if args.model == 'mlp':
        # net = Net(args, 784, 10, 500, 3000)
        net = Net(args, input_dim, args.class_num, 500, 3000)
    elif args.model == 'transformer':
        net = Net(args, args.patch_size * args.patch_size * args.channels, 10, 500, 3000)
    net.train()
    logging.info('loading best model for test')
    net.load_state_dict(torch.load(early_stop.save_path))
    loss_test, error_rate_test = net.validate_or_test('test')
    error_neu_val = net.neuron_test('test')
