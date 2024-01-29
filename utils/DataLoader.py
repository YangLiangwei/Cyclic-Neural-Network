import numpy as np
import torch
import pdb

from utils import utils

class FF_MNIST(torch.utils.data.Dataset):
    def __init__(self, args, partition, num_classes=10):
        self.opt = args
        self.mnist = utils.get_MNIST_partition(args, partition)
        self.num_classes = num_classes
        self.device = args.device
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes
        self.uniform_label = self.uniform_label.to(self.device)

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample.reshape(28 * 28),
            "neg_images": neg_sample.reshape(28 * 28),
            "neutral_sample": neutral_sample.reshape(28 * 28)
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def batch_fuse_label(self, samples, class_labels):
        b, d = samples.shape
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_labels), num_classes=self.num_classes
        ).to(self.device)
        pos_samples = samples.clone().reshape(b, 28, 28)

        pos_samples[:, 0, : self.num_classes] = one_hot_label
        return pos_samples.reshape(b, d)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).to(self.device)
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).to(self.device)
        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[:, 0, : self.num_classes] = self.uniform_label
        return z

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        sample = sample.to(self.device)
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label

        
class FF_FashionMNIST(torch.utils.data.Dataset):
    def __init__(self, args, partition, num_classes=10):
        self.opt = args
        self.fashionmnist = utils.get_FashionMNIST_partition(args, partition)
        self.num_classes = num_classes
        self.device = args.device
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes
        self.uniform_label = self.uniform_label.to(self.device)

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample.reshape(28 * 28),
            "neg_images": neg_sample.reshape(28 * 28),
            "neutral_sample": neutral_sample.reshape(28 * 28)
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.fashionmnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).to(self.device)
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).to(self.device)
        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[:, 0, : self.num_classes] = self.uniform_label
        return z

    def _generate_sample(self, index):
        # Get FashionMNIST sample.
        sample, class_label = self.fashionmnist[index]
        sample = sample.to(self.device)
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label


        
class FF_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, args, partition, num_classes=10):
        self.opt = args
        self.cifar10 = utils.get_cifar10_partition(args, partition)
        self.num_classes = num_classes
        self.device = args.device
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes
        self.uniform_label = self.uniform_label.to(self.device)

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample.reshape(32 * 32 * 3),
            "neg_images": neg_sample.reshape(32 * 32 * 3),
            "neutral_sample": neutral_sample.reshape(32 * 32 * 3)
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.cifar10)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).to(self.device)
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).to(self.device)
        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[:, 0, : self.num_classes] = self.uniform_label
        return z

    def _generate_sample(self, index):
        # Get CIFAR10 sample.
        sample, class_label = self.cifar10[index]
        sample = sample.to(self.device)
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label


class FF_IMDB(torch.utils.data.Dataset):
    def __init__(self, args, partition, num_classes=2):
        self.opt = args
        self.imdb = utils.get_imdb_partition(args, partition)
        self.num_classes = num_classes
        self.device = args.device
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes
        self.uniform_label = self.uniform_label.to(self.device)

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample.reshape(-1),
            "neg_images": neg_sample.reshape(-1),
            "neutral_sample": neutral_sample.reshape(-1)
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.imdb)

    def batch_fuse_label(self, samples, class_labels):
        b, d = samples.shape
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_labels), num_classes=self.num_classes
        ).to(self.device)
        pos_samples = samples.clone()

        pos_samples[:, :self.num_classes] = one_hot_label
        return pos_samples

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).to(self.device)
        pos_sample = sample.clone()
        pos_sample = torch.concat((one_hot_label, pos_sample))
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).to(self.device)
        neg_sample = sample.clone()
        neg_sample = torch.concat((one_hot_label, neg_sample))
        return neg_sample

    def _get_neutral_sample(self, z):
        z = torch.concat((self.uniform_label, z))
        return z

    def _generate_sample(self, index):
        # Get IMDB sample.
        sample, class_label = self.imdb[index]
        sample = sample.to(self.device)
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label


class FF_Newsgroup(torch.utils.data.Dataset):
    def __init__(self, args, partition, num_classes=20):
        self.opt = args
        self.newsgroup = utils.get_newsgroup_partition(args, partition)
        self.num_classes = num_classes
        self.device = args.device
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes
        self.uniform_label = self.uniform_label.to(self.device)

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample.reshape(-1),
            "neg_images": neg_sample.reshape(-1),
            "neutral_sample": neutral_sample.reshape(-1)
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.newsgroup)

    def batch_fuse_label(self, samples, class_labels):
        b, d = samples.shape
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_labels), num_classes=self.num_classes
        ).to(self.device)
        pos_samples = samples.clone()

        pos_samples[:, :self.num_classes] = one_hot_label
        return pos_samples

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).to(self.device)
        pos_sample = sample.clone()
        pos_sample = torch.concat((one_hot_label, pos_sample))
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).to(self.device)
        neg_sample = sample.clone()
        neg_sample = torch.concat((one_hot_label, neg_sample))
        return neg_sample

    def _get_neutral_sample(self, z):
        z = torch.concat((self.uniform_label, z))
        return z

    def _generate_sample(self, index):
        # Get newsgroup sample.
        sample, class_label = self.newsgroup[index]
        sample = sample.to(self.device)
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label