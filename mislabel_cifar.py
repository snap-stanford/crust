import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import copy
import warnings

class MISLABELCIFAR10(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, mislabel_type='agnostic', mislabel_ratio=0.5, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(MISLABELCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.gen_mislabeled_data(mislabel_type=mislabel_type, mislabel_ratio=mislabel_ratio)
        
    def gen_mislabeled_data(self, mislabel_type, mislabel_ratio):
        """Gen a list of imbalanced training data, and replace the origin data with the generated ones."""
        new_targets = []
        num_cls = np.max(self.targets) + 1
        
        if mislabel_type == 'agnostic':
            for i, target in enumerate(self.targets):
                if np.random.rand() < mislabel_ratio:
                    new_target = target
                    while new_target == target:
                        new_target = np.random.randint(num_cls)
                    new_targets.append(new_target)
                else:
                    new_targets.append(target)
        elif mislabel_type == 'asym':
            ordered_list = np.arange(num_cls)
            while True:
                permu_list = np.random.permutation(num_cls)
                if np.any(ordered_list == permu_list):
                    continue
                else:
                    break
            for i, target in enumerate(self.targets):
                if np.random.rand() < mislabel_ratio:
                    new_target = permu_list[target]
                    new_targets.append(new_target)
                else:
                    new_targets.append(target)
        else:
            warnings.warn('Noise type is not listed')

        self.real_targets = self.targets
        self.targets = new_targets
        self.whole_data = self.data.copy()
        self.whole_targets = copy.deepcopy(self.targets)
        self.whole_real_targets = copy.deepcopy(self.real_targets)

    def switch_data(self):
        self.data = self.whole_data
        self.targets = self.whole_targets
        self.real_targets = self.whole_real_targets

    def adjust_base_indx_tmp(self, idx):
        new_data = self.whole_data[idx, ...]
        targets_np = np.array(self.whole_targets)
        new_targets = targets_np[idx].tolist()
        real_targets_np = np.array(self.whole_real_targets)
        new_real_targets = real_targets_np[idx].tolist()
        self.data = new_data
        self.targets = new_targets
        self.real_targets = new_real_targets

    def adjust_base_indx_perma(self, idx):
        new_data = self.whole_data[idx, ...]
        targets_np = np.array(self.whole_targets)
        new_targets = targets_np[idx].tolist()
        real_targets_np = np.array(self.whole_real_targets)
        new_real_targets = real_targets_np[idx].tolist()
        self.whole_data = new_data
        self.whole_targets = new_targets
        self.whole_real_targets = new_real_targets
        self.data = self.whole_data
        self.targets = self.whole_targets
        self.real_targets = self.whole_real_targets

    def estimate_label_acc(self):
        targets_np = np.array(self.targets) 
        real_targets_np = np.array(self.real_targets)
        label_acc = np.sum((targets_np == real_targets_np)) / len(targets_np)
        return label_acc

    def fetch(self, targets):
        whole_targets_np = np.array(self.whole_targets)
        uniq_targets = np.unique(whole_targets_np)
        idx_dict = {}
        for uniq_target in uniq_targets:
            idx_dict[uniq_target] = np.where(whole_targets_np == uniq_target)[0]

        idx_list = []
        for target in targets:
            idx_list.append(np.random.choice(idx_dict[target.item()], 1))
        idx_list = np.array(idx_list).flatten()
        imgs = []
        for idx in idx_list:
            img = self.whole_data[idx]
            img = Image.fromarray(img)
            img = self.transform(img)
            imgs.append(img[None, ...])
        train_data = torch.cat(imgs, dim=0)
        return train_data

    def __getitem__(self, index):  
        img, target, real_target = self.data[index], self.targets[index], self.real_targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, real_target, index   

class MISLABELCIFAR100(MISLABELCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    