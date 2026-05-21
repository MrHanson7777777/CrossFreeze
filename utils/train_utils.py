# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py
# credit goes to: Paul Pu Liang

from torchvision import datasets, transforms
from models.Nets import CNNCifar, CNNCifar100, RNNSent, MLP, CNN_FEMNIST, ResNet18_CrossFreeze
from utils.sampling import noniid, dirichlet_split_noniid, iid, global_stratified_prune, enforce_min_samples
import numpy as np
import os
import json

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def get_data(args):
    # 安全检查：Shard 模式暂不支持 global pruning
    if args.shard_per_user > 0 and args.data_usage < 1.0:
        raise NotImplementedError("Error: --shard_per_user cannot be used with --data_usage yet. Set --shard_per_user 0.")

    # 设置类别数
    if args.dataset == 'cifar100': args.num_classes = 100
    elif args.dataset == 'tinyimagenet': args.num_classes = 200
    else: args.num_classes = 10

    # 内部辅助函数：处理裁剪和一致性划分
    def partition_data_consistent(dataset_train, dataset_test, args):
        # 1. 全局裁剪 (Global Pruning)
        if args.data_usage < 1.0:
            print(f"✂️ Applying Global Stratified Pruning: Using {args.data_usage*100}% of training data...")
            kept_indices = global_stratified_prune(dataset_train, args.data_usage, args.seed)
        else:
            kept_indices = np.arange(len(dataset_train))

        # 获取裁剪后的标签子集
        if hasattr(dataset_train, 'targets'):
            all_targets = np.array(dataset_train.targets)
            test_targets = np.array(dataset_test.targets)
        elif hasattr(dataset_train, 'imgs'):
            all_targets = np.array([t for _, t in dataset_train.imgs])
            test_targets = np.array([t for _, t in dataset_test.imgs])
        else:
            raise ValueError("Dataset format error")

        subset_targets = all_targets[kept_indices]
        dict_users_train = {}
        
        # 2. 划分训练集
        if args.iid:
            # IID 逻辑：使用确定性 Shuffle 替代 Set
            # [修复] 避免数据丢失：使用 np.array_split 均匀分配
            rng = np.random.default_rng(args.seed)
            rel_indices = rng.permutation(len(kept_indices))
            
            # 使用 array_split 自动处理余数分配
            client_splits = np.array_split(rel_indices, args.num_users)
            
            for i in range(args.num_users):
                client_rel_idxs = client_splits[i]
                dict_users_train[i] = kept_indices[client_rel_idxs] # 映射回绝对索引
            
            # 测试集直接 IID
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            # Non-IID (Dirichlet) 逻辑
            # A. 划分训练集，并强制返回概率矩阵 dist_matrix
            # client_map_rel 存储的是相对索引 (0 ~ len(subset))
            client_map_rel, dist_matrix = dirichlet_split_noniid(
                subset_targets, args.alpha, args.num_users, return_dist=True
            )
            
            # B. 映射回绝对索引
            for client_idx, rel_idxs in client_map_rel.items():
                dict_users_train[client_idx] = kept_indices[rel_idxs]
            
            # C. 划分测试集：必须使用相同的 dist_matrix！
            # 确保 Client i 在 Train 和 Test 中面临相同的类别分布
            dict_users_test = dirichlet_split_noniid(
                test_targets, args.alpha, args.num_users, use_dist=dist_matrix
            )

        # === [关键] 最小生存保障 (min=1) ===
        # 只解决 "0数据" 导致的 Crash，允许 "1样本" 存在，最大程度保留异质性
        dict_users_train = enforce_min_samples(dict_users_train, min_samples=1)
        
        return dict_users_train, dict_users_test

    # ================= 数据加载入口 =================
    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, dict_users_test = partition_data_consistent(dataset_train, dataset_test, args)

    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        dict_users_train, dict_users_test = partition_data_consistent(dataset_train, dataset_test, args)
    
    elif args.dataset == 'tinyimagenet':
        data_dir = './data/tiny-imagenet-200/'
        train_path = os.path.join(data_dir, 'train')
        test_path = os.path.join(data_dir, 'val')
        
        # Transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        dataset_train = datasets.ImageFolder(root=train_path, transform=transform_train)
        dataset_test = datasets.ImageFolder(root=test_path, transform=transform_test)
        dict_users_train, dict_users_test = partition_data_consistent(dataset_train, dataset_test, args)
    
    elif args.dataset == 'mnist':
        # MNIST 暂时保持原逻辑 (或按需修改)
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.data_usage < 1.0: print("Warning: data_usage not implemented for MNIST yet.")
        dict_users_train, _ = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, _ = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes)

    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_model(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100(args=args).to(args.device)
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        net_glob = CNN_FEMNIST(args=args).to(args.device)
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet':
        net_glob = ResNet18_CrossFreeze(args=args).to(args.device)
        print("Initialized ResNet18-0.5x from Scratch (Kaiming Init)")
    elif 'sent140' in args.dataset:
        net_glob = model = RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob
