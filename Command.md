# cifar-10

## --alpha 0.5

**1. CrossFreeze **

```bash
python main_crossfreeze.py --dataset cifar10 --model resnet --alg crossfreeze \
  --epochs 50 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --gamma 1.0 --ld 1.0 --alpha 0.5 \
  --m_tr 500 --shard_per_user 0 --gpu 0
```

**2. FedRep**

```bash
python main_fedrep.py --dataset cifar10 --model resnet --alg fedrep \
  --epochs 50 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --local_rep_ep 1 --alpha 0.5 \
  --m_tr 500 --shard_per_user 0 --gpu 0
```

**3. FedProto **

```bash
python main_fedproto.py --dataset cifar10 --model resnet --alg fedproto \
  --epochs 50 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --ld 1.0 --alpha 0.5 \
  --m_tr 500 --shard_per_user 0 --gpu 0
```

**4. FedAvg**

```bash
python main_fedavg.py --dataset cifar10 --model resnet --alg fedavg \
  --epochs 50 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --alpha 0.5 \
  --m_tr 500 --shard_per_user 0 --gpu 0
```

## --alpha 0.1

1.CrossFreeze 

```bash
python main_crossfreeze.py --dataset cifar10 --model resnet --alg crossfreeze \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 64 \
  --lr 0.01 --gamma 0.5 --ld 1.0 --alpha 0.1 --shard_per_user 0 \
  --m_tr 2500 --gpu 0
```

2.FedAvg 

```bash
python main_fedavg.py --dataset cifar10 --model resnet --alg fedavg \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --alpha 0.1 --shard_per_user 0 \
  --m_tr 2500 --gpu 0
```

3.FedRep

```bash
python main_fedrep.py --dataset cifar10 --model resnet --alg fedrep \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 64 \
  --lr 0.01 --local_rep_ep 1 --alpha 0.1 --shard_per_user 0 \
  --m_tr 2500 --gpu 0
```

4. FedProto

```bash
python main_fedproto.py --dataset cifar10 --model resnet --alg fedproto \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --ld 1.0 --alpha 0.1 --shard_per_user 0 \
  --m_tr 2500 --gpu 0
```

# cifar-100

## 100 epochs

1. CrossFreeze

```bash
python main_crossfreeze.py --dataset cifar100 --model resnet --alg crossfreeze \
  --epochs 100 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 64 \
  --lr 0.01 --gamma 2.0 --ld 1.0 --alpha 0.1 --shard_per_user 0 \
  --m_tr 500 --num_classes 100 --gpu 0
```

2.FedAvg

```bash
python main_fedavg.py --dataset cifar100 --model resnet --alg fedavg \
  --epochs 100 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --alpha 0.1 --shard_per_user 0 \
  --m_tr 500 --num_classes 100 --gpu 0
```

3.FedRep

```bash
python main_fedrep.py --dataset cifar100 --model resnet --alg fedrep \
  --epochs 100 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 64 \
  --lr 0.01 --local_rep_ep 1 --alpha 0.1 --shard_per_user 0 \
  --m_tr 500 --num_classes 100 --gpu 0
```

4.FedProto

```bash
python main_fedproto.py --dataset cifar100 --model resnet --alg fedproto \
  --epochs 100 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --ld 1.0 --alpha 0.1 --shard_per_user 0 \
  --m_tr 500 --num_classes 100 --gpu 0
```

5.pFedEdit

```bash
python main_pfededit.py --dataset cifar100 --model resnet --alg pfededit \
  --epochs 100 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --edit_ratio 0.4 --subset_ratio 0.1 --alpha 0.1 --shard_per_user 0\
  --m_tr 500 --num_classes 100 --gpu 0
```

6.Ditto

```bash
python main_ditto.py --dataset cifar100 --model resnet --alg ditto \
--epochs 100 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
--lr 0.01 --lam_ditto 1.0 --alpha 0.1 --shard_per_user 0 \
--m_tr 500 --num_classes 100 --gpu 0
```

7.MOON

```bash
python main_moon.py --dataset cifar100 --model resnet --alg moon \
  --epochs 100 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 64 \
  --lr 0.01 --mu_moon 1.0 --temperature 0.5 --alpha 0.1 \
  --m_tr 500 --num_classes 100 --shard_per_user 0 --gpu 0
```

## 200 epochs

1.CrossFreeze

```bash
python main_crossfreeze.py --dataset cifar100 --model resnet --alg crossfreeze \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 64 \
  --lr 0.01 --gamma 2.0 --ld 1.0 --alpha 0.1 --shard_per_user 0 \
  --m_tr 500 --num_classes 100 --gpu 0
```

2.FedAvg

```bash
python main_fedavg.py --dataset cifar100 --model resnet --alg fedavg \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --alpha 0.1 --shard_per_user 0 \
  --m_tr 500 --num_classes 100 --gpu 0
```

3.FedRep

```bash
python main_fedrep.py --dataset cifar100 --model resnet --alg fedrep \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 64 \
  --lr 0.01 --local_rep_ep 1 --alpha 0.1 --shard_per_user 0 \
  --m_tr 500 --num_classes 100 --gpu 0
```

4.FedProto

```bash
python main_fedproto.py --dataset cifar100 --model resnet --alg fedproto \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --ld 1.0 --alpha 0.1 --shard_per_user 0 \
  --m_tr 500 --num_classes 100 --gpu 0
```

5.pFedEdit

```bash
python main_pfededit.py --dataset cifar100 --model resnet --alg pfededit \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --edit_ratio 0.4 --subset_ratio 0.1 --alpha 0.1 --shard_per_user 0\
  --m_tr 500 --num_classes 100 --gpu 0
```

6.Ditto

```bash
python main_ditto.py --dataset cifar100 --model resnet --alg ditto \
--epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
--lr 0.01 --lam_ditto 1.0 --alpha 0.1 --shard_per_user 0 \
--m_tr 500 --num_classes 100 --gpu 0
```

7.MOON

```bash
python main_moon.py --dataset cifar100 --model resnet --alg moon \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 64 \
  --lr 0.01 --mu_moon 1.0 --temperature 0.5 --alpha 0.1 \
  --m_tr 500 --num_classes 100 --shard_per_user 0 --gpu 0
```

# tiny-imagenet

## 100 epochs

1. CrossFreeze

```bash
python main_crossfreeze.py --dataset tinyimagenet --model resnet --alg crossfreeze \
  --num_classes 200 --epochs 100 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 32 \
  --lr 0.01 --gamma 1.0 --ld 1.0 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

2. FedAvg

```bash
python main_fedavg.py --dataset tinyimagenet --model resnet --alg fedavg \
  --num_classes 200 --epochs 100 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 32 \
  --lr 0.001 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

3.FedProto

```bash
python main_fedproto.py --dataset tinyimagenet --model resnet --alg fedproto \
  --num_classes 200 --epochs 100 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 32 \
  --lr 0.01 --ld 1.0 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

4.FedRep

```bash
python main_fedrep.py --dataset tinyimagenet --model resnet --alg fedrep \
  --num_classes 200 --epochs 100 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 32 \
  --lr 0.001 --local_rep_ep 5 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

5.pFedEdit

```bash
python main_pfededit.py --dataset tinyimagenet --model resnet --alg pfededit \
  --num_classes 200 --epochs 100 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --edit_ratio 0.4 --subset_ratio 0.1 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

6.Ditto

```bash
python main_ditto.py --dataset tinyimagenet --model resnet --alg ditto \
  --epochs 100 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --lam_ditto 1.0 --alpha 0.1 --shard_per_user 0 \
  --m_tr 1000 --num_classes 200 --gpu 0
```

7.MOON

```bash
python main_moon.py --dataset tinyimagenet --model resnet --alg moon \
  --epochs 100 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 32 \
  --lr 0.01 --mu_moon 0.5 --temperature 0.5 --alpha 0.1 \
  --m_tr 1000 --num_classes 200 --shard_per_user 0 --gpu 0
```

## 200 epochs

1. CrossFreeze

```bash
python main_crossfreeze.py --dataset tinyimagenet --model resnet --alg crossfreeze \
  --num_classes 200 --epochs 200 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 32 \
  --lr 0.01 --gamma 1.0 --ld 1.0 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

2. FedAvg

```bash
python main_fedavg.py --dataset tinyimagenet --model resnet --alg fedavg \
  --num_classes 200 --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 32 \
  --lr 0.001 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

3.FedProto

```bash
python main_fedproto.py --dataset tinyimagenet --model resnet --alg fedproto \
  --num_classes 200 --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 32 \
  --lr 0.01 --ld 1.0 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

4.FedRep

```bash
python main_fedrep.py --dataset tinyimagenet --model resnet --alg fedrep \
  --num_classes 200 --epochs 200 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 32 \
  --lr 0.001 --local_rep_ep 5 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

5.pFedEdit

```bash
python main_pfededit.py --dataset tinyimagenet --model resnet --alg pfededit \
  --num_classes 200 --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --edit_ratio 0.4 --subset_ratio 0.1 --alpha 0.1 \
  --m_tr 1000 --shard_per_user 0 --gpu 0
```

6.Ditto

```bash
python main_ditto.py --dataset tinyimagenet --model resnet --alg ditto \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 \
  --lr 0.01 --lam_ditto 1.0 --alpha 0.1 --shard_per_user 0 \
  --m_tr 1000 --num_classes 200 --gpu 0
```

7.MOON

```bash
python main_moon.py --dataset tinyimagenet --model resnet --alg moon \
  --epochs 200 --num_users 20 --frac 0.5 --local_ep 10 --local_bs 32 \
  --lr 0.01 --mu_moon 0.5 --temperature 0.5 --alpha 0.1 \
  --m_tr 1000 --num_classes 200 --shard_per_user 0 --gpu 0
```

