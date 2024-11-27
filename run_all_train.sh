#!/bin/bash

# 定义所有可能的参数组合
# optimizers=("sgd" "sgd8bit" "adam" "adam8bit" "eva" "eva8bit" "eva8bit_cpu" "shampoo4bit" "shampoo4bit_svd")
optimizers=("eva8bit")
learning_rates_sgd=("0.01")
learning_rates_adam=("0.0001")
learning_rates_eva=("0.1")
learning_rates_shampoo=("0.001")
# optimizers=("shampoo4bit" "shampoo4bit_svd")
# learning_rates_sgd=("0.04" "0.01")
# learning_rates_adam=("0.001" "0.01")
# learning_rates_eva=("0.4" "0.1")
datasets=("mnist" "cifar10")
mnist_net="autoencoder"
mnist_epochs=50
# cifar10_nets=("vit" "res20")
cifar10_nets=()
cifar10_epochs=150
batch_size=512

# 遍历所有参数组合并执行相应命令
for opt in "${optimizers[@]}"; do
    for dataset in "${datasets[@]}"; do
        # 设置 net 和 n_epochs
        if [[ "$dataset" == "mnist" ]]; then
            net="$mnist_net"
            n_epochs="$mnist_epochs"
        elif [[ "$dataset" == "cifar10" ]]; then
            n_epochs="$cifar10_epochs"
        fi

        # 设置不同优化器的学习率
        case $opt in
            "sgd"|"sgd8bit")
                for lr in "${learning_rates_sgd[@]}"; do
                    if [[ "$dataset" == "mnist" ]]; then
                        python train_cifar10.py --opt "$opt" --n_epochs "$n_epochs" --bs "$batch_size" --net "$net" --dataset "$dataset" --lr "$lr"
                    elif [[ "$dataset" == "cifar10" ]]; then
                        for net in "${cifar10_nets[@]}"; do
                            python train_cifar10.py --opt "$opt" --n_epochs "$n_epochs" --bs "$batch_size" --net "$net" --dataset "$dataset" --lr "$lr"
                        done
                    fi
                done
                ;;
            "adam"|"adam8bit")
                for lr in "${learning_rates_adam[@]}"; do
                    if [[ "$dataset" == "mnist" ]]; then
                        python train_cifar10.py --opt "$opt" --n_epochs "$n_epochs" --bs "$batch_size" --net "$net" --dataset "$dataset" --lr "$lr"
                    elif [[ "$dataset" == "cifar10" ]]; then
                        for net in "${cifar10_nets[@]}"; do
                            python train_cifar10.py --opt "$opt" --n_epochs "$n_epochs" --bs "$batch_size" --net "$net" --dataset "$dataset" --lr "$lr"
                        done
                    fi
                done
                ;;
            "shampoo4bit"|"shampoo4bit_svd"|"shampoo32bit")
                for lr in "${learning_rates_shampoo[@]}"; do
                    if [[ "$dataset" == "mnist" ]]; then
                        python train_cifar10.py --opt "$opt" --n_epochs "$n_epochs" --bs "$batch_size" --net "$net" --dataset "$dataset" --lr "$lr"
                    elif [[ "$dataset" == "cifar10" ]]; then
                        for net in "${cifar10_nets[@]}"; do
                            python train_cifar10.py --opt "$opt" --n_epochs "$n_epochs" --bs "$batch_size" --net "$net" --dataset "$dataset" --lr "$lr"
                        done
                    fi
                done
                ;;
            "eva"|"eva8bit"|"eva8bit_cpu"|"eva_fused"|"eva8bit_fused")
                for lr in "${learning_rates_eva[@]}"; do
                    if [[ "$dataset" == "mnist" ]]; then
                        python train_cifar10.py --opt "$opt" --n_epochs "$n_epochs" --bs "$batch_size" --net "$net" --dataset "$dataset" --lr "$lr"
                    elif [[ "$dataset" == "cifar10" ]]; then
                        for net in "${cifar10_nets[@]}"; do
                            python train_cifar10.py --opt "$opt" --n_epochs "$n_epochs" --bs "$batch_size" --net "$net" --dataset "$dataset" --lr "$lr"
                        done
                    fi
                done
                ;;
        esac
    done
done
