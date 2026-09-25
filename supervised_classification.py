import argparse

import numpy as np
import torch
import torch.distributions as D
import torchvision
from sklearn.neural_network import MLPClassifier
from torchvision import transforms

from datasets import Fast_3DShapes, Fast_MNIST, Fast_SVHN
from load_model import load_model_from_save_dict

# 用 sklearn 的一层 MLP 分类器预测标签，最后报告 test accuracy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)    # 读取训练好的模型
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embedding_mode", choices=["sample", "mean"], default="sample")   #todo: 决定latent embedding是用均值还是采样。每个输入x不是直接得到确定的zj，有两种方法得到embedding
    parser.add_argument("--max_iter", type=int, default=200)    # MLPClassifier 最大训练迭代次数
    parser.add_argument("--classifier_seed", type=int, default=0)   # 控制 MLP 分类器的随机种子
    return parser.parse_args()

# 根据已经保存到模型里的 run_args.dataset决定使用哪个数据集
def make_loaders(args, device):
    if args.dataset == "fast_mnist":
        train_data = Fast_MNIST("./data", train=True, download=True, device=device)
        test_data = Fast_MNIST("./data", train=False, download=True, device=device)
        train_data.targets = train_data.targets.unsqueeze(1)
        test_data.targets = test_data.targets.unsqueeze(1)
    elif args.dataset == "fast_svhn":
        train_data = Fast_SVHN("./data", split="train", download=True, device=device)
        test_data = Fast_SVHN("./data", split="test", download=True, device=device)
        train_data.labels = train_data.labels.unsqueeze(1)  # 增加一个纬度，后面统一处理多个label
        test_data.labels = test_data.labels.unsqueeze(1)
    elif args.dataset == "fast_3dshapes":
        train_data = Fast_3DShapes(
            train=True,
            device=device,
            train_frac=args.threedshapes_train_frac,
            factors_variation_dict=args.factors_variation_dict,
            factors_label_list=args.factors_label_list,
            seed=args.seed,
            # 使用模型训练时的参数。保证eval使用的配置和模型训练时对应
        )
        test_data = Fast_3DShapes(
            train=False,
            device=device,
            train_frac=args.threedshapes_train_frac,
            factors_variation_dict=args.factors_variation_dict,
            factors_label_list=args.factors_label_list,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 创建DataLoader，把数据分成batch
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader

# 把所有输入图片转换成 MFCVAE 的 latent representations。输入所有图片->mfcvae encoder->得到z1z2并保存
@torch.no_grad()
def collect_embeddings(model, loader, embedding_mode, run_args):
    model.eval()    # evaluation mode
    z_parts, y_parts = [[] for _ in range(model.J_n_mixtures)], []

    for x, y in loader:
        if run_args.model_type in ["fc_shared", "fc_per_facet_enc_shared_dec", "fc_vlae"]:
            x = x.view(x.size(0), -1).float()
            # FC全连接模型需要把图片flatten展平
        elif run_args.model_type in ["conv_vlae"]:
            x = x.float()
            # 卷积网络不能flatten

        mu_list, log_var_list = model.encode(x)
        if embedding_mode == "mean":
            batch_z_list = mu_list  #zj = uj
        else:
            batch_z_list = [
                D.Independent(
                    D.Normal(loc=mu_list[j], scale=torch.sqrt(torch.exp(log_var_list[j]))),
                    1,
                ).sample()
                for j in range(model.J_n_mixtures)
            ]

        for j, z_j in enumerate(batch_z_list):
            z_parts[j].append(z_j.detach().cpu().numpy())   # z_parts[0] = 所有样本的 z1 z_parts[1] = 所有样本的 z2
        y_parts.append(y.detach().cpu().numpy())    # 保存真实标签（shape / floor_hue）

    z_list = [np.concatenate(parts, axis=0) for parts in z_parts]   # 拼接所有batch
    z_all = np.concatenate(z_list, axis=1)  # 把所有zj拼接成一个大的z（z = [z1,z2]）
    labels = np.concatenate(y_parts, axis=0)
    if labels.ndim == 1:
        labels = labels[:, None]
    return z_list, z_all, labels.astype(int)

# 训练监督分类器 - 真正使用 MLPClassifier多层感知机分类器
def fit_and_score(x_train, y_train, x_test, y_test, max_iter, seed):
    clf = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(x_train, y_train)   # 训练
    return clf.score(x_test, y_test)    # 测试 （accuracy = 正确预测数量/测试样本数量）

# mfcvae的 progressive training渐进式训练
def initialize_eval_progressive_state(model, run_args):
    if not getattr(model, "do_progressive_training", False):
        return

    final_eval_epoch = sum(run_args.n_epochs_per_progressive_step) - 1  # 计算最终epoch
    alpha_enc, alpha_dec, gamma_kl_z, gamma_kl_c = model.compute_progressive_training_coefficients(
        final_eval_epoch, 0
    )   # 重新计算alpha_enc, alpha_dec, gamma_kl_z, gamma_kl_c, 最终恢复模型在最终progressive training阶段的状态
    # 重要。否则eval时可能处于错误 progressive-training coefficient 状态
    model.alpha_enc_fade_in_list = alpha_enc
    model.alpha_dec_fade_in_list = alpha_dec
    model.gamma_kl_z_list = gamma_kl_z
    model.gamma_kl_c_list = gamma_kl_c
    model.encoder.alpha_enc_fade_in_list = alpha_enc
    model.decoder.alpha_dec_fade_in_list = alpha_dec


def main():
    cli_args = parse_args()
    model, run_args = load_model_from_save_dict(cli_args.model_path, map_location=cli_args.device)

    model = model.to(cli_args.device)
    model.eval()
    initialize_eval_progressive_state(model, run_args)

    print("Model device:", next(model.parameters()).device)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    train_loader, test_loader = make_loaders(run_args, cli_args.device)

    print("dataset:", run_args.dataset)
    if run_args.dataset == "fast_3dshapes":
        print("labels:", run_args.factors_label_list)
    else:
        print("labels:", ["digit_class"])
    print("z_j_dim_list:", run_args.z_j_dim_list)
    print("embedding_mode:", cli_args.embedding_mode)

    z_train_list, z_train_all, y_train = collect_embeddings(
        model, train_loader, cli_args.embedding_mode, run_args
    )
    z_test_list, z_test_all, y_test = collect_embeddings(
        model, test_loader, cli_args.embedding_mode, run_args
    )

    x_train_by_name = {f"z{j + 1}": z_train for j, z_train in enumerate(z_train_list)}
    x_test_by_name = {f"z{j + 1}": z_test for j, z_test in enumerate(z_test_list)}
    x_train_by_name["z"] = z_train_all
    x_test_by_name["z"] = z_test_all

    label_names = (
        run_args.factors_label_list if run_args.dataset == "fast_3dshapes" else ["digit_class"]
    )

    print("\nSupervised classification test accuracy")
    print("embedding," + ",".join(label_names))
    for emb_name in [f"z{j + 1}" for j in range(model.J_n_mixtures)] + ["z"]:
        scores = []
        for label_idx in range(y_train.shape[1]):
            score = fit_and_score(
                x_train_by_name[emb_name],
                y_train[:, label_idx],
                x_test_by_name[emb_name],
                y_test[:, label_idx],
                cli_args.max_iter,
                cli_args.classifier_seed,
            )
            scores.append(score)
        print(emb_name + "," + ",".join(f"{100.0 * score:.2f}" for score in scores))


if __name__ == "__main__":
    main()
