import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import imageio
import torchvision.transforms as transforms
import copy
from .poi_util import *
import math
from Meta_Sift.models import ResNet18, PreActResNet18
from .util import *

device = 'cuda'

def get_dataset(args):
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),])
    trainset = h5_dataset(args.dataset_root, True, None)
    args.num_classes = len(np.unique(np.array(trainset.targets)))
    train_poi_set, poi_idx = poi_dataset(trainset, poi_methond=args.corruption_type, transform=train_transform, poi_rates=args.corruption_ratio,random_seed=args.random_seed, tar_lab=args.tar_lab)

    return train_poi_set, poi_idx
    

def build_training(args):
    model = ResNet18(num_classes = args.num_classes).cuda()
    optimizer_a = torch.optim.AdamW(model.parameters(), lr=2.5e-5 * args.batch_size / 32, betas=(0.9, 0.95), weight_decay=0.05)

    vnet = nnVent(1, 100, 150, 1).cuda()
    optimizer_c = torch.optim.SGD(vnet.parameters(), args.v_lr)
    return model, optimizer_a, vnet, optimizer_c

def train_sifter(args, dataset):
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    mnet_list = []
    vnet_list = []
    for i in range(args.num_sifter):
        print("-----------Training sifter number: " + str(i) + "-----------")
        model, optimizer_a, vnet, optimizer_c = build_training(args)
        grad_models, grad_optimizers = build_grad_models(args, model)
        model, optimizer_a = warmup(model, optimizer_a, train_dataloader, args)
        raw_meta_model = ResNet18(num_classes = args.num_classes).cuda()
        for epoch in range(args.res_epochs):
            train_iter = tqdm(enumerate(train_dataloader), total=int(len(dataset)/args.batch_size)+1)
            for iteration, (input_train, target_train) in train_iter:
                input_var,target_var = input_train.cuda(), target_train.cuda()

                # virtual training
                meta_model = copy.deepcopy(raw_meta_model)
                meta_model.load_state_dict(model.state_dict())
                y_f_hat = meta_model(input_var)
                cost = criterion(y_f_hat, target_var)
                cost_v = torch.reshape(cost, (len(cost), 1))
                v_lambda = vnet(cost_v.data)
                v_lambda = v_lambda.view(-1)
                v_lambda = norm_weight(v_lambda)
                l_f_meta = torch.sum(v_lambda * cost)

                # virtual backward & update
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta,(meta_model.parameters()),create_graph=True, allow_unused=True)

                # compute gradient gates and update the model
                new_grads,_ = compute_gated_grad(grads, grad_models, args.top_k, args.num_act)
                pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.meta_lr)
                pseudo_optimizer.load_state_dict(optimizer_a.state_dict())
                pseudo_optimizer.meta_step(new_grads)

                res_y_f_hat = meta_model(input_var)
                res_cost = criterion(res_y_f_hat, target_var)
                res_cost_v = torch.reshape(res_cost, (len(res_cost), 1))
                res_v_bf_lambda = vnet(res_cost_v.data)
                res_v_bf_lambda = res_v_bf_lambda.view(-1)
                res_v_lambda = 1-res_v_bf_lambda
                res_v_lambda = norm_weight(res_v_lambda)

                valid_loss = -torch.sum((res_v_lambda) * res_cost)
                
                optimizer_c.zero_grad()
                for go in grad_optimizers:
                    go.zero_grad()
                valid_loss.backward()
                optimizer_c.step()
                for go in grad_optimizers:
                    go.step()
                del grads, new_grads

                #actuall update
                y_f = model(input_var)
                cost_w = criterion(y_f, target_var)
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))

                with torch.no_grad():
                    w_new = vnet(cost_v)

                w_new = w_new.view(-1)#
                w_new = norm_weight(w_new)
                l_f = torch.sum(w_new * cost_w)

                optimizer_a.zero_grad()
                l_f.backward()
                optimizer_a.step()
        vnet_list.append(copy.deepcopy(vnet))
        mnet_list.append(copy.deepcopy(model))
    return vnet_list, mnet_list

def test_sifter(args, dataset, vnet_list, mnet_list):
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    v_res = np.zeros((args.num_sifter, len(dataset)), dtype=np.float32)
    for i in range(args.num_sifter):
        v = np.zeros((len(dataset)), dtype=np.float32)
        meta_model = mnet_list[i]
        meta_model.eval()
        vnet = vnet_list[i]
        # meta_model.train()
        for b, (images, labels) in tqdm(enumerate(test_dataloader),total=int(len(dataset) / args.batch_size)):
            input_var, target_var = images.cuda(), labels.cuda()
            y_f_hat = meta_model(input_var)
            cost = criterion(y_f_hat, target_var)
            cost_v = torch.reshape(cost, (len(cost), 1))
            
            v_lambda = vnet(cost_v.data)
            batch_size = v_lambda.size()[0]
            v_lambda = v_lambda.view(-1)
            
            zero_idx = b*batch_size
            v[zero_idx:zero_idx+batch_size] = v_lambda.detach().cpu().numpy()
        
        v_res[i,:] = copy.deepcopy(v)
    return v_res

def get_sifter_result(args, dataset, v_res, total_pick = 1000):

    class_per = []
    for i in np.unique(dataset.targets):
        percent = len(np.where(np.array(dataset.targets) == i)[0])/len(dataset)
        class_per.append(math.ceil(total_pick*percent))

    new_mat = np.mean(v_res,axis=0)
    new_idx = []
    for i in range(args.num_classes):
        pick_p = class_per[i]
        tar_idx = np.where(np.array(dataset.targets) == i)[0]
        p_tail = (len(tar_idx) - pick_p)/len(tar_idx)*100
        cutting = np.percentile(new_mat[tar_idx],p_tail)
        tar_new_idx = np.where(new_mat[tar_idx]>=cutting)[0]
        if tar_new_idx.shape[0] > pick_p:
            tar_new_idx = tar_new_idx[:pick_p]
        new_idx.append(tar_idx[tar_new_idx])
    new_idx = [i for item in new_idx for i in item]
    new_idx = np.array(new_idx)
    return new_idx

class MetaSiftSubset(Dataset):
    """
    根据原始数据集索引构造子集。

    indices 是相对于传入 dataset 的索引。
    子集保留 targets、labels 等属性，并将 transform 操作转发到底层数据集。
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

        if hasattr(dataset, "targets"):
            self.targets = [
                dataset.targets[int(i)] for i in self.indices
            ]

        if hasattr(dataset, "labels"):
            self.labels = [
                dataset.labels[int(i)] for i in self.indices
            ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        return self.dataset[original_index]

    def __getattr__(self, name):
        # 防止 deepcopy 初始化过程中发生递归调用
        if name in {"dataset", "indices"}:
            raise AttributeError(name)

        dataset = self.__dict__.get("dataset", None)
        if dataset is None:
            raise AttributeError(name)

        return getattr(dataset, name)

    def __setattr__(self, name, value):
        # transform 修改需要作用于底层数据集
        if name == "transform" and "dataset" in self.__dict__:
            setattr(self.dataset, name, value)
        else:
            object.__setattr__(self, name, value)


def _meta_sift_single(args, dataset, total_pick=1000):
    """
    执行一次完整的 Meta-Sift 筛选。

    输入：
        args
        dataset
        total_pick

    输出：
        相对于当前 dataset 的样本索引。
    """

    set_seed(args.random_seed)

    test_poi_set = copy.deepcopy(dataset)

    test_poi_set.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
    ])

    vnet_list, mnet_list = train_sifter(args, dataset)

    v_res = test_sifter(
        args,
        test_poi_set,
        vnet_list,
        mnet_list
    )

    selected_indices = get_sifter_result(
        args,
        test_poi_set,
        v_res,
        total_pick=total_pick
    )

    return np.asarray(selected_indices, dtype=np.int64).reshape(-1)


def meta_sift(
    args,
    dataset,
    total_pick=1000,
    coarse_ratio=0.5,
    coarse_sifter_ratio=0.5,
):
    """
    Two-stage Lightweight Meta-Sift.

    Stage 1: Coarse screening
        在完整数据集上进行初步筛选，得到候选样本集。

    Stage 2: Fine screening
        在候选样本集上重新训练 Sifter，并筛选最终样本。

    Args:
        args:
            原始 Meta-Sift 参数。
        dataset:
            输入数据集，例如 BackdoorBench 中的 bd_train。
        total_pick:
            最终需要筛选的样本数量。
        coarse_ratio:
            第一阶段候选样本数量占原始数据集的比例。
            例如 0.2 表示保留约 20%。
        coarse_sifter_ratio:
            第一阶段使用的 Sifter 数量占 args.num_sifter 的比例。
            例如 0.5 表示使用一半的 Sifter。

    Returns:
        final_indices:
            np.ndarray，最终样本在原始输入 dataset 中的索引。
    """

    dataset_size = len(dataset)

    # ---------------------------------
    # Parameter validation
    # ---------------------------------

    if total_pick <= 0 or total_pick > dataset_size:
        raise ValueError(
            f"total_pick must be in [1, {dataset_size}], "
            f"got {total_pick}"
        )

    if not 0 < coarse_ratio <= 1:
        raise ValueError(
            f"coarse_ratio must be in (0, 1], got {coarse_ratio}"
        )

    if not 0 < coarse_sifter_ratio <= 1:
        raise ValueError(
            "coarse_sifter_ratio must be in (0, 1], "
            f"got {coarse_sifter_ratio}"
        )

    if args.num_sifter <= 0:
        raise ValueError(
            f"args.num_sifter must be positive, got {args.num_sifter}"
        )

    # ---------------------------------
    # Stage 1: Coarse screening
    # ---------------------------------

    coarse_args = copy.deepcopy(args)

    # 第一阶段使用较少的 Sifter
    coarse_args.num_sifter = max(
        1,
        math.ceil(args.num_sifter * coarse_sifter_ratio)
    )

    # 第一阶段候选样本数量
    # 至少需要保留 total_pick 个样本
    coarse_pick = max(
        total_pick,
        math.ceil(dataset_size * coarse_ratio)
    )

    coarse_pick = min(coarse_pick, dataset_size)

    print("\n========== Meta-Sift Coarse Stage ==========")
    print(f"Original dataset size: {dataset_size}")
    print(f"Coarse candidate ratio: {coarse_ratio}")
    print(f"Coarse candidate count: {coarse_pick}")
    print(
        f"Coarse num_sifter: "
        f"{coarse_args.num_sifter}/{args.num_sifter}"
    )

    # 在完整数据集上执行第一次筛选
    coarse_indices = _meta_sift_single(
        coarse_args,
        dataset,
        total_pick=coarse_pick
    )

    coarse_indices = np.asarray(
        coarse_indices,
        dtype=np.int64
    ).reshape(-1)

    # 检查第一阶段结果
    if len(coarse_indices) < total_pick:
        raise RuntimeError(
            f"Coarse screening returned only {len(coarse_indices)} "
            f"samples, fewer than total_pick={total_pick}."
        )

    if np.any(coarse_indices < 0) or np.any(coarse_indices >= dataset_size):
        raise IndexError(
            "Coarse screening returned indices outside the original dataset."
        )

    print(
        f"Coarse screening finished: "
        f"{len(coarse_indices)} candidates selected."
    )

    # ---------------------------------
    # Stage 2: Fine screening
    # ---------------------------------

    print("\n========== Meta-Sift Fine Stage ==========")
    print(f"Fine num_sifter: {args.num_sifter}")
    print(f"Fine input count: {len(coarse_indices)}")
    print(f"Final target count: {total_pick}")

    # 根据第一阶段索引构造候选数据集
    candidate_dataset = MetaSiftSubset(
        dataset,
        coarse_indices
    )

    # 第二阶段使用原始 args.num_sifter
    # 并在候选数据集上重新执行 Meta-Sift
    fine_local_indices = _meta_sift_single(
        args,
        candidate_dataset,
        total_pick=total_pick
    )

    fine_local_indices = np.asarray(
        fine_local_indices,
        dtype=np.int64
    ).reshape(-1)

    # 检查第二阶段索引
    if np.any(fine_local_indices < 0) or np.any(
        fine_local_indices >= len(coarse_indices)
    ):
        raise IndexError(
            "Fine screening returned indices outside the candidate dataset."
        )

    # ---------------------------------
    # Map indices back to original dataset
    # ---------------------------------

    final_indices = coarse_indices[fine_local_indices]

    final_indices = np.asarray(
        final_indices,
        dtype=np.int64
    ).reshape(-1)

    print("\n========== Meta-Sift Finished ==========")
    print(f"Final selected count: {len(final_indices)}")
    print(f"Requested count: {total_pick}")

    return final_indices