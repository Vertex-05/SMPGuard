'''
Pre-activation Distributions Expose Backdoor Neurons

This file is modified based on the following source:
link : https://github.com/RJ-T/NIPS2022_EP_BNP.
The defense method is called bnp.

@article{zheng2022pre,
            title={Pre-activation Distributions Expose Backdoor Neurons},
            author={Zheng, Runkai and Tang, Rongjun and Li, Jianze and Liu, Li},
            journal={Advances in Neural Information Processing Systems},
            volume={35},
            pages={18667--18680},
            year={2022}}

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. reconstruct the layer norm for convnext and transformer
    7. draw the corresponding images of asr and acc according to different proportions
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. dde defense:
        a. calculate the entropy of each norm layer
        b. prune the model depend on the mask
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import copy
import os,sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense
import utils.defense_utils.mbns.mbns_model as mbns_model
from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

from utils.bd_dataset_v2 import dataset_wrapper_with_transform
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Meta_Sift.meta_sift import meta_sift


class MetaSiftDataset(Dataset):
    """
    Adapt BackdoorBench's poisoned dataset to Meta-Sift.

    Returns only:
        image, training_label

    The training label may already be poisoned.
    """

    def __init__(self, bd_dataset, transform=None):
        self.bd_dataset = bd_dataset
        self.transform = transform

        # Use the labels actually provided for model training,
        # NOT the original labels in field 4.
        self.targets = [
            int(bd_dataset[i][1])
            for i in range(len(bd_dataset))
        ]

    def __len__(self):
        return len(self.bd_dataset)

    def __getitem__(self, index):
        sample = self.bd_dataset[index]

        image = sample[0]
        label = int(sample[1])

        if self.transform is not None:
            image = self.transform(image)

        return image, label




def batch_entropy(x, step_size=0.1):
	n_bars = int((x.max()-x.min())/step_size)
	entropy = 0
	for n in range(n_bars):
		num = ((x > x.min() + n*step_size) * (x < x.min() + (n+1)*step_size)).sum(-1)
		p = num / x.shape[-1]
		entropy += - p * p.log().nan_to_num(0)
	return entropy

def adaptive_mad_threshold(kl_div, k=2.0, eps=1e-8):
    """
    Adaptive-A: Median + k * MAD

    kl_div: 当前归一化层所有通道的 KL 分数
    k: 控制阈值严格程度的参数
    """
    scores = kl_div.detach().reshape(-1)

    median = scores.median()
    mad = (scores - median).abs().median()

    # 将 MAD 转换为与标准差可比的稳健尺度
    robust_scale = 1.4826 * mad

    # 防止 MAD 为 0 时阈值完全退化
    robust_scale = robust_scale.clamp_min(eps)

    threshold = median + k * robust_scale

    return threshold

def evaluate_clean_acc(model, dataloader, criterion, device):
    """
    Evaluate clean accuracy and loss only.

    This function does NOT access the backdoor test set.
    """

    model.eval()
    model.to(device)

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    criterion = criterion.to(device)

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.long())

            predictions = outputs.argmax(dim=1)

            total_correct += predictions.eq(labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    clean_acc = total_correct / max(total_samples, 1)
    clean_loss = total_loss / max(total_samples, 1)

    return clean_acc, clean_loss

def collect_layer_kl_scores(net, trainloader, args):
    """
    Collect KL scores for each normalization layer once.
    All candidate u vectors reuse these fixed scores.
    """
    clean_data_loader = trainloader['clean_train']
    bd_data_loader = trainloader['bd_train']

    net.eval()

    bd_data = next(iter(bd_data_loader))[0].to(args.device)
    mixture_data = next(iter(clean_data_loader))[0].to(args.device)

    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.collect_stats = True

        if isinstance(m, nn.LayerNorm):
            m.collect_stats_clean = True
            m.collect_stats_bd = False

    # Collect clean-reference statistics
    with torch.no_grad():
        net(mixture_data)

    bn_clean_stats = {}

    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_clean_stats[name] = (
                m.batch_mean.detach().clone(),
                m.batch_var.detach().clone(),
            )

    # Collect backdoor-training statistics
    for m in net.modules():
        if isinstance(m, nn.LayerNorm):
            m.collect_stats_bd = True
            m.collect_stats_clean = False

    kl_scores = {}

    with torch.no_grad():
        net(bd_data)

        for name, m in net.named_modules():

            if isinstance(m, nn.BatchNorm2d):
                mean_1, var_1 = bn_clean_stats[name]
                mean_2 = m.running_mean
                var_2 = m.running_var

                eps = 1e-8
                var_1 = var_1.clamp_min(eps)
                var_2 = var_2.clamp_min(eps)

                kl_div = (
                    torch.log(var_2 / var_1)
                    + (var_1 + (mean_1 - mean_2).pow(2))
                    / (2 * var_2)
                    - 0.5
                )

                kl_scores[name] = kl_div.detach().clone()

            elif isinstance(m, nn.LayerNorm):
                var_1 = m.batch_var_clean
                var_2 = m.batch_var_bd
                mean_1 = m.batch_mean_clean
                mean_2 = m.batch_mean_bd

                eps = 1e-8
                var_1 = var_1.clamp_min(eps)
                var_2 = var_2.clamp_min(eps)

                kl_div = (
                    0.5 * torch.log(var_2 / var_1)
                    + (var_1 + (mean_1 - mean_2).pow(2))
                    / (2 * var_2)
                    - 0.5
                )

                kl_scores[name] = kl_div.detach().clone()

    if len(kl_scores) == 0:
        raise RuntimeError("No BatchNorm2d or LayerNorm KL scores were collected.")

    logging.info(
        f"Collected KL scores for {len(kl_scores)} normalization layers."
    )

    return kl_scores


def bnp_defense(net, u, kl_scores):
    """
    Apply Adaptive-A BNP pruning with layer-wise u values.

    u:
        Dictionary {layer_name: u_l}

    kl_scores:
        Pre-computed KL scores for each normalization layer.
    """
    params = net.state_dict()
    pruning_stats = {}

    for name, m in net.named_modules():

        if not (
            isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.LayerNorm)
        ):
            continue

        if name not in kl_scores:
            continue

        # Get the individual u value for this layer
        layer_u = u[name] if isinstance(u, dict) else u

        kl_div = kl_scores[name].to(params[name + '.weight'].device)

        # Adaptive-A threshold for this individual layer
        threshold = adaptive_mad_threshold(
            kl_div,
            k=float(layer_u)
        )

        index = kl_div > threshold

        pruned_channels = int(index.sum().item())
        total_channels = int(index.numel())
        prune_ratio = pruned_channels / max(total_channels, 1)

        # Apply the pruning mask
        params[name + '.weight'][index] = 0
        params[name + '.bias'][index] = 0

        pruning_stats[name] = {
            "u": float(layer_u),
            "threshold": float(threshold.item()),
            "pruned_channels": pruned_channels,
            "total_channels": total_channels,
            "prune_ratio": prune_ratio,
        }

        # logging.info(
        #     f"[Adaptive-A] Layer: {name}, "
        #     f"u: {layer_u:.4f}, "
        #     f"Threshold: {threshold.item():.6f}, "
        #     f"Pruned: {pruned_channels}/{total_channels}, "
        #     f"Ratio: {prune_ratio:.2%}"
        # )

    net.load_state_dict(params)

    total_pruned = sum(
        item["pruned_channels"]
        for item in pruning_stats.values()
    )

    total_channels = sum(
        item["total_channels"]
        for item in pruning_stats.values()
    )

    return {
        "layers": pruning_stats,
        "total_pruned": total_pruned,
        "total_channels": total_channels,
        "prune_ratio": total_pruned / max(total_channels, 1),
    }

def get_mbns_network(
	model_name: str,
	num_classes: int = 10,
	**kwargs,
):
	if model_name == 'preactresnet18':
		net = mbns_model.preact_mbns.PreActResNet18(num_classes = num_classes, **kwargs)
	elif model_name == 'vgg19_bn':
		net = mbns_model.vgg_mbns.vgg19_bn(num_classes = num_classes,  **kwargs)
	elif model_name == 'densenet161':
		net = mbns_model.den_mbns.densenet161(num_classes= num_classes, **kwargs)
	elif model_name == 'mobilenet_v3_large':
		net = mbns_model.mobilenet_mbns.mobilenet_v3_large(num_classes= num_classes, **kwargs)
	elif model_name == 'efficientnet_b3':
		net = mbns_model.eff_mbns.efficientnet_b3(num_classes= num_classes, **kwargs)
	elif model_name == 'convnext_tiny':
		try :
			net = mbns_model.conv_mbns.convnext_tiny(num_classes= num_classes, 
			)
		except :
			net = mbns_model.conv_new_mbns.convnext_tiny(num_classes= num_classes, 
			)
	elif model_name == 'vit_b_16':
		try :
			from torchvision.transforms import Resize
			net = mbns_model.vit_mbns.vit_b_16(
					pretrained = True,
				)
			net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
			net = torch.nn.Sequential(
					Resize((224, 224)),
					net,
				)
		except :
			from torchvision.transforms import Resize
			net = mbns_model.vit_new_mbns.vit_b_16(
					pretrained = True,
				)
			net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
			net = torch.nn.Sequential(
					Resize((224, 224)),
					net,
				)
	else:
		raise SystemError('NO valid model match in function generate_cls_model!')

	return net


class bnp(defense):

	def __init__(self,args):
		with open(args.yaml_path, 'r') as f:
			defaults = yaml.safe_load(f)

		defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

		args.__dict__ = defaults

		args.terminal_info = sys.argv

		args.num_classes = get_num_classes(args.dataset)
		args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
		args.img_size = (args.input_height, args.input_width, args.input_channel)
		args.dataset_path = f"{args.dataset_path}/{args.dataset}"
		self.args = args

		if 'result_file' in args.__dict__ :
			if args.result_file is not None:
				self.set_result(args.result_file)

	def add_arguments(parser):
		parser.add_argument('--device', type=str, help='cuda, cpu')
		parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
		parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
		parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
		parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

		parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
		parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
		parser.add_argument('--log', type=str, help='the location of log')
		parser.add_argument("--dataset_path", type=str, help='the location of data')
		parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
		parser.add_argument('--result_file', type=str, help='the location of result')
	
		parser.add_argument('--epochs', type=int)
		parser.add_argument('--batch_size', type=int)
		parser.add_argument("--num_workers", type=float)
		parser.add_argument('--lr', type=float)
		parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
		parser.add_argument('--steplr_stepsize', type=int)
		parser.add_argument('--steplr_gamma', type=float)
		parser.add_argument('--steplr_milestones', type=list)
		parser.add_argument('--model', type=str, help='resnet18')
		
		parser.add_argument('--client_optimizer', type=int)
		parser.add_argument('--sgd_momentum', type=float)
		parser.add_argument('--wd', type=float, help='weight decay of sgd')
		parser.add_argument('--frequency_save', type=int,
						help=' frequency_save, 0 is never')

		parser.add_argument('--random_seed', type=int, help='random seed')
		parser.add_argument('--yaml_path', type=str, default="./config/defense/bnp/config.yaml", help='the path of yaml')

		#set the parameter for the bnp defense
		# Adaptive-A search
		parser.add_argument('--u', type=float, default=2.5,	help='fallback/default u')
		parser.add_argument('--u_min', type=float, default=1.0,	help='minimum u for search')
		parser.add_argument('--u_max', type=float, default=5.0, help='maximum u for search')
		parser.add_argument('--u_num', type=int, default=41, help='number of candidate u values')

		# ACC-constrained layer-wise search
		parser.add_argument('--acc_drop_tolerance',type=float,default=0.02,help='maximum allowed ACC drop from clean validation baseline')
		parser.add_argument('--val_ratio',type=float, default=0.2, help='ratio of clean data used for validation')
		parser.add_argument('--u_search_max_passes', type=int, default=2, help='maximum number of coordinate search passes')
		parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
		parser.add_argument('--index', type=str, help='index of clean data')


	def set_result(self, result_file):
		attack_file = 'record/' + result_file
		save_path = 'record/' + result_file + '/defense/bnp/'
		if not (os.path.exists(save_path)):
			os.makedirs(save_path)
		# assert(os.path.exists(save_path))    
		self.args.save_path = save_path
		if self.args.checkpoint_save is None:
			self.args.checkpoint_save = save_path + 'checkpoint/'
			if not (os.path.exists(self.args.checkpoint_save)):
				os.makedirs(self.args.checkpoint_save) 
		if self.args.log is None:
			self.args.log = save_path + 'log/'
			if not (os.path.exists(self.args.log)):
				os.makedirs(self.args.log)  
		self.result = load_attack_result(attack_file + '/attack_result.pt')

	def set_trainer(self, model):
		self.trainer = PureCleanModelTrainer(
			model,
		)

	def set_logger(self):
		args = self.args
		logFormatter = logging.Formatter(
			fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
			datefmt='%Y-%m-%d:%H:%M:%S',
		)
		logger = logging.getLogger()

		fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
		fileHandler.setFormatter(logFormatter)
		logger.addHandler(fileHandler)

		consoleHandler = logging.StreamHandler()
		consoleHandler.setFormatter(logFormatter)
		logger.addHandler(consoleHandler)

		logger.setLevel(logging.INFO)
		logging.info(pformat(args.__dict__))

		try:
			logging.info(pformat(get_git_info()))
		except:
			logging.info('Getting git info fails.')
	
	def set_devices(self):
		# self.device = torch.device(
		# 	(
		# 		f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
		# 		# since DataParallel only allow .to("cuda")
		# 	) if torch.cuda.is_available() else "cpu"
		# )
		self.device =self.args.device

	def mitigation(self):
		self.set_devices()
		fix_random(self.args.random_seed)

		# Prepare model, optimizer, scheduler
		
		net = get_mbns_network(self.args.model,self.args.num_classes,norm_layer=mbns_model.BatchNorm2d_MBNS)
		net.load_state_dict(self.result['model'])
		if "," in self.device:
			net = torch.nn.DataParallel(
				net,
				device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
			)
			self.args.device = f'cuda:{net.device_ids[0]}'
			net.to(self.args.device)
		else:
			net.to(self.args.device)
		
		criterion = argparser_criterion(self.args)

		trainloader_all = {}

		train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
		train_dataset = self.result['bd_train'].wrapped_dataset
		data_set_without_tran = train_dataset
		data_set_o = self.result['bd_train']
		data_set_o.wrapped_dataset = data_set_without_tran
		data_set_o.wrap_img_transform = train_tran
		data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=self.args.pin_memory)
		trainloader_backdoor = data_loader
		trainloader_all['bd_train'] = trainloader_backdoor

		bd_train = self.result['bd_train']
		bd_wrapped = bd_train.wrapped_dataset

		# ==========================================
		# 1. Obtain the full potentially poisoned dataset
		# ==========================================
		bd_wrapped = self.result['bd_train'].wrapped_dataset
		print("Meta-Sift input size:", len(bd_wrapped))

		# ==========================================
		# 2. Build a dataset adapter for Meta-Sift
		# ==========================================
		meta_transform = transforms.Compose([
			transforms.Resize((32, 32)),
			transforms.ToTensor(),
		])

		meta_dataset = MetaSiftDataset(
			bd_dataset=bd_wrapped,
			transform=meta_transform
		)

		# ==========================================
		# 3. Configure Meta-Sift
		# ==========================================
		class MetaSiftArgs:
			num_classes = self.args.num_classes
			num_sifter = 5
			res_epochs = 1
			warmup_epochs = 1
			batch_size = 128
			num_workers = self.args.num_workers
			v_lr = 0.0005
			meta_lr = 0.1
			top_k = 15
			go_lr = 1e-1
			num_act = 4
			momentum = 0.9
			nesterov = True
			random_seed = self.args.random_seed

		meta_args = MetaSiftArgs()

		# ==========================================
		# 4. Select a small candidate-clean subset
		# ==========================================
		total_pick = 5000  # Change to 1000 if needed

		clean_idx = meta_sift(
			meta_args,
			meta_dataset,
			total_pick=total_pick,
			coarse_ratio=0.5,
			coarse_sifter_ratio=0.4,
		)

		clean_idx = np.asarray(clean_idx, dtype=np.int64)
		print("Meta-Sift selected:", len(clean_idx))

		log_index = self.args.log + 'index.txt'
		np.savetxt(log_index, clean_idx, fmt='%d')

		# ==========================================
		# 5. Create an independent dataset copy
		# ==========================================
		selected_bd_dataset = bd_wrapped.copy()
		selected_bd_dataset.subset(clean_idx)

		# ==========================================
		# 6. Build BNP's candidate-clean DataLoader
		# ==========================================
		train_tran = get_transform(
			self.args.dataset,
			self.args.input_height,
			self.args.input_width,
			train=True
		)

		data_set_o = dataset_wrapper_with_transform(
			selected_bd_dataset,
			wrap_img_transform=train_tran
		)

		data_loader = torch.utils.data.DataLoader(
			data_set_o,
			batch_size=self.args.batch_size,
			num_workers=self.args.num_workers,
			shuffle=True,
			pin_memory=self.args.pin_memory
		)

		trainloader_clean = data_loader
		trainloader_all['clean_train'] = trainloader_clean

		# ==========================================
		# Prepare clean validation and final test sets
		# ==========================================
		test_tran = get_transform(self.args.dataset, self.args.input_height, self.args.input_width,	train=False)

		data_bd_testset = self.result['bd_test']
		data_bd_testset.wrap_img_transform = test_tran
		data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, drop_last=False,	pin_memory=self.args.pin_memory)

		data_clean_all = self.result['clean_test']
		data_clean_all.wrap_img_transform = test_tran

		total_size = len(data_clean_all)
		val_size = int(total_size * self.args.val_ratio)
		test_size = total_size - val_size

		if val_size <= 0 or test_size <= 0:
			raise ValueError("Invalid clean validation/test split.")

		generator = torch.Generator()
		generator.manual_seed(self.args.random_seed)
		clean_val_subset, clean_test_subset = torch.utils.data.random_split(data_clean_all,	[val_size, test_size],	generator=generator)
		clean_val_loader = torch.utils.data.DataLoader(clean_val_subset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers,	pin_memory=self.args.pin_memory)
		data_clean_loader = torch.utils.data.DataLoader(clean_test_subset,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.num_workers,pin_memory=self.args.pin_memory)

		test_dataloader_dict = {"clean_test_dataloader": data_clean_loader, "bd_test_dataloader": data_bd_loader}
		print(f"Clean validation samples: {val_size}")
		print(f"Clean final test samples: {test_size}")

		# ==========================================
		# ACC-Constrained Layer-wise Coordinate Search
		# ==========================================
		default_u = np.linspace(
			self.args.u_min,
			self.args.u_max,
			self.args.u_num
		)
		default_u = np.sort(default_u)

		# 1. Collect KL scores only once
		kl_scores = collect_layer_kl_scores(
			net,
			trainloader_all,
			self.args
		)

		layer_names = list(kl_scores.keys())
		logging.info(f"Normalization layers for layer-wise search: {len(layer_names)}")

		# 2. Evaluate the original unpruned model
		baseline_model = copy.deepcopy(net)

		baseline_acc, baseline_loss = evaluate_clean_acc(
			baseline_model,
			clean_val_loader,
			criterion,
			self.args.device
		)

		acc_min = baseline_acc - self.args.acc_drop_tolerance

		print("\n========== Layer-wise ACC-Constrained Search ==========")
		print(f"Baseline Clean ACC: {baseline_acc:.4%}")
		print(f"Baseline Clean Loss: {baseline_loss:.6f}")
		print(f"Allowed ACC drop: {self.args.acc_drop_tolerance:.4%}")
		print(f"Minimum required ACC: {acc_min:.4%}")

		# 3. Initialize all layers with no pruning
		selected_u_values = {
			name: float('inf')
			for name in layer_names
		}

		selected_model = copy.deepcopy(net)
		selected_stats = {
			"total_pruned": 0,
			"total_channels": 0,
			"prune_ratio": 0.0,
			"layers": {}
		}
		search_results = []

		# 4. Layer-wise coordinate search
		for search_pass in range(self.args.u_search_max_passes):
			logging.info(
				f"\n========== Coordinate Search Pass "
				f"{search_pass + 1}/{self.args.u_search_max_passes} =========="
			)
			pass_improved = False
			for layer_name in layer_names:
				logging.info(f"\n--- Optimizing layer: {layer_name} ---")
				best_layer_u = selected_u_values[layer_name]
				best_layer_model = copy.deepcopy(selected_model)
				best_layer_stats = selected_stats
				best_pruned = selected_stats["total_pruned"]
				best_val_acc = evaluate_clean_acc(
					selected_model,
					clean_val_loader,
					criterion,
					self.args.device
				)[0]

				# Try each candidate u for the current layer.
				for candidate_u in default_u:
					candidate_u_values = selected_u_values.copy()
					candidate_u_values[layer_name] = float(candidate_u)

					candidate_model = copy.deepcopy(net)
					candidate_model.eval()

					pruning_stats = bnp_defense(
						candidate_model,
						candidate_u_values,
						kl_scores
					)

					val_acc, val_loss = evaluate_clean_acc(
						candidate_model,
						clean_val_loader,
						criterion,
						self.args.device
					)
					feasible = val_acc >= acc_min
					total_pruned = pruning_stats["total_pruned"]

					search_results.append({
						"pass": search_pass + 1,
						"layer": layer_name,
						"candidate_u": float(candidate_u),
						"clean_val_acc": val_acc,
						"clean_val_loss": val_loss,
						"total_pruned": total_pruned,
						"total_channels": pruning_stats["total_channels"],
						"prune_ratio": pruning_stats["prune_ratio"],
						"acc_min": acc_min,
						"feasible": feasible
					})

					logging.info(
						f"[Search] Layer: {layer_name}, "
						f"u: {candidate_u:.2f}, "
						f"ACC: {val_acc:.2%}, "
						f"Pruned: {total_pruned}, "
						f"Feasible: {feasible}"
						)

					if feasible and total_pruned > best_pruned:
						best_pruned = total_pruned
						best_layer_u = float(candidate_u)
						best_layer_model = copy.deepcopy(candidate_model)
						best_layer_stats = pruning_stats
						best_val_acc = val_acc

				if best_pruned > selected_stats["total_pruned"]:
					selected_u_values[layer_name] = best_layer_u
					selected_model = best_layer_model
					selected_stats = best_layer_stats
					pass_improved = True
					logging.info(
						f"[Layer Updated] {layer_name} | "
						f"Best u: {best_layer_u:.4f} | "
						f"Total Pruned: {best_pruned} | "
						f"Val ACC: {best_val_acc:.4%}"
						)
					logging.info(
						f"Updated {layer_name}: "
						f"u = {best_layer_u:.4f}, "
						f"Total pruned = {best_pruned}, "
						f"Validation ACC = {best_val_acc:.4%}"
					)
				else:
					print(
						f"No improvement for {layer_name}. "
						f"Keep u = {selected_u_values[layer_name]}"
					)

			if not pass_improved:
				print("\nNo further improvement. Coordinate search stopped.")
				break

		# 5. Final selected solution
		selected_acc, selected_loss = evaluate_clean_acc(
			selected_model,
			clean_val_loader,
			criterion,
			self.args.device
		)

		print("\n========== Layer-wise Search Result ==========")
		print(f"Selected Validation ACC: {selected_acc:.4%}")
		print(f"Minimum Required ACC: {acc_min:.4%}")
		print(f"Total Pruned Channels: {selected_stats['total_pruned']}")
		print(f"Overall Pruning Ratio: {selected_stats['prune_ratio']:.4%}")

		print("\nSelected layer-wise u values:")
		for name, value in selected_u_values.items():
			print(f"{name}: {value}")

		logging.info(
			f"Selected Validation ACC: {selected_acc:.4%}, "
			f"ACC Constraint: {acc_min:.4%}, "
			f"Total Pruned: {selected_stats['total_pruned']}, "
			f"Pruning Ratio: {selected_stats['prune_ratio']:.4%}"
		)

		# 6. Save search results
		import pandas as pd
		import json

		search_df = pd.DataFrame(search_results)
		search_df.to_csv(
			f"{self.args.save_path}layer_wise_search.csv",
			index=False
		)

		# 处理 inf 避免 json 报错
		saved_u_values = {
			name: (value if np.isfinite(value) else "inf")
			for name, value in selected_u_values.items()
		}

		with open(
			f"{self.args.save_path}selected_layer_u.json",
			"w"
		) as f:
			json.dump(saved_u_values, f, indent=4)

		print(f"Search results saved to: {self.args.save_path}layer_wise_search.csv")
		print(f"Selected u values saved to: {self.args.save_path}selected_layer_u.json")

		# ==========================================
		# Final evaluation using selected model
		# ==========================================
		selected_model.eval()

		model = generate_cls_model(
			self.args.model,
			self.args.num_classes
		)
		model.load_state_dict(selected_model.state_dict())
		model.to(self.args.device)

		self.set_trainer(model)
		self.trainer.set_with_dataloader(
			train_dataloader=trainloader_backdoor,
			test_dataloader_dict=test_dataloader_dict,
			criterion=criterion,
			optimizer=None,
			scheduler=None,
			device=self.args.device,
			amp=self.args.amp,
			frequency_save=self.args.frequency_save,
			save_folder_path=self.args.save_path,
			save_prefix='bnp',
			prefetch=self.args.prefetch,
			prefetch_transform_attr_name="ori_image_transform_in_loading",
			non_blocking=self.args.non_blocking,
		)

		# Final evaluation
		clean_test_loss_avg_over_batch, \
		bd_test_loss_avg_over_batch, \
		test_acc, \
		test_asr, \
		test_ra = self.trainer.test_current_model(
			test_dataloader_dict,
			self.args.device
		)

		logging.info(
			f"\n========== Final Selected Layer-wise Model ==========\n"
			f"Selected u values: {saved_u_values}\n"
			f"Clean ACC: {test_acc:.4%} | "
			f"ASR: {test_asr:.4%} | "
			f"RA: {test_ra:.4%} | "
			f"Clean Loss: {clean_test_loss_avg_over_batch:.6f} | "
			f"BD Loss: {bd_test_loss_avg_over_batch:.6f}"
		)	


		# Save selected model
		result = {}
		result['model'] = model

		save_defense_result(
			model_name=self.args.model,
			num_classes=self.args.num_classes,
			model=model.cpu().state_dict(),
			save_path=self.args.save_path,
		)

		return result

	def defense(self,result_file):
		self.set_result(result_file)
		self.set_logger()
		result = self.mitigation()
		return result
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=sys.argv[0])
	bnp.add_arguments(parser)
	args = parser.parse_args()
	method = bnp(args)
	if "result_file" not in args.__dict__:
		args.result_file = 'defense_test_badnet'
	elif args.result_file is None:
		args.result_file = 'defense_test_badnet'
	result = method.defense(args.result_file)