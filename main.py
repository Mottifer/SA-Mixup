import torch
import copy
import argparse
import inspect
import random
from tqdm import tqdm
import numpy as np

from utils.conf import read_config
from utils.util import load_data, load_model, train_model_pyg_dynamic_data, \
    generate_inclass_mixup_with_high_confidence_edge, generate_pseduo_label, generate_pseduo_label_same, \
    reduce_train_mask_per_class

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

PAPER_DATASET_PARAMS = {
    'Cora': {'confidence_threshold': 0.85, 'alpha': 2.0, 'beta': 1.0},
    'CiteSeer': {'confidence_threshold': 0.85},
    'Pubmed': {'confidence_threshold': 0.80, 'alpha': 1.5, 'beta': 1.0},
}


def supports_keyword(function, keyword):
    parameters = inspect.signature(function).parameters
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


# 创建一个命令行参数解析器，可以从命令行读取参数配置。适合在训练不同实验时灵活设置参数。
parser = argparse.ArgumentParser(description="model training")

parser.add_argument('--dataset_name', type=str, default='Cora')
parser.add_argument('--model_name', type=str, default='gcn')
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--step', type=int, default=30)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_mixup', type=float, default=0.02)
parser.add_argument('--hid_num', type=int, default=128)
parser.add_argument('--alpha', type=float, default=2.0)
parser.add_argument('--beta', type=float, default=1.5)
parser.add_argument('--candidate_filter_passes', type=int, default=5)
parser.add_argument('--mc_dropout_passes', type=int, default=10)
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--omega', type=float, choices=(0.0, 1.0), default=1.0)
parser.add_argument('--nb_heads', type=int, default=4)
parser.add_argument('--nb_heads2', type=int, default=1)
parser.add_argument('--dropout_rate1_before', type=float, default=0.9)
parser.add_argument('--dropout_rate2_before', type=float, default=0.005)
parser.add_argument('--dropout_rate1_high', type=float, default=0.5)
parser.add_argument('--dropout_rate1_mixup', type=float, default=0.9)
parser.add_argument('--dropout_rate2_mixup', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--weight_decay_mixup', type=float, default=0.0001)
parser.add_argument('--with_test_data', choices=('yes', 'no'), default='yes')
parser.add_argument('--score_threshold_mixup', type=float, default=0.85)
parser.add_argument('--score_threshold_high', type=float, default=0.85)
parser.add_argument('--patient', type=int, default=40)
parser.add_argument('--dataset_root_path', type=str, default='./data')
parser.add_argument('--train_model_root_path', type=str, default='./store')
parser.add_argument('--config_root_path', type=str, default='./config')

# 读取命令行参数
args = parser.parse_args()

# 从config文件补充或替代上面的参数，目前就在这个里面读取的参数
args = read_config(args)

# Enforce the experiment-wide settings reported in Section 4.1.3 after the
# dataset/backbone-specific configuration has been loaded.
args.step = 30
args.epoch = 500
args.patient = 40
args.candidate_filter_passes = 5
args.mc_dropout_passes = 10

# Apply the dataset-specific values reported by the confidence-threshold and
# alpha/beta sensitivity experiments. Other datasets retain their config values.
paper_params = PAPER_DATASET_PARAMS.get(args.dataset_name, {})
if 'confidence_threshold' in paper_params:
    args.score_threshold_mixup = paper_params['confidence_threshold']
    args.score_threshold_high = paper_params['confidence_threshold']
if 'alpha' in paper_params:
    args.alpha = paper_params['alpha']
if 'beta' in paper_params:
    args.beta = paper_params['beta']

device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')

# 数据加载
ori_x, ori_y, ori_edge_index, train_index, val_index, test_index, dataset = load_data(args.dataset_name,
                                                                                      args.dataset_root_path, device)
# 定义mask
ori_train_mask, ori_val_mask, ori_test_mask = torch.zeros(ori_x.shape[0], dtype=torch.bool).to(device), torch.zeros(
    ori_x.shape[0], dtype=torch.bool).to(device), torch.zeros(ori_x.shape[0], dtype=torch.bool).to(device)
ori_train_mask[train_index], ori_val_mask[val_index], ori_test_mask[test_index] = True, True, True
# train_mask 只在训练集中为 True，其他为 False


test_accs = []
best_model_save_path = "{}/{}_{}.pt".format(args.train_model_root_path, args.model_name, args.dataset_name)

# 如果with_test_data=no，则测试集的数据不会用于训练
if args.with_test_data == "no":
    unlabel_mask = ~(ori_train_mask + ori_val_mask + ori_test_mask)
elif args.with_test_data == "yes":
    unlabel_mask = ~(ori_train_mask + ori_val_mask)

# 训练循环
for i in tqdm(range(args.step)):
    seed = args.seed + i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x, y, train_y, train_mask, val_mask, test_mask, edge_index = copy.deepcopy(ori_x).to(device), copy.deepcopy(
        ori_y).to(device), \
        copy.deepcopy(ori_y).to(device), copy.deepcopy(ori_train_mask), copy.deepcopy(ori_val_mask), copy.deepcopy(
        ori_test_mask), copy.deepcopy(ori_edge_index)
    unlabel_index = torch.nonzero(unlabel_mask, as_tuple=True)[0]


    # 降低训练集
    train_mask = reduce_train_mask_per_class(
        y=y,
        train_mask=train_mask,
        num_classes=dataset.num_classes,
        k_per_class=20
    )
    # 14labels → 每类2个
    # 21labels → 每类3个
    # 28labels → 每类4个
    # 140labels → 每类20个

    train_index = torch.nonzero(train_mask, as_tuple=True)[0]


    # 用于初步训练并生成伪标签
    pretrain_model = load_model(args, dataset.num_node_features, dataset.num_classes, args.hid_num,
                                args.dropout_rate1_before, args.dropout_rate2_before).to(device)
    # 用于 Mixup 之后的训练
    model_intra_mixup = load_model(args, dataset.num_node_features, dataset.num_classes, args.hid_num,
                                   args.dropout_rate1_mixup, args.dropout_rate2_mixup).to(device)

    # 预训练模型
    optimizer_pretrain = torch.optim.Adam(pretrain_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    optimizer_intra_mixup = torch.optim.Adam(model_intra_mixup.parameters(), lr=args.lr_mixup,
                                             weight_decay=args.weight_decay_mixup)

    # 训练GNN，保存模型权重
    train_model_pyg_dynamic_data(pretrain_model, optimizer_pretrain, x, edge_index, y, train_mask, val_mask, test_mask,
                                 args.epoch, best_model_save_path, patient=args.patient)

    # 加载模型权重，获取稳定的预测
    pretrain_model.load_state_dict(torch.load(best_model_save_path))

    # 生成伪标签
    # 从无标签节点中挑选预测置信度高的样本，生成伪标签，并加入训练集
    train_mask, train_y, label_index_ori = generate_pseduo_label(pretrain_model, x, edge_index, train_mask, train_y,
                                                                 unlabel_index,
                                                                 score_threshold=args.score_threshold_mixup)
    # 初始化一个高置信度伪标签处理的版本
    train_mask_high, train_y_high, label_index_high = copy.deepcopy(ori_train_mask), copy.deepcopy(y), \
        torch.nonzero(unlabel_mask, as_tuple=True)[0]
    # 生成高置信度标签
    consistency_kwargs = {
        'model_dropout_rate': args.dropout_rate1_high,
        'score_threshold': args.score_threshold_high,
    }
    if supports_keyword(generate_pseduo_label_same, 'n'):
        consistency_kwargs['n'] = args.candidate_filter_passes
    elif supports_keyword(generate_pseduo_label_same, 'num_forward_passes'):
        consistency_kwargs['num_forward_passes'] = args.candidate_filter_passes

    train_mask_high, train_y_high, label_index_high = generate_pseduo_label_same(
        pretrain_model, x, edge_index, train_mask_high, train_y_high,
        label_index_high, **consistency_kwargs
    )
    # 进行mixup数据增强，对高置信度的伪标签数据进行 Mixup
    # 是含有噪声的训练标签，可能用于模拟标签不确定性
    # 是干净的标签，用于监督和对比
    x, noise_y, clean_y, edge_index, train_mask, val_mask, test_mask = \
        generate_inclass_mixup_with_high_confidence_edge(
            x, edge_index, train_y, train_y_high, y,
            label_index_ori, label_index_high,
            train_mask, val_mask, test_mask,
            dataset.num_classes, device, pretrain_model,
            args.alpha,  # 语义关系系数 alpha
            args.beta,  # 不确定性系数 beta
            args.mc_dropout_passes,  # MC Dropout 次数 T=10
            args.omega,  # 1 表示余弦相似度
            args.K  # 每个生成节点的邻居数量
        )

    noise_y[label_index_high] = train_y_high[label_index_high]  # 使用高置信度伪标签
    noise_y[train_index] = y[train_index]  # 保留原始训练节点的真实标签
    train_mask[label_index_high] = True  # 把高置信度样本纳入训练集中
    train_mask[train_index] = True  # 确保原始训练集的节点仍然保留在训练集中

    # 训练最终模型
    best_test_acc = train_model_pyg_dynamic_data(model_intra_mixup, optimizer_intra_mixup, x, edge_index, noise_y,
                                                 train_mask, val_mask, test_mask, args.epoch, best_model_save_path,
                                                 test_y=clean_y)

    print(f"Train nodes: {train_mask.sum().item()}")
    print(f"Val nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")
    print(f"Total nodes: {x.size(0)}")

    # 记录实验结果
    test_accs.append(best_test_acc.item())


print([f"{acc:.2f}" for acc in test_accs])
print(f'Average test accuracy after SA-Mixup: {np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}')

print('-' * 50)
