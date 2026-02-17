from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F
# from data import *
from model import *
from util import *

import data as data_mod
from data import Tinto
# [ADD]
from torch.optim.swa_utils import AveragedModel

use_ema = True
ema_decay = 0.999  # 可试 0.999~0.9999

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# from PCM_models.openpoints.models.PCM.PCM_frequency_channel import PointMambaEncoder, PointMambaDecoder
from PCM_models.openpoints.models.segmentation.base_seg import BaseSeg, SegHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_without_xyz = 85  # 88 - 3（不含 xyz 的特征通道数）

# encoder_args = dict(
#     NAME="PointMambaEncoderDual",
#     fuse="gate",  # sum 融合：解码器通道无需改
#
#     shared=dict(
#         in_channels=85, embed_dim=96, groups=1, res_expansion=1.0,
#         activation="relu", bias=False, use_xyz=True, normalize="anchor",
#         dim_expansion=[1, 2, 2, 2], pre_blocks=[1, 1, 1, 1],
#         mamba_blocks=[1, 2, 2, 4], pos_blocks=[0, 0, 0, 0],
#         k_neighbors=[12, 12, 12, 12], reducers=[4, 4, 2, 2],
#         rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
#         bimamba_type="v2", drop_path_rate=0.1, mamba_pos=True,
#         use_order_prompt=True, prompt_num_per_order=6,
#         use_windows=True, windows_size=[1024, 512, 256, 128],
#         grid_size=0.04, combine_pos=False
#     ),
#
#     # 空间通道：保留 6×坐标置换 + H/Z/Zt（9 条）
#     spatial=dict(
#         mamba_layers_orders=[
#             "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "hilbert", "z", "z-trans"
#         ]
#     ),
#
#     # 频域通道：仅 {hilbert, z, z-trans}，按 1/2/2/4 展开
#     # 频域通道：容量减半，作为辅助分支
#     freq=dict(
#         # ① 频域 encoder Mamba 深度显著减小：总共 0+1+1+2 = 4 层
#         mamba_blocks=[1, 1, 2, 2],
#
#         # ② 对应 4 个 layer 的顺序（注意长度必须等于 sum(mamba_blocks)）
#         #   Stage1: 无 Mamba
#         #   Stage2: 1 层，用 hilbert
#         #   Stage3: 1 层，用 z
#         #   Stage4: 2 层，用 hilbert + z-trans
#         mamba_layers_orders=[
#             "rtp-snake",  # stage1
#             "rpt-snake",  # stage2
#             "trp-snake",  # stage3-1
#             "tpr-snake",  # stage3-2
#             "prt-snake",  # stage4-1
#             "ptr-snake",  # stage4-2
#         ],
#
#         # ③ 更小的 drop_path_rate，减弱频域分支的抖动和表达能力
#         # drop_path_rate=0.05,
#
#         # ④ 可以先关闭 window，让频域分支更像全局/粗粒度先验
#         # use_windows=False,
#
#         # ⑤ grid_size 稍微变大一点，频域序列划分更粗糙（可选）
#         # grid_size=0.06,
#     ),
#
# )

# # ===================== [REPLACE] encoder_args: Dual -> Single PCM Encoder =====================
# encoder_args = dict(
#     NAME="PointMambaEncoder",          # 对应 PCM.py 里的 PointMambaEncoder
#     in_channels=85,                    # 你的 x 输入通道数 (C_without_xyz)
#     embed_dim=96,
#     groups=1,
#     res_expansion=1.0,
#     activation="relu",
#     bias=False,
#     use_xyz=True,
#     normalize="anchor",
#
#     dim_expansion=[1, 2, 2, 2],
#     pre_blocks=[1, 1, 1, 1],
#     pos_blocks=[0, 0, 0, 0],
#
#     k_neighbors=[12, 12, 12, 12],
#     k_strides=[1, 1, 1, 1],
#     reducers=[4, 4, 2, 2],
#
#     # Encoder 各 stage 的 Mamba 层数：sum=9
#     mamba_blocks=[1, 2, 2, 4],
#
#     # 必须长度=9（与 sum(mamba_blocks) 一致）
#     mamba_layers_orders=[
#         "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "hilbert", "z", "z-trans"
#     ],
#
#     # 其它与 PCM.py 参数一致/兼容
#     rms_norm=True,
#     residual_in_fp32=True,
#     fused_add_norm=True,
#     bimamba_type="v2",
#     drop_path_rate=0.1,
#
#     mamba_pos=True,
#     pos_type="share",
#     pos_proj_type="linear",
#     grid_size=0.04,
#     combine_pos=False,
#
#     use_order_prompt=True,
#     prompt_num_per_order=6,
#
#     use_windows=True,
#     windows_size=[1024, 512, 256, 128],
#
#     block_residual=True,
#     cls_pooling="max",
# )
# # ============================================================================================

encoder_args = dict(
    NAME="PointMambaEncoder",
    in_channels=85, embed_dim=96, groups=1, res_expansion=1.0,
    activation="relu", bias=False, use_xyz=True, normalize="anchor",
    dim_expansion=[1, 2, 2, 2], pre_blocks=[1, 1, 1, 1],
    # [MOD] 4-stage blocks: 1+1+2+2 = 6 layers
    mamba_blocks=[1, 1, 2, 2], pos_blocks=[0, 0, 0, 0],
    k_neighbors=[12, 12, 12, 12], reducers=[4, 4, 2, 2],
    rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
    bimamba_type="v2", drop_path_rate=0.1, mamba_pos=True,
    # [MOD] 按你的要求改 6-order 顺序
    mamba_layers_orders=["xyz", "zyx", "yxz", "yzx", "zxy", "xzy"],
    # [MOD] len(mamba_layers_orders) must == sum(mamba_blocks) == 6
    # mamba_layers_orders=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
    # mamba_layers_orders=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "hilbert", "z", "z-trans"],
    use_order_prompt=True, prompt_num_per_order=6,
    use_windows=True, windows_size=[1024, 512, 256, 128],
    grid_size=0.04, combine_pos=False
)

decoder_args = dict(
    NAME="PointMambaDecoder",
    encoder_channel_list=[96, 96, 192, 384, 768],
    decoder_channel_list=[384, 192, 96, 96],
    decoder_blocks=[1, 1, 1, 1],
    mamba_blocks=[0, 0, 0, 0],
    mamba_layers_orders=[]
)

cls_args = dict(
    NAME="SegHead",
    num_classes=11, global_feat='max,avg',
    in_channels=None, norm_args={'norm': 'bn'}
)

model = BaseSeg(
    encoder_args=encoder_args,
    decoder_args=decoder_args,
    cls_args=cls_args,
    # 其他参数按你原来传
).to(device)
print(str(model))

# Define Train function

# ----*********--*******------****-------***---*****----***
# -------***-----***--***----***-***-----***---***-***--***
# -------***-----******-----***---***----***---***--***-***
# -------***-----***--***--***-----****--***---***---******

# 运行时注入路径（不要写带 'data/' 的前缀）
data_mod.train_path = 'lithonet_sem_seg_hdf5_data_Experiment'
data_mod.test_path = 'lithonet_sem_seg_hdf5_data_Experiment'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 超参数（与当前数据一致）======
args_k = 20  # EdgeConv K
args_dropout = 0.5  # dropout 概率
args_num_class = 11  # 你检查得到的类别数 0..10 共 11 类
dim_rgb = 3  # RGB 通道
dim_vnir = 51  # 你的 vnir 维度（按你之前代码与数据）
dim_geo = 28  # 你的几何特征维度
data_dimension = 88  # XYZ(3)+normXYZ(3)+RGB(3)+VNIR(51)+Geo(28)

# 放在 import 后或 main 前都可
os.makedirs('Mamba_model', exist_ok=True)

# ===================== [ADD] Semantic similarity (prototype cosine) logging =====================
# 使用 order_prompt embedding 的“语义近似”，按论文图示仅采用主相似度：prototype cosine（不加增强项）
# ORDER_VOCAB = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "hilbert", "z", "z-trans"]
# ORDER_VOCAB = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
# [MOD] 为了 similarity_log 的顺序与 mamba_layers_orders 一致
ORDER_VOCAB = ["xyz", "zyx", "yxz", "yzx", "zxy", "xzy"]

# SIM_EVERY_EPOCHS = 10
SIMILARITY_LOG_PATH = os.path.join("Mamba_model", "similarity_log.txt")


# [REPLACE] Fixed interval -> staged schedule (front-dense, later-sparse)
# Epoch 1-20: every 2 epochs
# Epoch 21-60: every 5 epochs
# Epoch 61+  : every 10 epochs
def _sim_interval(epoch1: int) -> int:
    if epoch1 <= 20:
        return 2
    elif epoch1 <= 60:
        return 5
    else:
        return 10


# def should_compute_similarity(epoch1: int) -> bool:
#     return (epoch1 % _sim_interval(epoch1)) == 0

# [REPLACE] Similarity schedule -> compute EVERY epoch (as requested)
def should_compute_similarity(epoch1: int) -> bool:
    # epoch1 is 1-based
    return True


def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def _find_order_prompt_encoder(model_like):
    """Find the encoder module that holds order_prompt/per_layer_prompt_indexe/mamba_layers_orders."""
    m = _unwrap_model(model_like)
    # common: model.encoder exists
    if hasattr(m, "encoder"):
        enc = getattr(m, "encoder")
        if hasattr(enc, "use_order_prompt") and getattr(enc, "use_order_prompt", False) \
                and hasattr(enc, "order_prompt") and hasattr(enc, "per_layer_prompt_indexe") \
                and hasattr(enc, "mamba_layers_orders") and hasattr(enc, "order_prompt_proj") \
                and hasattr(enc, "mamba_blocks_list"):
            return enc
        # some wrappers: encoder might be nested
        for mod in enc.modules():
            if hasattr(mod, "use_order_prompt") and getattr(mod, "use_order_prompt", False) \
                    and hasattr(mod, "order_prompt") and hasattr(mod, "per_layer_prompt_indexe") \
                    and hasattr(mod, "mamba_layers_orders") and hasattr(mod, "order_prompt_proj") \
                    and hasattr(mod, "mamba_blocks_list"):
                return mod

    # fallback: scan all modules
    for mod in m.modules():
        if hasattr(mod, "use_order_prompt") and getattr(mod, "use_order_prompt", False) \
                and hasattr(mod, "order_prompt") and hasattr(mod, "per_layer_prompt_indexe") \
                and hasattr(mod, "mamba_layers_orders") and hasattr(mod, "order_prompt_proj") \
                and hasattr(mod, "mamba_blocks_list"):
            return mod

    return None


@torch.no_grad()
def compute_order_similarity_rankings(model_like, order_vocab=ORDER_VOCAB):
    """按图中“投影后 prompt -> 回到 384 -> L2 归一化 -> cosine 排序”的策略计算语义相似度（不做 Stage 去偏 / Debias）。

    关键点（对应你当前“跳过 Step3，直接到 Step4”的要求）：
    1) 不再直接在原始 384 prompt 空间里做 cosine，而是先走每个 stage 的 order_prompt_proj（与模型前向一致）；
    2) 通过 W_s^T 把投影后的原型回到 384（g_{o,s} = W_s^T W_s ē_o），统一可比空间；
    3) 不做 (g_{o,s} - μ_s)；直接对 g_{o,s} 做 L2 归一化，若同一 order 在多个 stage/层出现则做均值融合后再归一化。

    返回：
        rankings: dict[order] -> list[(other_order, cosine)]
        sim: (O,O) numpy 相似度矩阵（已 L2 归一化后点积）
    """

    enc = _find_order_prompt_encoder(model_like)
    if enc is None:
        raise RuntimeError("Cannot find encoder module with order_prompt for similarity logging.")

    # 必要组件：原始 prompt、per-layer prompt 索引、per-stage 的投影矩阵
    if not hasattr(enc, "order_prompt") or not hasattr(enc, "per_layer_prompt_indexe"):
        raise RuntimeError("Encoder missing order_prompt/per_layer_prompt_indexe.")
    if not hasattr(enc, "order_prompt_proj"):
        raise RuntimeError("Encoder missing order_prompt_proj (per-stage prompt projection).")
    if not hasattr(enc, "mamba_blocks_list") or not hasattr(enc, "mamba_layers_orders"):
        raise RuntimeError("Encoder missing mamba_blocks_list/mamba_layers_orders for stage mapping.")

    weights = enc.order_prompt.weight  # (overall_prompt_nums, 384)

    # -------- Step 0: layer -> stage 映射（通过 mamba_blocks_list 计数推出来）--------
    layer2stage = []
    for s in range(len(enc.mamba_blocks_list)):
        n_layers_s = len(enc.mamba_blocks_list[s])
        layer2stage += [s] * n_layers_s
    if len(layer2stage) != len(enc.mamba_layers_orders):
        raise RuntimeError(
            f"layer2stage length {len(layer2stage)} != len(mamba_layers_orders) {len(enc.mamba_layers_orders)}. "
            "Stage/layer config mismatch."
        )

    # -------- Step 1-2: 对每个 (stage, order) 计算“回投影到 384”的有效原型 g_{o,s} --------
    # g_{o,s} = W_s^T * ( 1/K * sum_k (W_s e_{o,k}) ) = W_s^T W_s ē_o
    stage_order_to_g = {}  # (s, order) -> list[g_vec]  (可能同一 order 在同一 stage 出现多次 layer)
    for li, order in enumerate(enc.mamba_layers_orders):
        if order not in order_vocab:
            continue
        s = layer2stage[li]

        # 该 layer 对应的 K 个 prompt token（384）
        start, end = enc.per_layer_prompt_indexe[li]
        e_bar = weights[start:end].mean(dim=0)  # (384,)

        # 投影到该 stage 的通道空间（与 forward 一致：order_prompt_proj[s](...)）
        v = enc.order_prompt_proj[s](e_bar.unsqueeze(0)).squeeze(0)  # (C_s,)

        # 回到 384：g = W_s^T v
        W = enc.order_prompt_proj[s].weight  # (C_s, 384)
        g = torch.matmul(W.t(), v)  # (384,)

        stage_order_to_g.setdefault((s, order), []).append(g)

    # 对同一 (stage, order) 多次出现做均值（更鲁棒）
    stage_order_g = {}  # (s, order) -> (384,)
    for (s, order), g_list in stage_order_to_g.items():
        stage_order_g[(s, order)] = torch.stack(g_list, dim=0).mean(dim=0)

    # -------- Step 3: 不做 Stage 去偏（NO Debias），仅做 L2 归一化 +（可选）跨 stage 融合 --------
    # 方案 B（贴合 forward）：仍然使用 “投影到 stage -> 回到 384” 得到 g_{o,s}，
    # 但不做 (g_{o,s} - μ_s) 的去均值；直接将 g_{o,s} 作为最终向量来源。
    #
    # 先对每个 (stage, order) 做 L2 normalize：ĝ_{o,s} = g_{o,s} / ||g_{o,s}||_2
    stage_order_g_hat = {}  # (s, order) -> (384,)
    for (s, order), g in stage_order_g.items():
        g_hat = g / (g.norm(p=2) + 1e-12)
        stage_order_g_hat[(s, order)] = g_hat

    # 若同一 order 出现在多个 stage（或同一 stage 多层），做融合：
    # p_o = norm( mean_{(s,·) in S_o} ĝ_{o,s} )
    order_to_vecs = {o: [] for o in order_vocab}
    for (s, o), gh in stage_order_g_hat.items():
        if o in order_to_vecs:
            order_to_vecs[o].append(gh)

    final_proto = []
    for o in order_vocab:
        vecs = order_to_vecs.get(o, [])
        if len(vecs) == 0:
            raise RuntimeError(f"Order '{o}' not found in encoder.mamba_layers_orders (or prompts).")
        v = torch.stack(vecs, dim=0).mean(dim=0)
        v = v / (v.norm(p=2) + 1e-12)
        final_proto.append(v)

    protos = torch.stack(final_proto, dim=0)  # (O, 384)
    sim = (protos @ protos.t()).float().cpu().numpy()  # cosine(sim) since L2-normalized

    # -------- Step 4: 每个 order 对其它 order 做降序排序 --------
    rankings = {}
    for i, oi in enumerate(order_vocab):
        pairs = []
        for j, oj in enumerate(order_vocab):
            if i == j:
                continue
            pairs.append((oj, float(sim[i, j])))
        pairs.sort(key=lambda x: x[1], reverse=True)
        rankings[oi] = pairs

    return rankings, sim


# def compute_order_similarity_rankings(model_like, order_vocab=ORDER_VOCAB):
#     """按图中“投影后 prompt -> 回到 384 -> 不做 Debias，只做 L2 normalize + cosine 排序 -> cosine 排序”的策略计算语义相似度。
#
#     关键点（对应你截图的三个问题修复）：
#     1) 不再直接在原始 384 prompt 空间里做 cosine，而是先走每个 stage 的 order_prompt_proj（与模型前向一致）；
#     2) 通过 W_s^T 把投影后的原型回到 384（g_o = W_s^T W_s ē_o），统一可比空间；
#     3) 在每个 stage 内做去偏（减去 stage 均值 μ_s），尽量剥离 “stage/layer 训练目标差异”，保留更纯的 “order 差异”。
#
#     返回：
#         rankings: dict[order] -> list[(other_order, cosine)]
#         sim: (O,O) numpy 相似度矩阵（已 L2 归一化后点积）
#     """
#     enc = _find_order_prompt_encoder(model_like)
#     if enc is None:
#         raise RuntimeError("Cannot find encoder module with order_prompt for similarity logging.")
#
#     # 必要组件：原始 prompt、per-layer prompt 索引、per-stage 的投影矩阵
#     if not hasattr(enc, "order_prompt") or not hasattr(enc, "per_layer_prompt_indexe"):
#         raise RuntimeError("Encoder missing order_prompt/per_layer_prompt_indexe.")
#     if not hasattr(enc, "order_prompt_proj"):
#         raise RuntimeError("Encoder missing order_prompt_proj (per-stage prompt projection).")
#     if not hasattr(enc, "mamba_blocks_list") or not hasattr(enc, "mamba_layers_orders"):
#         raise RuntimeError("Encoder missing mamba_blocks_list/mamba_layers_orders for stage mapping.")
#
#     weights = enc.order_prompt.weight  # (overall_prompt_nums, 384)
#
#     # -------- Step 0: layer -> stage 映射（通过 mamba_blocks_list 计数推出来）--------
#     layer2stage = []
#     for s in range(len(enc.mamba_blocks_list)):
#         n_layers_s = len(enc.mamba_blocks_list[s])
#         layer2stage += [s] * n_layers_s
#     if len(layer2stage) != len(enc.mamba_layers_orders):
#         raise RuntimeError(
#             f"layer2stage length {len(layer2stage)} != len(mamba_layers_orders) {len(enc.mamba_layers_orders)}. "
#             "Stage/layer config mismatch."
#         )
#
#     # -------- Step 1-2: 对每个 (stage, order) 计算“回投影到 384”的有效原型 g_{o,s} --------
#     # g_{o,s} = W_s^T * ( 1/K * sum_k (W_s e_{o,k}) ) = W_s^T W_s ē_o
#     stage_order_to_g = {}  # (s, order) -> list[g_vec]  (可能同一 order 在同一 stage 出现多次 layer)
#     for li, order in enumerate(enc.mamba_layers_orders):
#         if order not in order_vocab:
#             continue
#         s = layer2stage[li]
#
#         # 该 layer 对应的 K 个 prompt token（384）
#         start, end = enc.per_layer_prompt_indexe[li]
#         e_bar = weights[start:end].mean(dim=0)  # (384,)
#
#         # 投影到该 stage 的通道空间（与 forward 一致：order_prompt_proj[s](...)）
#         v = enc.order_prompt_proj[s](e_bar.unsqueeze(0)).squeeze(0)  # (C_s,)
#
#         # 回到 384：g = W_s^T v
#         W = enc.order_prompt_proj[s].weight  # (C_s, 384)
#         g = torch.matmul(W.t(), v)  # (384,)
#
#         stage_order_to_g.setdefault((s, order), []).append(g)
#
#     # 对同一 (stage, order) 多次出现做均值（更鲁棒）
#     stage_order_g = {}  # (s, order) -> (384,)
#     for (s, order), g_list in stage_order_to_g.items():
#         stage_order_g[(s, order)] = torch.stack(g_list, dim=0).mean(dim=0)
#
#     # # -------- Step 3: 不做 Debias，只做 L2 normalize + cosine 排序（减 μ_s），并归一化 --------
#     # # μ_s = mean_{o in O_s} g_{o,s}
#     # stage_mu = {}      # s -> (384,)
#     # stage_orders = {}  # s -> [order...]
#     # for (s, order) in stage_order_g.keys():
#     #     stage_orders.setdefault(s, set()).add(order)
#     #
#     # for s, oset in stage_orders.items():
#     #     gs = [stage_order_g[(s, o)] for o in sorted(list(oset))]
#     #     mu = torch.stack(gs, dim=0).mean(dim=0)
#     #     stage_mu[s] = mu
#     #     stage_orders[s] = sorted(list(oset))
#     #
#     # # g~_{o,s} = normalize(g_{o,s} - μ_s)
#     # stage_order_g_tilde = {}  # (s, order) -> (384,)
#     # for (s, order), g in stage_order_g.items():
#     #     g_center = g - stage_mu[s]
#     #     g_center = g_center / (g_center.norm(p=2) + 1e-12)
#     #     stage_order_g_tilde[(s, order)] = g_center
#
#     # -------- Step 3.5: 若同一 order 出现在多个 stage，做 stage 融合（均值后再归一化）--------
#
#     # -------- Step 3: 不做 Debias，只做 L2 normalize + cosine 排序（减 μ_s），并归一化 --------
#     # μ_s = mean_{o in O_s} g_{o,s}
#     stage_mu = {}  # s -> (384,)
#     stage_orders = {}  # s -> [order...]
#     stage_debias = {}  # [ADD] s -> bool，若该 stage 内只有 1 个 order，则跳过去偏
#
#     for (s, order) in stage_order_g.keys():
#         stage_orders.setdefault(s, set()).add(order)
#
#     for s, oset in stage_orders.items():
#         orders_s = sorted(list(oset))
#         gs = [stage_order_g[(s, o)] for o in orders_s]
#
#         # [ADD] singleton stage: skip debias, avoid g - mu = 0
#         if len(gs) <= 1:
#             stage_mu[s] = torch.zeros_like(gs[0])  # 占位，不会被用来做减法
#             stage_debias[s] = False
#         else:
#             stage_mu[s] = torch.stack(gs, dim=0).mean(dim=0)
#             stage_debias[s] = True
#
#         stage_orders[s] = orders_s
#
#     # g~_{o,s} = normalize(g_{o,s} - μ_s)  (若 |O_s|==1，则 g~_{o,s} = normalize(g_{o,s}))
#     stage_order_g_tilde = {}  # (s, order) -> (384,)
#     for (s, order), g in stage_order_g.items():
#         # [MOD] 若该 stage 只有 1 个 order，则不减 μ_s
#         if stage_debias.get(s, True):
#             g_center = g - stage_mu[s]
#         else:
#             g_center = g
#
#         g_center = g_center / (g_center.norm(p=2) + 1e-12)
#         stage_order_g_tilde[(s, order)] = g_center
#
#     order_to_vecs = {o: [] for o in order_vocab}
#     for (s, o), gt in stage_order_g_tilde.items():
#         if o in order_to_vecs:
#             order_to_vecs[o].append(gt)
#
#     final_proto = []
#     for o in order_vocab:
#         vecs = order_to_vecs.get(o, [])
#         if len(vecs) == 0:
#             raise RuntimeError(f"Order '{o}' not found in encoder.mamba_layers_orders (or prompts).")
#         v = torch.stack(vecs, dim=0).mean(dim=0)
#         v = v / (v.norm(p=2) + 1e-12)
#         final_proto.append(v)
#
#     protos = torch.stack(final_proto, dim=0)  # (O, 384)
#     sim = (protos @ protos.t()).float().cpu().numpy()  # cosine(sim) since L2-normalized
#
#     # -------- Step 4: 每个 order 对其它 order 做降序排序 --------
#     rankings = {}
#     for i, oi in enumerate(order_vocab):
#         pairs = []
#         for j, oj in enumerate(order_vocab):
#             if i == j:
#                 continue
#             pairs.append((oj, float(sim[i, j])))
#         pairs.sort(key=lambda x: x[1], reverse=True)
#         rankings[oi] = pairs
#
#     return rankings, sim


# def _find_order_prompt_encoder(model_like):
#     """在 BaseSeg/EMA wrapper 内部找到带 order_prompt 的 PointMambaEncoder 模块。"""
#     m = _unwrap_model(model_like)
#
#     # fast path: BaseSeg.encoder
#     if hasattr(m, "encoder"):
#         enc = getattr(m, "encoder")
#         for mod in enc.modules():
#             if hasattr(mod, "use_order_prompt") and getattr(mod, "use_order_prompt", False) \
#                and hasattr(mod, "order_prompt") and hasattr(mod, "per_layer_prompt_indexe") \
#                and hasattr(mod, "mamba_layers_orders"):
#                 return mod
#
#     # fallback: search whole model tree
#     for mod in m.modules():
#         if hasattr(mod, "use_order_prompt") and getattr(mod, "use_order_prompt", False) \
#            and hasattr(mod, "order_prompt") and hasattr(mod, "per_layer_prompt_indexe") \
#            and hasattr(mod, "mamba_layers_orders"):
#             return mod
#     return None

# @torch.no_grad()
# def compute_order_similarity_rankings(model_like, order_vocab=ORDER_VOCAB):
#     """返回：
#     rankings: dict[order] -> list[(other_order, cosine)]
#     sim: (O,O) numpy 相似度矩阵（已 L2 归一化后点积）
#     """
#     enc = _find_order_prompt_encoder(model_like)
#     if enc is None:
#         raise RuntimeError("Cannot find encoder module with order_prompt/per_layer_prompt_indexe for similarity logging.")
#
#     weights = enc.order_prompt.weight  # (overall_prompt_nums, 384)
#
#     # —— prototype 定义：对某个 order 的 K 个 prompt token 做平均池化（就是你图里的 v_o = 1/K sum e_{o,k}）
#     protos = []
#     for order in order_vocab:
#         # 一个 order 可能在多个 layer 出现；这里做“按 layer 平均”以保持鲁棒（你当前配置每个 order 只出现一次）
#         layer_idxs = [i for i, o in enumerate(enc.mamba_layers_orders) if o == order]
#         if len(layer_idxs) == 0:
#             raise RuntimeError(f"Order '{order}' not found in encoder.mamba_layers_orders.")
#
#         vecs = []
#         for li in layer_idxs:
#             s, e = enc.per_layer_prompt_indexe[li]
#             v = weights[s:e].mean(dim=0)  # 平均池化：K 个 token -> 1 个 384 维 prototype
#             vecs.append(v)
#
#         v = torch.stack(vecs, dim=0).mean(dim=0)
#         v = v / (v.norm(p=2) + 1e-12)  # L2 normalize（cosine 的标准做法）
#         protos.append(v)
#
#     protos = torch.stack(protos, dim=0)                 # (O, 384)
#     sim = (protos @ protos.t()).float().cpu().numpy()   # prototype cosine: cos(v_i, v_j)
#
#     # —— 每个 order 对其它 8 个 order 做降序排序
#     rankings = {}
#     for i, oi in enumerate(order_vocab):
#         pairs = []
#         for j, oj in enumerate(order_vocab):
#             if i == j:
#                 continue
#             pairs.append((oj, float(sim[i, j])))
#         pairs.sort(key=lambda x: x[1], reverse=True)
#         rankings[oi] = pairs
#
#     return rankings, sim
# # ================================================================================================


# train.py  —— 放在 train() 外或内都可
def collate_pcm(batch):
    pos = torch.stack([b["pos"] for b in batch], dim=0)  # (B, N, 3)
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B, C, N)  已在 dataset 转置过
    y = torch.stack([b["y"] for b in batch], dim=0)  # (B, N)
    return {"pos": pos, "x": x, "y": y}


# [ADD] —— 构建 EMA 模型
if use_ema:
    def ema_avg(ema_param, param, n_averaged):
        return ema_decay * ema_param + (1.0 - ema_decay) * param


    ema_model = AveragedModel(model, avg_fn=ema_avg)


def train():
    # train_loader = DataLoader(Tinto(partition='train', num_points=4096, test_area=args_test_area),
    #                           num_workers=2, batch_size=args_batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(Tinto(partition='test', num_points=4096, test_area=args_test_area),
    #                         num_workers=2, batch_size=args_batch_size, shuffle=True, drop_last=False)

    # train_loader = DataLoader(
    #     Tinto(partition='train', num_points=4096),
    #     batch_size=args_batch_size, shuffle=True,
    #     num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True
    # )
    # test_loader = DataLoader(
    #     Tinto(partition='test', num_points=4096),
    #     batch_size=args_batch_size, shuffle=False,
    #     num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
    # )
    train_loader = DataLoader(
        Tinto(partition='train', num_points=4096),
        batch_size=args_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True,
        collate_fn=collate_pcm  # ★ 新增
    )
    test_loader = DataLoader(
        Tinto(partition='test', num_points=4096),
        batch_size=args_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False,
        collate_fn=collate_pcm  # ★ 新增
    )

    device = torch.device("cuda" if args_cuda else "cpu")

    # 改--------------------------------------------------------------------
    # if args_use_sgd:
    #     print("Use SGD")
    #     opt = optim.SGD(model.parameters(), lr=args_lr, momentum=args_momentum, weight_decay=1e-4)
    # else:
    #     print("Use Adam")
    #     opt = optim.Adam(model.parameters(), lr=args_lr, weight_decay=1e-4)
    #
    # if args_scheduler == 'cos':
    #     scheduler = CosineAnnealingLR(opt, args_epochs, eta_min=1e-3)
    # elif args_scheduler == 'step':
    #     scheduler = StepLR(opt, 20, 0.5, args_epochs)

    # AdamW 是 PCM 默认的优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args_lr,
        weight_decay=0.05,  # 与 PCM 配置一致
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Cosine Annealing 调度器，最小学习率 1e-5 与默认配置匹配
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=args_epochs,
    #     eta_min=1e-5
    # )
    # [REPLACE]
    min_lr = 1e-6
    warmup_epochs = 5
    scheduler = CosineAnnealingLR(optimizer, T_max=args_epochs - warmup_epochs, eta_min=min_lr)

    # criterion = cal_loss    #计算预测结果 pred 和真实标签 gold 之间的交叉熵损失
    label_smoothing = 0.10  # 可在 0.05~0.2 间微调

    # [ADD] 放在定义 criterion 的位置，替换原 cal_loss
    def criterion_fn(seg_pred_logits, seg_target):
        # seg_pred_logits: (B, N, num_classes)
        # seg_target     : (B, N)
        return F.cross_entropy(
            seg_pred_logits.view(-1, 11),
            seg_target.view(-1),
            label_smoothing=label_smoothing
        )

    criterion = criterion_fn

    best_test_iou = 0
    best_test_acc = 0

    plot_train = np.zeros((args_epochs, 3))
    plot_test = np.zeros((args_epochs, 3))
    # [ADD] 统计缓存
    lr_hist = np.zeros(args_epochs)  # 学习率
    grad_hist = np.zeros(args_epochs)  # 梯度范数（被裁剪前的总范数）
    test_iou_cls = np.zeros((args_epochs, args_num_class))  # 每类 IoU（验证集）
    train_iou_cls = np.zeros((args_epochs, args_num_class))  # 每类 IoU（训练集，可选）

    for epoch in range(args_epochs):
        model.train()
        print(f"当前训练轮次: {epoch + 1}/{args_epochs}")
        ####################
        # Train
        ####################

        # [ADD] —— 线性 warmup：把 LR 从 0 线性拉到 args_lr
        if epoch < warmup_epochs:
            lr_scale = float(epoch + 1) / float(warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = args_lr * lr_scale
        else:
            scheduler.step()

        # [ADD]
        bn_freeze_epoch = 40
        if epoch == bn_freeze_epoch:
            def _freeze_bn(m):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()  # 固定 running_mean/var

            model.apply(_freeze_bn)

        # [ADD]
        current_lr = optimizer.param_groups[0]['lr']
        lr_hist[epoch] = current_lr
        print(f"[Epoch {epoch + 1}/{args_epochs}] LR={current_lr:.3e}")

        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []  # 计算总体准确率
        train_pred_cls = []  # 存每个 batch 的二维标签矩阵，方便计算每类 IoU
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        # for data, seg in train_loader:
        #     data, seg = data.to(device), seg.to(device) #seg: (B, N)，对应每个点的语义标签
        #     assert data.shape[-1] == 88, f"期望 88 维，拿到 {data.shape[-1]}"
        #     data = data.permute(0, 2, 1)
        #     batch_size = data.size()[0]
        #     opt.zero_grad()
        #     seg_pred = model(data)  #(B, num_class, N)，表示每个点属于每个类别的预测分数（logits）。
        #     seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        #     loss = criterion(seg_pred.view(-1, args_num_class), seg.view(-1,1).squeeze())
        for batch in train_loader:
            pos = batch["pos"].to(device)  # (B, N, 3)
            x = batch["x"].to(device)  # (B, C, N)  C=85（示例）
            seg = batch["y"].to(device)  # (B, N)
            B = pos.size(0)

            optimizer.zero_grad()
            # PCM 前向（PointMambaEncoder/Decoder/SegHead 组合）
            logits = model({"pos": pos, "x": x})
            # _eval_model = ema_model if use_ema else model
            # logits = _eval_model({"pos": pos, "x": x})

            seg_pred = logits.permute(0, 2, 1).contiguous()  # (B, N, num_class)
            # loss = criterion(seg_pred.view(-1, args_num_class), seg.reshape(-1))
            # [REPLACE]
            loss = criterion(seg_pred, seg)

            loss.backward()
            # [ADD]
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            # [REPLACE] 记录梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            grad_hist[epoch] = float(grad_norm)

            optimizer.step()
            # [ADD]
            if use_ema:
                ema_model.update_parameters(model)

            pred = seg_pred.max(dim=2)[1]  # 预测类别 (batch_size, num_points)
            count += B
            train_loss += loss.item() * B
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)

        # 常让学习率 lr 随 epoch 变化，例如：
        # 先大后小 → 提高稳定性；在模型收敛时逐步减小 lr → 防止振荡。
        # if args_scheduler == 'cos':
        #     scheduler.step()
        # elif args_scheduler == 'step':
        #     if optimizer.param_groups[0]['lr'] > 1e-5:
        #         scheduler.step()
        #     if optimizer.param_groups[0]['lr'] < 1e-5:
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = 1e-5
        # scheduler.step()
        train_true_cls = np.concatenate(train_true_cls)  # 所有批次的真实标签
        train_pred_cls = np.concatenate(train_pred_cls)  # 所有批次的预测类别
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)  # 每类的平均准确率
        train_true_seg = np.concatenate(train_true_seg, axis=0)  # 真实标签矩阵
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)  # 预测标签矩阵
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg, args_num_class)  # 计算语义分割 IoU
        print('Train IoU = ', train_ious)
        print('Train mean IoU = ', np.mean(train_ious))
        print('Train loss = ', train_loss * 1.0 / count)
        plot_train[epoch, 0] = train_loss * 1.0 / count
        plot_train[epoch, 1] = np.mean(train_ious)
        plot_train[epoch, 2] = train_acc
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,

                                                                                                  np.mean(train_ious))
        # io.cprint(outstr)
        print(outstr)
        ####################
        # Test
        ####################
        # model.eval()
        _eval_model = ema_model if use_ema else model
        _eval_model.eval()
        test_loss = 0.0
        count = 0.0

        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        # for data, seg in test_loader:
        #     assert data.shape[-1] == 88, f"期望 88 维，拿到 {data.shape[-1]}"
        #     data, seg = data.to(device), seg.to(device)
        #     data = data.permute(0, 2, 1)
        #     batch_size = data.size()[0]
        #     seg_pred = model(data)
        #     seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        #     loss = criterion(seg_pred.view(-1, args_num_class), seg.view(-1,1).squeeze())
        with torch.no_grad():
            for batch in test_loader:
                pos = batch["pos"].to(device)  # (B, N, 3)
                x = batch["x"].to(device)  # (B, C, N)  C=85（示例）
                seg = batch["y"].to(device)  # (B, N)
                B = pos.size(0)
                # PCM 前向（PointMambaEncoder/Decoder/SegHead 组合）
                # logits = model({"pos": pos, "x": x})
                # _eval_model = ema_model if use_ema else model
                logits = _eval_model({"pos": pos, "x": x})
                seg_pred = logits.permute(0, 2, 1).contiguous()  # (B, N, num_class)
                # loss = criterion(seg_pred.view(-1, args_num_class), seg.reshape(-1))
                # [REPLACE]
                loss = criterion(seg_pred, seg)

                pred = seg_pred.max(dim=2)[1]
                count += B
                test_loss += loss.item() * B
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                # [ADD] 混淆矩阵（仅在关键 epoch 保存）
                # if (epoch + 1) in (30, 100):
                #     from sklearn.metrics import confusion_matrix
                #     import matplotlib.pyplot as plt
                #     cm = confusion_matrix(test_true_cls, test_pred_cls, labels=list(range(args_num_class)))
                #     fig_cm, ax_cm = plt.subplots()
                #     im = ax_cm.imshow(cm, interpolation='nearest')
                #     ax_cm.set_title(f'Confusion Matrix @ epoch {epoch + 1}')
                #     ax_cm.set_xlabel('Predicted');
                #     ax_cm.set_ylabel('True')
                #     fig_cm.colorbar(im, ax=ax_cm)
                #     plt.tight_layout()
                #     plt.savefig(f'Mamba_model/confmat_epoch{epoch + 1}.png', dpi=200)
                #     plt.close(fig_cm)

        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)

        # 1) 断言一维 & 整型标签（避免类型/形状隐性错误）
        assert test_true_cls.ndim == 1 and test_pred_cls.ndim == 1, "y_true/y_pred 必须是 1D"
        assert np.issubdtype(test_true_cls.dtype, np.integer) and np.issubdtype(test_pred_cls.dtype,
                                                                                np.integer), "标签需为整型"

        # 2) 断言类别取值在 [0, args_num_class)
        assert test_true_cls.min() >= 0 and test_pred_cls.min() >= 0, "存在负标签，请检查 ignore_index"
        assert test_true_cls.max() < args_num_class and test_pred_cls.max() < args_num_class, "预测或真值超出类别上限"

        # 3) 断言网络输出通道数与全局类别数一致（防止改配置后通道不匹配）
        # 在得到 seg_pred 之后加一次（训练或测试任一处都可以）：
        # assert seg_pred.shape[-1] == args_num_class, f"预测通道={seg_pred.shape[-1]} 与 args_num_class={args_num_class} 不一致"

        # [ADD] —— 现在是一维向量后再计算
        from sklearn.metrics import confusion_matrix
        y_true = test_true_cls.astype(np.int64).ravel()
        y_pred = test_pred_cls.astype(np.int64).ravel()
        valid = (y_true >= 0) & (y_true < args_num_class) & (y_pred >= 0) & (y_pred < args_num_class)
        y_true, y_pred = y_true[valid], y_pred[valid]
        if (epoch + 1) in (30, 100):
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(args_num_class))
            fig_cm, ax_cm = plt.subplots()
            im = ax_cm.imshow(cm, interpolation='nearest')
            ax_cm.set_title(f'Confusion Matrix @ epoch {epoch + 1}')
            ax_cm.set_xlabel('Predicted');
            ax_cm.set_ylabel('True')
            fig_cm.colorbar(im, ax=ax_cm)
            ax_cm.set_xticks(np.arange(args_num_class));
            ax_cm.set_yticks(np.arange(args_num_class))
            plt.tight_layout()
            plt.savefig(f'Mamba_model/confmat_epoch{epoch + 1}.png', dpi=200)
            plt.close(fig_cm)

        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg, args_num_class)
        print('Test IoU = ', test_ious)
        print('Test mean IoU = ', np.mean(test_ious))
        print('Test loss = ', test_loss * 1.0 / count)
        plot_test[epoch, 0] = test_loss * 1.0 / count
        plot_test[epoch, 1] = np.mean(test_ious)
        plot_test[epoch, 2] = test_acc
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss * 1.0 / count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        # io.cprint(outstr)
        print(outstr)

        # ===================== [REPLACE] 每个 epoch 计算一次语义相似度排序，并写入 similarity_log =====================
        epoch1 = epoch + 1
        if should_compute_similarity(epoch1):
            try:
                # 用 _eval_model：若启用 EMA，则记录的是 EMA 参数下的 prompt 语义（更稳）
                sim_rankings, _ = compute_order_similarity_rankings(_eval_model, ORDER_VOCAB)

                with open(SIMILARITY_LOG_PATH, "a", encoding="utf-8") as sf:
                    sf.write(f"Epoch {epoch1:03d}\n")
                    for oi in ORDER_VOCAB:
                        pairs = sim_rankings[oi]  # 8 个其它 order 的降序列表
                        sf.write(oi + ": " + ", ".join([f"{oj}({s:.4f})" for oj, s in pairs]) + "\n")
                    sf.write("-" * 60 + "\n")

                print(f"[Epoch {epoch1}] semantic similarity ranking saved -> {SIMILARITY_LOG_PATH}")

            except Exception as e:
                # 不让训练中断：把错误也写到日志里
                with open(SIMILARITY_LOG_PATH, "a", encoding="utf-8") as sf:
                    sf.write(f"Epoch {epoch1:03d} | similarity logging failed: {repr(e)}\n")
                    sf.write("-" * 60 + "\n")
                print(f"[Epoch {epoch1}] similarity logging failed: {repr(e)}")
        # ============================================================================================================

        # # ===================== [ADD] 每 N 个 epoch 计算一次语义相似度排序，并写入 similarity_log =====================
        # epoch1 = epoch + 1
        # if should_compute_similarity(epoch1):
        #     try:
        #         # 用 _eval_model：若启用 EMA，则记录的是 EMA 参数下的 prompt 语义（更稳）
        #         sim_rankings, _ = compute_order_similarity_rankings(_eval_model, ORDER_VOCAB)
        #
        #         with open(SIMILARITY_LOG_PATH, "a", encoding="utf-8") as sf:
        #             sf.write(f"Epoch {epoch1:03d} (interval={_sim_interval(epoch1)})\n")
        #             for oi in ORDER_VOCAB:
        #                 pairs = sim_rankings[oi]  # 8 个其它 order 的降序列表
        #                 sf.write(oi + ": " + ", ".join([f"{oj}({s:.4f})" for oj, s in pairs]) + "\n")
        #             sf.write("-" * 60 + "\n")
        #
        #         print(f"[Epoch {epoch1}] semantic similarity ranking saved -> {SIMILARITY_LOG_PATH}")
        #
        #     except Exception as e:
        #         # 不让训练中断：把错误也写到日志里
        #         with open(SIMILARITY_LOG_PATH, "a", encoding="utf-8") as sf:
        #             sf.write(f"Epoch {epoch1:03d} | similarity logging failed: {repr(e)}\n")
        #             sf.write("-" * 60 + "\n")
        #         print(f"[Epoch {epoch1}] similarity logging failed: {repr(e)}")
        # # ============================================================================================================

        # # ===================== [ADD] 每 N 个 epoch 计算一次语义相似度排序，并写入 similarity_log =====================
        # if (epoch + 1) % SIM_EVERY_EPOCHS == 0:
        #     try:
        #         # 用 _eval_model：若启用 EMA，则记录的是 EMA 参数下的 prompt 语义（更稳）
        #         sim_rankings, _ = compute_order_similarity_rankings(_eval_model, ORDER_VOCAB)
        #
        #         with open(SIMILARITY_LOG_PATH, "a", encoding="utf-8") as sf:
        #             sf.write(f"Epoch {epoch + 1:03d}\n")
        #             for oi in ORDER_VOCAB:
        #                 pairs = sim_rankings[oi]  # 8 个其它 order 的降序列表
        #                 sf.write(oi + ": " + ", ".join([f"{oj}({s:.4f})" for oj, s in pairs]) + "\n")
        #             sf.write("-" * 60 + "\n")
        #
        #         print(f"[Epoch {epoch + 1}] semantic similarity ranking saved -> {SIMILARITY_LOG_PATH}")
        #
        #     except Exception as e:
        #         # 不让训练中断：把错误也写到日志里
        #         with open(SIMILARITY_LOG_PATH, "a", encoding="utf-8") as sf:
        #             sf.write(f"Epoch {epoch + 1:03d} | similarity logging failed: {repr(e)}\n")
        #             sf.write("-" * 60 + "\n")
        #         print(f"[Epoch {epoch + 1}] similarity logging failed: {repr(e)}")
        # # ============================================================================================================

        # === 保存每个 epoch 的训练和测试指标到文件 ===
        log_path = 'Mamba_model/acc_log.txt'
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch + 1:03d}\n")
            f.write(
                f"Train: loss={plot_train[epoch, 0]:.6f}, acc={plot_train[epoch, 2]:.6f}, IoU={plot_train[epoch, 1]:.6f}\n")
            f.write(
                f"Test : loss={plot_test[epoch, 0]:.6f}, acc={plot_test[epoch, 2]:.6f}, IoU={plot_test[epoch, 1]:.6f}\n")
            f.write("-" * 60 + "\n")

        # 保存三类 checkpoint：最佳 IoU 模型,最佳 Acc 模型,每个 epoch 的模型
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'Mamba_model/%s.t7' % (args_exp_name))
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'Mamba_model/%s_best_acc.t7' % (args_exp_name))
        # [ADD] —— 在“best_acc”保存之后
        if use_ema and test_acc >= best_test_acc:
            torch.save(ema_model.module.state_dict(), f'Mamba_model/{args_exp_name}_best_acc_ema.t7')

        torch.save(model.state_dict(), 'Mamba_model/%s-%d.t7' % (args_exp_name, epoch))

    return plot_train, plot_test, lr_hist, grad_hist


import numpy as np


def _safe_div(num, den):
    den = np.maximum(den, 1e-12)
    return num / den


# To train !!

# Need to change:
# train_path = 'lithonet_sem_seg_hdf5_data_Experiment'
# test_path = 'lithonet_sem_seg_hdf5_data_Experiment'
args_exp_name = 'Experiment_vnir'
# ----------------------------------需要修改--------
# 控制分区
args_epochs = 100
# args_num_class = 10
args_num_class = 11
args_batch_size = 24
dim_rgb = 3
dim_vnir = 51
dim_geo = 28
# data_dimension = 175 # RGB, vnir (144 features), geometric (28 features)
data_dimension = 88  # XYZ(3)+normXYZ(3)+RGB(3)+VNIR(51)+Geo(28)

# Arguments no need to change (following original codes):
args_cuda = True
args_model = 'mamba'
args_k = 20
args_emb_dims = 1024
args_dropout = 0.5
args_use_sgd = True
args_scheduler = 'cos'
args_lr = 0.001
args_momentum = 0.9

if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)  # Windows/多进程安全

    plot_train, plot_test, lr_hist, grad_hist = train()

    # --- 以下保持你原来的绘图与保存 ---
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(plot_train[:, 0], 'tab:red');
    axs[0, 0].set_title('Train loss')
    axs[0, 1].plot(plot_test[:, 0], 'tab:red');
    axs[0, 1].set_title('Test loss')
    # axs[1, 0].plot(plot_train[:,1], 'tab:red'); axs[1, 0].set_title('Train IoU')
    # axs[1, 1].plot(plot_test[:,1], 'tab:red');  axs[1, 1].set_title('Test IoU')
    axs[1, 0].plot(plot_train[:, 2], 'tab:green')
    axs[1, 0].set_title('Train Accuracy')
    axs[1, 1].plot(plot_test[:, 2], 'tab:green')
    axs[1, 1].set_title('Test Accuracy')

    for ax in axs.flat: ax.label_outer()
    plt.savefig('Mamba_model/plot.png', bbox_inches='tight')
    np.savetxt('Mamba_model/plot.txt', np.hstack((plot_train, plot_test)))

    plt.figure()
    plt.plot(lr_hist)
    plt.title('Learning Rate (warmup + cosine)')
    plt.xlabel('Epoch');
    plt.ylabel('LR')
    plt.savefig('Mamba_model/lr_curve.png', bbox_inches='tight');
    plt.close()

    plt.figure()
    plt.plot(grad_hist)
    plt.title('Gradient Norm (pre-clip)')
    plt.xlabel('Epoch');
    plt.ylabel('||grad||')
    plt.savefig('Mamba_model/grad_norm.png', bbox_inches='tight');
    plt.close()
