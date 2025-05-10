import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lang2vec.lang2vec as l2v

from espnet2.lid.loss.abs_loss import AbsLoss


class AAMSoftmaxSCTopKLang2Vec(AbsLoss):
    r"""
    AAMSoftmax with intertopk and subcenter, and lang2vec prediction.

    The AAMSoftmax part is same with ArcMarginProduct_intertopk_subcenter in 
    `aamsoftmax_subcenter_intertopk.py`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: norm of input feature
        margin: margin
        cos(theta + margin)
        K: number of sub-centers
        k_top: number of hard samples
        mp: margin penalty of hard samples
        do_lm: whether do large margin finetune

        NOTE(qingzheng):
        apply_last: used when use transformer_ecapa, which use lang2vec self-condition
        on downstream transformer layers, since the lang2vec prediction loss has been
        applied to the downstream intermediate layers, the lang2vec prediction loss which
        was on the final layer should be considered to be removed.
    """

    def __init__(
        self,
        nout,
        nclasses,
        scale=32.0,
        margin=0.2,
        easy_margin=False,
        K=3,
        mp=0.06,
        k_top=5,
        do_lm=False,
        lang2vec_dim: int = None,
        lang2vec_type: str = None, # geo, phonology_knn, syntax_knn, inventory_knn
        lang2vec_weight: float = None,
        apply_last: bool = True,
    ):
        super().__init__(nout)
        self.in_features = nout
        self.out_features = nclasses
        self.scale = scale
        self.margin = margin
        self.do_lm = do_lm

        # intertopk + subcenter
        self.K = K
        if do_lm:  # if do LMF, remove hard sample penalty
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp
            self.k_top = k_top

        # initial classifier
        self.weight = nn.Parameter(torch.FloatTensor(self.K * nclasses, nout))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin
        )  # this can make the output more continuous
        ########
        self.m = self.margin
        ########
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

        self.ce = nn.CrossEntropyLoss()

        self.lang2vec_dim = lang2vec_dim
        self.lang2vec_type = lang2vec_type
        self.lang2vec_weight = lang2vec_weight
        self.apply_last = apply_last

        # NOTE(qingzheng): no matter whether apply_last is True or False,
        # the lang2vec_head, lang2vec_loss should be initialized, because 
        # when apply_last is False these will be used in the transformer_ecapa
        if (
            self.lang2vec_dim is not None and 
            self.lang2vec_type is not None and 
            self.lang2vec_weight is not None
        ):
            if lang2vec_type == "geo":
                self.lang2vec_head = nn.Sequential(
                    nn.Linear(nout, lang2vec_dim),
                )
                self.lang2vec_loss = nn.MSELoss()
            elif lang2vec_type in ["phonology_knn", "syntax_knn", "inventory_knn"]:
                self.lang2vec_head = nn.Sequential(
                    nn.Linear(nout, lang2vec_dim),
                )
                self.lang2vec_loss = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss combines sigmoid and binary cross entropy, which use the log-sum-exp trick for numerical stability.
            else:
                raise ValueError(f"Unknown lang2vec type: {lang2vec_type}, support lang2vec types: geo, phonology_knn, syntax_knn, inventory_knn")


    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        # hard sample margin is increasing as margin
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(self, input, label=None, lang2vec=None):
        
        # there are k subcenter per class in weight
        cosine = F.linear(
            F.normalize(input), F.normalize(self.weight)
        )  # (batch, out_dim * k)
        cosine = torch.reshape(
            cosine, (-1, self.out_features, self.K)
        )  # (batch, out_dim, k)
        # subcenter max pooling, compute k max cosine, use the max one
        # k subcenter per class in weight, cosine is the simlarity to these subcenters
        # then select the top one. This is for intra-class marginization.
        cosine, _ = torch.max(cosine, 2)  # (batch, out_dim)
        pred_lids = torch.argmax(cosine, dim=1) # (batch,)

        if label is not None:
            if len(label.size()) == 2:
                label = label.squeeze(1)
            accuracy = (pred_lids == label).float().mean()
        else: # inference
            loss = None
            accuracy = None
            return loss, accuracy, pred_lids

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # phi: cos(theta + m), true class, +m, /, cos(+m) \, 
        # -logcos(+m) /, original goal is \, hence is the penalty
        phi = cosine * self.cos_m - sine * self.sin_m 
        # phi_mp: for the topk samples, negative class
        # cos(theta - mp), -mp, \, cos(-mp) /, -logcos(-mp) \,
        # original goal is / (for negative class, we want max loss),
        # hence is the penalty
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp # cos(theta - mp)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        if self.k_top > 0:
            # topk (j != y_i), the top k expect the true class
            _, top_k_index = torch.topk(
                cosine - 2 * one_hot, self.k_top
            )  # exclude j = y_i
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)

            # sum
            output = (
                (one_hot * phi) # true class, phi
                + (top_k_one_hot * phi_mp) # topk class (negative), phi_mp
                + ((1.0 - one_hot - top_k_one_hot) * cosine) # other class, cosine, without margin
            )
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        loss = self.ce(output, label)
        class_loss = loss # classification loss
        lang2vec_loss = None

        if (
            lang2vec is not None and
            self.lang2vec_dim is not None and 
            self.lang2vec_type is not None and 
            self.lang2vec_weight is not None and
            self.apply_last
        ):
            assert 0 < self.lang2vec_weight < 1, f"lang2vec_weight should be in (0, 1), but got {self.lang2vec_weight}"
            lang2vec_loss = self.lang2vec_loss(self.lang2vec_head(input), lang2vec)
            loss *= (1 - self.lang2vec_weight)
            loss += self.lang2vec_weight * lang2vec_loss

        return loss, accuracy, pred_lids, class_loss, lang2vec_loss
