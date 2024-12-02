# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
from torch import nn
from mgda import mgda

class ofa(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(ofa_loss, self).__init__(*args, **kwargs)
        
    def forward(stu_logit, tea_logit, target_mask):
        return ofa_loss(stu_logit, tea_logit, target_mask)


class infonce_loss(nn.Module):

    def __init__(self, temperature=0.5, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode


    def forward(self, stu_logit, tea_logit, negative_keys=None):
        return info_nce(stu_logit, tea_logit, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def ofa_loss(logits_student, logits_teacher, target_mask, eps=4, temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
        # print(logits.shape,labels.shape)
        
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.stage = [3,6,13,18]

    def forward(self, inputs, outputs, labels, dis_dict, optimizer, model):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        losses = {}
        
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        else:
            outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        losses["gt"] = base_loss.item()
        if self.distillation_type == 'none':
            return base_loss,losses

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        
        losses["kd"] = distillation_loss.item()
        
        #ofa
        num_classes = 1000
        if len(labels.shape) != 1:  # label smoothing
            target_mask = F.one_hot(labels.argmax(-1), num_classes)
        else:
            target_mask = F.one_hot(labels, num_classes)
        stage = self.stage
        ofaloss = ofa_loss(dis_dict[stage[0]],teacher_outputs,target_mask)+ofa_loss(dis_dict[stage[1]],teacher_outputs,target_mask)+ofa_loss(dis_dict[stage[2]],teacher_outputs,target_mask)
        losses["ofa"] = ofaloss.item()
        #contrastive loss
        info = infonce_loss()
        info_loss = info(dis_dict[stage[0]],teacher_outputs)+info(dis_dict[stage[1]],teacher_outputs)+info(dis_dict[stage[2]],teacher_outputs)
        losses["infonce"] = info_loss.item()
        #MGDA
        # scale = mgda(optimizer=optimizer, model=model, losses=losses)
        base_loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        # loss = scale["gt"] * base_loss + scale["kd"]*distillation_loss + scale["infonce"] * info_loss + scale["ofa"] * ofaloss
        # loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha + scale["infonce"] * info_loss + scale["ofa"] * ofaloss
        # print(base_loss.item(),distillation_loss.item(),info_loss.item(),ofaloss.item())
        loss = base_loss + info_loss/(info_loss/base_loss).detach() + ofaloss/(ofaloss/base_loss).detach()
        
        return loss,losses
    
    
# from timm.models import create_model
# from devr import devr_tiny
# from torchinfo import summary
# # import timm
# model = create_model(
#         "devr_tiny",
#         pretrained=False,
#         num_classes=1000,
#         drop_rate=0.,
#         drop_path_rate=0.,
#         drop_block_rate=None,
#         img_size=224
#     )
# summary(model)
# # model.train()
# # model = model.to("cuda")
# # x = torch.rand((64,3,224,224))
# # x = x.to("cuda")

# # # 检查模型的所有子模块和参数
# # def check_device(module):
# #     for name, param in module.named_parameters():
# #         print(f"Parameter {name} is on device: {param.device}")
# #     for name, buffer in module.named_buffers():
# #         print(f"Buffer {name} is on device: {buffer.device}")

# # check_device(model)
# # output,dic = model(x)
# teacher_model = create_model(
#             "convnext_small",
#             pretrained=True,
#             num_classes=1000,
#             global_pool='avg',
#         )
# # print(teacher_model.state_dict().keys())
# # checkpoint = torch.load("convnext_small_22k_1k_224.pth", map_location='cpu')
# # # print([key for key,_ in checkpoint['model'].items()])
# # # teacher_model.load_state_dict(checkpoint['model'])
# # teacher_model.to("cuda")
# # for key,_ in checkpoint['model'].items():
# #     if key not in teacher_model.state_dict().keys():
# #         print(key)

# # criterion = DistillationLoss(nn.CrossEntropyLoss().to("cuda"),teacher_model,"hard",0.5,1.0)
# # labels = nn.functional.softmax(torch.rand(64,1000)).to("cuda")
# # optimizer = torch.optim.SGD(model.parameters())
# # loss = criterion(x, output, labels, dic, optimizer, model)
# # summary(teacher_model)
# # print(loss)

# # model_list = timm.list_models()

# # print(model_list)

# print(teacher_model.default_cfg)