# from vrwkv import VRWKV
import torch
from torch import nn as nn
from torchvision import datasets
from torchinfo import summary
from timm.models import create_model
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from engine import evaluate
def main():
    x = torch.randn((1,3,224,224)).cuda()
    # model = VRWKV().cuda()
    # teacher_model = create_model(
    #             args.teacher_model,
    #             pretrained=False,
    #             num_classes=args.nb_classes,
    #             global_pool='avg',
    #         )
    device = torch.device("cuda")
    teacher_model = create_model(
            'regnety_160',
            pretrained=False,
            num_classes=1000,
            global_pool='avg',
        )
    teacher_model.to('cuda')
    checkpoint = torch.load('regnety_160-a5fe301d.pth', map_location='cpu')
    teacher_model.load_state_dict(checkpoint['model'])
    teacher_model.eval()
    t = []
    t.append(
            transforms.Resize(int(224*0.875), interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
    t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(t)
    dataset_val = datasets.ImageFolder("~/Vision-RWKV/classification/data/imagenet/data/val", transform=transform)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(64),
        num_workers=10,
        pin_memory=True,
        drop_last=False
    )
    # print(data_loader_val)
    test_stats = evaluate(data_loader_val, teacher_model, device)
    print(test_stats)
    # summary(model)
    # x = model(x,(14,14))
    # output = model(x)
    # print(output.shape)
    # print(x[0].shape,x[1].shape)
    
if __name__=="__main__":
    main()
    