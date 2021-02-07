# EXTERNAL LIBRARIES
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import recoloradv.color_spaces as cs
import recoloradv.color_transformers as ct
import recoloradv.mister_ed.adversarial_attacks as aa
import recoloradv.mister_ed.adversarial_perturbations as ap
import recoloradv.mister_ed.loss_functions as lf
import recoloradv.mister_ed.utils.pytorch_utils as utils
import recoloradv.perturbations as pt
from cifar10_models import (alexnet, densenet121, googlenet,
                            inception_v3, mobilenet_v2,
                            resnet18, resnet50, squeezenet1_0,
                            vgg16_bn)

# module_path = os.path.abspath('mister_ed')
# if module_path not in sys.path:
#     sys.path.append(module_path)


def tensor2cv2(t):
    """
    converts the pytorch tensor to img by transposing the tensor and normalizing it
    :param t: input tensor
    :return: numpy array with last dim be the channels and all values in range [0, 1]
    """
    t_np = t.detach().cpu().numpy().transpose(1, 2, 0)
    t_np = (t_np - t_np.min()) / (t_np.max() - t_np.min())
    t_np *= 255
    t_np = cv2.cvtColor(t_np, cv2.COLOR_RGB2BGR)
    return t_np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pre(img):
    trans = transforms.Compose([transforms.ToTensor()])
    return trans(img)


def main(args):
    ############ Ground Truth #########################
    df = pd.read_csv(args.gt)
    image_list = df['filename'].values
    # truth = df['groundtruth'].values
    truth = dict(zip(image_list, df['groundtruth'].values))

    # set random seed
    torch.manual_seed(args.trial)
    np.random.seed(args.trial)
    torch.backends.cudnn.deterministic = True

    # Source model
    if args.model == 'resnet50':
        source_model = resnet50(pretrained=True)
    elif args.model == 'vgg16':
        source_model = vgg16_bn(pretrained=True)
    elif args.model == 'inception_v3':
        source_model = inception_v3(pretrained=True)
    else:
        raise NotImplementedError('{} is not allowed!'.format(args.model))

    source_model.eval()
    source_model.to(device)
    # print("source_model", next(source_model.parameters()).is_cuda)

    total = 0
    model_list = []

    # model_name = ["resnet50", "resnet18", "vgg16", "densenet121",
    #               "googlenet", "mobilenet_v2", "inception_v3", "alexnet", "squeezenet1_0"]
    model_name = ["squeezenet1_0", "alexnet"]

    if "resnet50" in model_name:
        model_list.append(resnet50(pretrained=True))
    if "resnet18" in model_name:
        model_list.append(resnet18(pretrained=True))
    if "vgg16" in model_name:
        model_list.append(vgg16_bn(pretrained=True))
    if "densenet121" in model_name:
        model_list.append(densenet121(pretrained=True))
    if "googlenet" in model_name:
        model_list.append(googlenet(pretrained=True))
    if "mobilenet_v2" in model_name:
        model_list.append(mobilenet_v2(pretrained=True))
    if "inception_v3" in model_name:
        model_list.append(inception_v3(pretrained=True))
    if "squeezenet1_0" in model_name:
        model_list.append(squeezenet1_0(pretrained=True))
    if "alexnet" in model_name:
        model_list.append(alexnet(pretrained=True))

    [model.eval() for model in model_list]
    [model.to(device) for model in model_list]
    trans = np.zeros(len(model_list))
    sr = np.zeros(len(model_list))

    recoloradv_threat = ap.ThreatModel(pt.ReColorAdv, {
        'xform_class': ct.FullSpatial,
        'cspace': cs.RGBColorSpace(),  # controls the color space used
        'lp_style': 'inf',
        'lp_bound': 0.047,  # [epsilon_1, epsilon_2, epsilon_3]
        'xform_params': {
            'resolution_x': 25,            # R_1
            'resolution_y': 25,            # R_2
            'resolution_z': 25,            # R_3
        },
        'use_smooth_loss': True,
    })

    # Attack Model
    normalizer = utils.DifferentiableNormalize(mean=[0.4914, 0.4822, 0.4465],
                                               std=[0.2023, 0.1994, 0.2010])

    # Now, we define the main optimization term (the Carlini & Wagner f6 loss).
    adv_loss = lf.CWLossF6(source_model, normalizer)

    # We also need the smoothness loss.
    smooth_loss = lf.PerturbationNormLoss(lp=2)

    # We combine them with a RegularizedLoss object.
    attack_loss = lf.RegularizedLoss({'adv': adv_loss, 'smooth': smooth_loss},
                                     # lambda = 0.05
                                     {'adv': 1.0,      'smooth': 0.05},
                                     negate=True)  # Need this true for PGD type attacks

    # PGD is used to optimize the above loss.
    pgd_attack_obj = aa.PGD(source_model, normalizer,
                            recoloradv_threat, attack_loss)

    image_list = [f for f in os.listdir(
        args.dataset) if os.path.isfile(os.path.join(args.dataset, f))]

    for img_name in tqdm(image_list):
        # print(i)
        # if i % 100 == 0:
        #     print(i, S, T)
        im_orig = Image.open(os.path.join(
            args.dataset, img_name)).convert("RGB")
        im = pre(im_orig).to(device)
        labels = truth[img_name]

        # We run the attack for 10 iterations at learning rate 0.01.
        adv_inputs = pgd_attack_obj.attack(im.unsqueeze(0), (torch.ones(1) * labels).long().to(device),
                                           num_iterations=10, signed=False, optimizer=optim.Adam,
                                           optimizer_kwargs={'lr': 0.01},
                                           verbose=False).adversarial_tensors()

        with torch.no_grad():
            adv_logits = source_model(normalizer(adv_inputs))

        # if org_logits.argmax(1) == labels and not adv_logits.argmax(1) == labels:
        if not adv_logits.argmax(1) == labels:
            total += 1
            if args.save_fig:
                cv2.imwrite(os.path.join(args.save_path, img_name),
                            tensor2cv2(adv_inputs.squeeze()))

            # save_res(inputs, adv_inputs, files[i])
            # print(S)

            # Transferability
            adv_inputs = normalizer(adv_inputs)
            for i, model in enumerate(model_list):
                pred_trans = model(adv_inputs).detach().cpu().numpy()
                pred_target_lbl_adv = np.argmax(pred_trans, 1)
                if not pred_target_lbl_adv == labels:
                    trans[i] += 1

        # print(total, S, T)

        # import csv

        # with open("./ColorAdv_{}_cifar.csv".format(args.model), "a") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(
        #         [float(100.0 * S / total), float(100.0 * T[0] / S), float(100.0 * T[1] / S), float(100.0 * T[2] / S),
        #          float(100.0 * T[3] / S), float(100.0 * T[4] / S), float(100.0 * T[5] / S), float(100.0 * T[6] / S)])

    success_rate = trans / total * 100
    sr = list(success_rate)
    print(f"total: {total}")
    for n, r in zip(model_name, sr):
        print(f"SR of {n}: {r:.2f}")
    print(str(total)+'_'+'_'.join([str(s) for s in sr]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of SparseFool')
    parser.add_argument('--dataset', default="GMM_val_data", type=str,
                        help='path to input img')
    parser.add_argument('--gt', default="Images_val.csv", type=str,
                        help='ground truth')
    parser.add_argument('--model', type=str, default="vgg16")
    parser.add_argument('--save_path', type=str, help='path to save input img')
    parser.add_argument('--trial', type=int, default=1, help='trial num')
    parser.add_argument('--save_fig', default=False, action='store_true',
                        help='store figure')
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join("Save", f"{args.model}",
                                      args.dataset.split('/')[-1],
                                      f"attack_{args.trial}")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print(f'Save {args.save_path}')

    main(args)
