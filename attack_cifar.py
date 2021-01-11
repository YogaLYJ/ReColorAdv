# EXTERNAL LIBRARIES
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

import torch
import pandas as pd
import torch.optim as optim
assert float(torch.__version__[:3]) >= 0.3

# Add local libraries to the pythonpath
import os
import sys
module_path = os.path.abspath('mister_ed')
if module_path not in sys.path:
    sys.path.append(module_path)

# mister_ed
import recoloradv.mister_ed.loss_functions as lf
import recoloradv.mister_ed.utils.pytorch_utils as utils
import recoloradv.mister_ed.adversarial_perturbations as ap
import recoloradv.mister_ed.adversarial_attacks as aa

# ReColorAdv
import recoloradv.perturbations as pt
import recoloradv.color_transformers as ct
import recoloradv.color_spaces as cs

# CIFAR10
from cifar10_models import *

# Preprocessed
def pre(img):
    trans = transforms.Compose([transforms.ToTensor()])

    return trans(img)

def main(args):
    # Source model
    if args.model == 'resnet50':
        model = resnet50(pretrained=True)
    elif args.model == 'vgg16':
        model = vgg16_bn(pretrained=True)
    elif args.model == 'inceptionv3':
        model = inception_v3(pretrained=True)
    else:
        raise NotImplementedError('{} is not allowed!'.format(args.model))

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Target model
    target_model_1 = resnet50(pretrained=True)
    target_model_2 = resnet18(pretrained=True)
    target_model_3 = vgg16_bn(pretrained=True)
    target_model_4 = densenet121(pretrained=True)
    target_model_5 = googlenet(pretrained=True)
    target_model_6 = mobilenet_v2(pretrained=True)
    target_model_7 = inception_v3(pretrained=True)

    target_model_1.eval()
    target_model_2.eval()
    target_model_3.eval()
    target_model_4.eval()
    target_model_5.eval()
    target_model_6.eval()
    target_model_7.eval()

    target_model_1 = target_model_1.cuda()
    target_model_2 = target_model_2.cuda()
    target_model_3 = target_model_3.cuda()
    target_model_4 = target_model_4.cuda()
    target_model_5 = target_model_5.cuda()
    target_model_6 = target_model_6.cuda()
    target_model_7 = target_model_7.cuda()

    recoloradv_threat = ap.ThreatModel(pt.ReColorAdv, {
        'xform_class': ct.FullSpatial,
        'cspace': cs.RGBColorSpace(), # controls the color space used
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
    adv_loss = lf.CWLossF6(model, normalizer)

    # We also need the smoothness loss.
    smooth_loss = lf.PerturbationNormLoss(lp=2)

    # We combine them with a RegularizedLoss object.
    attack_loss = lf.RegularizedLoss({'adv': adv_loss, 'smooth': smooth_loss},
                                     {'adv': 1.0,      'smooth': 0.05},   # lambda = 0.05
                                     negate=True) # Need this true for PGD type attacks

    # PGD is used to optimize the above loss.
    pgd_attack_obj = aa.PGD(model, normalizer, recoloradv_threat, attack_loss)

    # Images Prepare
    df = pd.read_csv('/media/yoga/DATA/Project/Adversarial_Attack_cifar10/cifar10_{}.csv'.format(args.iter_num))
    # df = pd.read_csv('/data2/YogaData/Adversarial_Attack_ImageNet/selected_list_{}.csv'.format(args.iter_num))
    files = df['filename'].values
    truth = df['groundtruth'].values

    S = 0
    T = np.zeros(7)  # transferability
    total = len(files)

    for i in range(len(files)):
        # print(i)
        if i % 100 == 0:
            print(i, S, T)

        img_path = '/media/yoga/DATA/Project/Adversarial_Attack_cifar10/cifar10_{}/'.format(args.iter_num) + files[i]
        # img_path = '/data2/YogaData/Adversarial_Attack_ImageNet/val_data_{}/'.format(args.iter_num) + files[i]
        examples = Image.open(img_path).convert("RGB")
        examples = pre(examples).unsqueeze(0)

        labels = torch.ones(1) * int(truth[i])
        labels = labels.long()

        if torch.cuda.is_available():
            examples = examples.cuda()
            labels = labels.cuda()

        # We run the attack for 10 iterations at learning rate 0.01.
        adv_inputs = pgd_attack_obj.attack(examples, labels, num_iterations=10, signed=False,
                                             optimizer=optim.Adam, optimizer_kwargs={'lr': 0.01},
                                             verbose=False).adversarial_tensors()

        with torch.no_grad():
            adv_logits = model(normalizer(adv_inputs))

        # if org_logits.argmax(1) == labels and not adv_logits.argmax(1) == labels:
        if not adv_logits.argmax(1) == labels:
            S += 1
            # save_res(inputs, adv_inputs, files[i])
            # print(S)

            # Transferability
            pred_trans = target_model_1(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[0] += 1

            pred_trans = target_model_2(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[1] += 1

            pred_trans = target_model_3(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[2] += 1

            pred_trans = target_model_4(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[3] += 1

            pred_trans = target_model_5(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[4] += 1

            pred_trans = target_model_6(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[5] += 1

            pred_trans = target_model_7(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[6] += 1

        print(total, S, T)

        import csv

        with open("./ColorAdv_{}_cifar.csv".format(args.model), "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [float(100.0 * S / total), float(100.0 * T[0] / S), float(100.0 * T[1] / S), float(100.0 * T[2] / S),
                 float(100.0 * T[3] / S), float(100.0 * T[4] / S), float(100.0 * T[5] / S), float(100.0 * T[6] / S)])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='resnet50, vgg16, inceptionv3', type=str, default='resnet50')
    parser.add_argument('--iter_num', type=int, default='1')
    args = parser.parse_args()

    main(args)

