# EXTERNAL LIBRARIES
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
assert float(torch.__version__[:3]) >= 0.3

# Add local libraries to the pythonpath
import os
import sys
module_path = os.path.abspath('mister_ed')
if module_path not in sys.path:
    sys.path.append(module_path)

# mister_ed
# import config
# import prebuilt_loss_functions as plf
# import utils.image_utils as img_utils
# import cifar10.cifar_resnets as cifar_resnets
# import adversarial_training as advtrain
# import adversarial_evaluation as adveval
# import utils.checkpoints as checkpoints
# import spatial_transformers as st
import loss_functions as lf
import utils.pytorch_utils as utils
import cifar10.cifar_loader as cifar_loader
import adversarial_perturbations as ap
import adversarial_attacks as aa
import cv2
import torchvision.transforms as T

# ReColorAdv
import perturbations as pt
import color_transformers as ct
import color_spaces as cs
import norms

# Quick check to ensure cifar 10 data and pretrained classifiers are loaded
cifar_valset = cifar_loader.load_cifar_data('val')
model, normalizer = cifar_loader.load_pretrained_cifar_resnet(flavor=32, return_normalizer=True)

cifar_valset = cifar_loader.load_cifar_data('val', batch_size=500, shuffle=False)
examples, labels = next(iter(cifar_valset))

model, normalizer = cifar_loader.load_pretrained_cifar_resnet(flavor=32, return_normalizer=True)
target_model_1, normalizer = cifar_loader.load_pretrained_cifar_resnet(flavor=44, return_normalizer=True)
target_model_2, normalizer = cifar_loader.load_pretrained_cifar_resnet(flavor=56, return_normalizer=True)
target_model_3, normalizer = cifar_loader.load_pretrained_cifar_resnet(flavor=110, return_normalizer=True)

torch.backends.cudnn.deterministic = True

if utils.use_gpu():
    examples = examples.cuda()
    labels = labels.cuda()
    model.cuda()
    target_model_1.cuda()
    target_model_2.cuda()
    target_model_3.cuda()

# img_utils.show_images(examples)

# This threat model defines the regularization parameters of the attack.
# recoloradv_threat = ap.ThreatModel(pt.ReColorAdv, {
#     'xform_class': ct.FullSpatial,
#     'cspace': cs.CIELUVColorSpace(), # controls the color space used
#     'lp_style': 'inf',
#     'lp_bound': [0.06, 0.06, 0.06],  # [epsilon_1, epsilon_2, epsilon_3]
#     'xform_params': {
#       'resolution_x': 16,            # R_1
#       'resolution_y': 32,            # R_2
#       'resolution_z': 32,            # R_3
#     },
#     'use_smooth_loss': True,
# })
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

# We run the attack for 10 iterations at learning rate 0.01.
perturbation = pgd_attack_obj.attack(examples, labels, num_iterations=10, signed=False,
                                     optimizer=optim.Adam, optimizer_kwargs={'lr': 0.01},
                                     verbose=True)

# Now, we can collect the successful adversarial examples and display them.
successful_advs, successful_origs, idxs = perturbation.collect_successful(model, normalizer,labels)
# successful_diffs = ((successful_advs - successful_origs) * 3 + 0.5).clamp(0, 1)
# img_utils.show_images([successful_origs, successful_advs, successful_diffs])
print(len(successful_advs))

num = 0
j = 0

label = torch.index_select(labels, 0, idxs)

# ResNet 44
new_out = torch.max(target_model_1(normalizer(successful_advs)), 1)[1]
adv_idx_bytes = new_out != label
print(adv_idx_bytes.sum())

# ResNet 56
new_out = torch.max(target_model_2(normalizer(successful_advs)), 1)[1]
adv_idx_bytes = new_out != label
print(adv_idx_bytes.sum())

# ResNet 110
new_out = torch.max(target_model_3(normalizer(successful_advs)), 1)[1]
adv_idx_bytes = new_out != label
print(adv_idx_bytes.sum())

# import csv
# with open('Color.csv', 'w') as f:
#     w = csv.writer(f)
#     w.writerow(['Num', 'point to point', 'distance to boundary', '44', '44-0', '56', '56-0', '110', '110-0'])

# for i in range(100):
#
#     from boundary_to_boundary import boundary_to_boundary
#
#     p2p, p2b1, b2b1 = boundary_to_boundary(successful_origs[i].unsqueeze(0), successful_advs[i].unsqueeze(0), model, target_model_1, normalizer)
#     if b2b1 < 0:
#         temp1 = 0
#     else:
#         temp1 = b2b1
#
#     p2p, p2b2, b2b2 = boundary_to_boundary(successful_origs[i].unsqueeze(0), successful_advs[i].unsqueeze(0), model, target_model_2, normalizer)
#     if b2b2 < 0:
#         temp2 = 0
#     else:
#         temp2 = b2b2
#
#     p2p, p2b3, b2b3 = boundary_to_boundary(successful_origs[i].unsqueeze(0), successful_advs[i].unsqueeze(0), model, target_model_3, normalizer)
#     if b2b3 < 0:
#         temp3 = 0
#     else:
#         temp3 = b2b3
#
#     with open('Color.csv', 'a') as f:
#         w = csv.writer(f)
#         w.writerow([str(i), float(p2p), float(p2b1), float(b2b1), float(temp1), float(b2b2), float(temp2), float(b2b3), float(temp3)])

