import torch
from torch import optim
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import argparse


# mister_ed
import recoloradv.mister_ed.loss_functions as lf
import recoloradv.mister_ed.utils.pytorch_utils as utils
import recoloradv.mister_ed.adversarial_perturbations as ap
import recoloradv.mister_ed.adversarial_attacks as aa

# ReColorAdv
import recoloradv.perturbations as pt
import recoloradv.color_transformers as ct
import recoloradv.color_spaces as cs


# Preprocessed
def pre(img):
    trans = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),])

    return trans(img)


# Save_PNG/JPEG
import cv2
def save_res(org_img, pert_img, filename):
    res_pert_png = '/media/yoga/DATA/Project/GAA_Quality/Color'
    res_org_png = '/media/yoga/DATA/Project/GAA_Quality/Original'

    saved_pert_png = '{}/{}'.format(res_pert_png, filename[:-5]+'.png')
    saved_org_png = '{}/{}'.format(res_org_png, filename[:-5]+'.png')

    img = pert_img.data.cpu().numpy()[0]
    img = img[::-1, :, :]

    # pert = (pert_img - org_img).data.cpu().numpy()[0]
    # pert = pert[::-1, :, :]

    img_o = org_img.data.cpu().numpy()[0]
    img_o = img_o[::-1, :, :]

    # final_img = np.concatenate([img_o, img, pert], axis=2)
    # cv2.imwrite(saved_file_png, (final_img.transpose(1, 2, 0) * 255).astype('uint8'))
    cv2.imwrite(saved_pert_png, (img.transpose(1, 2, 0) * 255).astype('uint8'))
    cv2.imwrite(saved_org_png, (img_o.transpose(1, 2, 0) * 255).astype('uint8'))

def main(args):
    # Source model
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.model == 'inceptionv3':
        model = models.inception_v3(pretrained=True)
    else:
        raise NotImplementedError('{} is not allowed!'.format(args.model))

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Attack Model
    normalizer = utils.DifferentiableNormalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    cw_loss = lf.CWLossF6(model, normalizer, kappa=float('inf'))
    # cw_loss = lf.CWLossF6(model, normalizer)
    perturbation_loss = lf.PerturbationNormLoss(lp=2)
    adv_loss = lf.RegularizedLoss(
        {'cw': cw_loss, 'pert': perturbation_loss},
        {'cw': 1.0, 'pert': 0.05},
        negate=True,
    )

    # ReColorAdv
    pgd_attack = aa.PGD(
        model,
        normalizer,
        ap.ThreatModel(pt.ReColorAdv, {
            'xform_class': ct.FullSpatial,
            'cspace': cs.RGBColorSpace(),
            'lp_style': 'inf',
            'lp_bound': 0.047,
            'xform_params': {
                'resolution_x': 25,
                'resolution_y': 25,
                'resolution_z': 25,
            },
            'use_smooth_loss': True,
        }),
        adv_loss,
    )

    # Target model
    target_model_1 = models.resnet50(pretrained=True)
    target_model_2 = models.resnet18(pretrained=True)
    target_model_3 = models.vgg16(pretrained=True)
    target_model_4 = models.densenet121(pretrained=True)
    target_model_5 = models.googlenet(pretrained=True)
    target_model_6 = models.mobilenet_v2(pretrained=True)
    target_model_7 = models.inception_v3(pretrained=True)
    target_model_8 = models.squeezenet1_0(pretrained=True)
    target_model_9 = models.alexnet(pretrained=True)

    target_model_1.eval()
    target_model_2.eval()
    target_model_3.eval()
    target_model_4.eval()
    target_model_5.eval()
    target_model_6.eval()
    target_model_7.eval()
    target_model_8.eval()
    target_model_9.eval()

    target_model_1 = target_model_1.cuda()
    target_model_2 = target_model_2.cuda()
    target_model_3 = target_model_3.cuda()
    target_model_4 = target_model_4.cuda()
    target_model_5 = target_model_5.cuda()
    target_model_6 = target_model_6.cuda()
    target_model_7 = target_model_7.cuda()
    target_model_8 = target_model_8.cuda()
    target_model_9 = target_model_9.cuda()

    # Attack images in Images_tiny.csv
    import pandas as pd
    # df = pd.read_csv('/media/yoga/DATA/Project/Adversarial Attack/GMM_2/Images_val.csv')
    df = pd.read_csv('/data2/YogaData/Adversarial_Attack_ImageNet/selected_list_{}.csv'.format(args.iter_num))
    files = df['filename'].values
    truth = df['groundTruth'].values

    import numpy as np
    S = 0
    P = np.zeros(2) # png compression/ jpeg compression
    T = np.zeros(9) # transferability
    total = len(files)

    import csv
    # with open('Color.csv', 'w') as f:
    #     w = csv.writer(f)
    #     w.writerow(['filename', 'distance to boundary', 'VGG16', 'VGG160', 'SN', 'SN0'])

    for i in range(len(files)):
        print(i)

        # print(i)
        # img_path = '/media/yoga/DATA/Project/Adversarial Attack/GMM_val_data/' + files[i]
        # img_path = '/media/yoga/DATA/Project/Adversarial_Attack_ImageNet/val_data_{}/'.format(args.iter_num) + files[i]
        img_path = '/data2/YogaData/Adversarial_Attack_ImageNet/val_data_{}/'.format(args.iter_num) + files[i]
        inputs = Image.open(img_path).convert("RGB")
        inputs = pre(inputs).unsqueeze(0)

        labels = torch.ones(1) * int(truth[i])
        labels = labels.long()

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        adv_inputs = pgd_attack.attack(
            inputs,
            labels,
            optimizer=optim.Adam,
            optimizer_kwargs={'lr': 0.01},
            signed=False,
            verbose=False,
            num_iterations=100,
        ).adversarial_tensors()


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

            pred_trans = target_model_8(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[7] += 1

            pred_trans = target_model_9(normalizer(adv_inputs))
            _, cls = pred_trans.data.max(1)
            if not int(cls.cpu()) == truth[i]:
                T[8] += 1

    print(total, S, T, P)

    import csv

    with open("./ColorAdv_{}_float.csv".format(args.model),"a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [float(100.0 * S / total), float(100.0 * T[0] / S), float(100.0 * T[1] / S), float(100.0 * T[2] / S),
             float(100.0 * T[3] / S), float(100.0 * T[4] / S), float(100.0 * T[5] / S), float(100.0 * T[6] / S),
             float(100.0 * T[7] / S), float(100.0 * T[8] / S)])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='resnet50, vgg16, inceptionv3', type=str, default='resnet50')
    parser.add_argument('--iter_num', type=int, default='1')
    args = parser.parse_args()

    main(args)
