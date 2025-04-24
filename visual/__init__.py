import argparse
import os, sys
import datetime
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_source = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["source"] = ImageList_idx(txt_source, transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def calculate_gaussian_kl_divergence(m1, m2, v1, v2):
    # m1,m2 指两个高斯分布的均值
    # v1，v2指两个高斯分布的方差
    # return torch.log(v2 / v1) + torch.div(torch.add(torch.square(v1), torch.square(m1 - m2)), 2 * torch.square(v2) ) - 0.5
    return torch.log(v2 / v1) * 0.5 + torch.div(torch.add(v1, torch.square(m1 - m2)), 2 * v2) - 0.5


class ShotHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 or KL 散度 as a loss
    '''

    def __init__(self, module1, module2):
        self.hook = module1.register_forward_hook(self.hook_fn)
        self.mean_orignal = module2.running_mean
        self.var_orignal = module2.running_var

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        if isinstance(module, nn.BatchNorm2d):
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        if isinstance(module, nn.BatchNorm1d):
            mean = input[0].mean([0])
            var = input[0].permute(1, 0).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # KL divergence loss
        # p = torch.cat((self.mean_orignal, self.var_orignal), 0)
        # q = torch.cat((mean, var), 0)
        # x = F.log_softmax(q, 0)
        # y = F.softmax(p, 0)
        # r_feature = F.kl_div(x, y)
        # print(mean.size())

        klc = 0.0
        for i in range(mean.size()[0]):
            klc += calculate_gaussian_kl_divergence(self.mean_orignal[i], mean[i], self.var_orignal[i], var[i])
        r_feature = klc / mean.size()[0]

        # # l2 norm loss
        # r_feature = torch.norm(self.var_orignal.data.type(var.type()) - var, 2) + torch.norm(
        #     self.mean_orignal.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        netF_orignal = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netB_orignal = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                           bottleneck_dim=args.bottleneck).cuda()

    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF_orignal.load_state_dict(torch.load(modelpath))
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    netB_orignal.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netF_orignal.named_parameters():
        v.requires_grad = False
    for k, v in netB_orignal.named_parameters():
        v.requires_grad = False

    tSNE(dset_loaders, netF, netB, netF_orignal, netB_orignal, netC, args)
    acc_s, _ = cal_acc(dset_loaders['source'], netF_orignal, netB_orignal, netC, False)
    print(acc_s)
    acc_st, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
    print(acc_st)
    return netF, netB, netC

def tSNE(loader, netF, netB, netF_orignal, netB_orignal, netC, args):
    netF.eval()
    netB.eval()
    all_fea, all_label = obtain(loader['test'], netF, netB, netC, args)
    sour_all_fea, sour_all_label = obtain(loader["source"], netF_orignal, netB_orignal, netC, args)
    netF.train()
    netB.train()
    tsne = TSNE(n_components=2, random_state=33)
    result = tsne.fit_transform(all_fea)
    sour_result = tsne.fit_transform(sour_all_fea)

    plt.figure()


    # sour_label0 = []
    # sour_label1 = []
    # sour_label2 = []
    # sour_label3 = []
    # sour_label4 = []
    # for i in range(len(sour_all_fea)):
    #     if sour_all_label[i] == 0:
    #         sour_label0.append(i)
    #     if sour_all_label[i] == 1:
    #         sour_label1.append(i)
    #     if sour_all_label[i] == 2:
    #         sour_label2.append(i)
    #     if sour_all_label[i] == 3:
    #         sour_label3.append(i)
    #     if sour_all_label[i] == 5:
    #         sour_label4.append(i)
    # sour_s1 = sour_result[sour_label0, :]
    # sour_s2 = sour_result[sour_label1, :]
    # sour_s3 = sour_result[sour_label2, :]
    # sour_s4 = sour_result[sour_label3, :]
    # sour_s5 = sour_result[sour_label4, :]
    # sour_t1 = plt.scatter(sour_s1[:, 0], sour_s1[:, 1], marker='o', c='coral', s=10)  # marker:点符号 c:点颜色 s:点大小
    # sour_t2 = plt.scatter(sour_s2[:, 0], sour_s2[:, 1], marker='o', c='royalblue', s=10)
    # sour_t3 = plt.scatter(sour_s3[:, 0], sour_s3[:, 1], marker='o', c='violet', s=10)
    # sour_t4 = plt.scatter(sour_s4[:, 0], sour_s4[:, 1], marker='o', c='paleturquoise', s=10)
    # sour_t5 = plt.scatter(sour_s5[:, 0], sour_s5[:, 1], marker='o', c='grey', s=10)

    label0 = []
    label1 = []
    label2 = []
    label3 = []
    label4 = []
    for i in range(len(all_fea)):
        if all_label[i] == 0:
            label0.append(i)
        if all_label[i] == 1:
            label1.append(i)
        if all_label[i] == 2:
            label2.append(i)
        if all_label[i] == 3:
            label3.append(i)
        if all_label[i] == 5:
            label4.append(i)
    s1 = result[label0, :]
    s2 = result[label1, :]
    s3 = result[label2, :]
    s4 = result[label3, :]
    s5 = result[label4, :]
    t1 = plt.scatter(s1[:, 0], s1[:, 1], marker='*', c='r', s=10)  # marker:点符号 c:点颜色 s:点大小
    t2 = plt.scatter(s2[:, 0], s2[:, 1], marker='*', c='b', s=10)
    t3 = plt.scatter(s3[:, 0], s3[:, 1], marker='*', c='m', s=10)
    t4 = plt.scatter(s4[:, 0], s4[:, 1], marker='*', c='c', s=10)
    t5 = plt.scatter(s5[:, 0], s5[:, 1], marker='*', c='k', s=10)

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend((t1, t2, t3, t4, t5), ('pos', 'neg'))
    # plt.show()

    plt.savefig(args.name+'source.png')

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    return all_fea, all_label

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return pred_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--plabel', action='store_true')
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--bn_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = 'data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][
            0].upper() + datetime.datetime.now().strftime("%m-%d_%H:%M"))
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'cls_par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
