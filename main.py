import argparse

import numpy as np
import pickle
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import logging
import datetime
import matplotlib as mpl
mpl.use('Agg')
import sys
import os
import time
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.optim

from data_loader import *
from models.model import MLCMF
from models.loss import Reg_Loss, GIoU_Loss
from utils.utils import *
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume

def main():
    parser = argparse.ArgumentParser(description='MLCMF training and evaluation scrip')
    parser.add_argument('--size', default=640, type=int, help='image size')
    parser.add_argument('--images_path', type=str, default='RSVGD\\JPEGImages',
                        help='path to dataset splits data folder')
    parser.add_argument('--anno_path', type=str, default='RSVGD\\Annotations',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--time', default=40, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--gpu', default='0,1', help='gpu id')
    parser.add_argument('--workers', default=0, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=150, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_dec', default=0.1, type=float, help='decline of learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',  #
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--print_freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='train')
    # parser.add_argument('--test', dest='test', default=True, action='store_true', help='test')
    parser.add_argument('--tunebert', dest='tunebert', default=True, action='store_true', help='if tunebert')

    # * MLCM
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400 + 40 + 1, type=int,
                        help="Number of query slots in VLFusion")
    parser.add_argument('--pre_norm', action='store_true')

    global args, anchors_full
    args = parser.parse_args()

    print('----------------------------------------------------------------------')
    print('模型参数：', args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # fix random seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 3)

    # save logs
    if args.savename == 'default':
        args.savename = 'MLCMF_batch%d_epoch%d_lr%d_seed%d' % (args.batch_size, args.nb_epoch,args.lr, args.seed)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s" % args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = RSVGDataset(images_path=args.images_path,
                         anno_path=args.anno_path,
                         split = 'train',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)

    val_dataset = RSVGDataset(images_path=args.images_path,
                         anno_path=args.anno_path,
                         split = 'val',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)

    test_dataset = RSVGDataset(images_path=args.images_path,
                         anno_path=args.anno_path,
                         split = 'test',
                         testmode=True,
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)
    print('trainset:', len(train_dataset), 'validationset:', len(val_dataset), 'testset:', len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=0)

    # Model
    model = MLCMF(bert_model=args.bert_model, tunebert=args.tunebert, args=args)
    model = torch.nn.DataParallel(model).cuda()

    # load pretrain model
    if args.pretrain:
        model = load_pretrain(model, args, logging)
    if args.resume:
        model = load_resume(model, args, logging)

    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d' % int(sum([param.nelement() for param in model.parameters()])))

    if args.tunebert:
        visu_param = model.module.visumodel.parameters()
        text_param = model.module.textmodel.parameters()
        rest_param = [param for param in model.parameters() if ((param not in visu_param) and (param not in text_param))]
        visu_param = list(model.module.visumodel.parameters())
        text_param = list(model.module.textmodel.parameters())
        sum_visu = sum([param.nelement() for param in visu_param])
        sum_text = sum([param.nelement() for param in text_param])
        sum_fusion = sum([param.nelement() for param in rest_param])
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)
    else:
        visu_param = model.module.visumodel.parameters()
        rest_param = [param for param in model.parameters() if param not in visu_param]
        visu_param = list(model.module.visumodel.parameters())
        sum_visu = sum([param.nelement() for param in visu_param])
        sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
        sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    # optimizer
    if args.tunebert:
        optimizer = torch.optim.AdamW([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr/10.},
                {'params': text_param, 'lr': args.lr/10.}], lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = torch.optim.AdamW([{'params': rest_param},
                {'params': visu_param}],lr=args.lr, weight_decay=0.0001)

    # training and testing
    best_accu = -float('Inf')
    trainResultList, vaildResultList = [],[]
    if args.test:
        # testing
        _ = Mytest_epoch(test_loader, model)
    else:
        for epoch in range(args.nb_epoch):
            adjust_learning_rate(args, optimizer, epoch)
            # training
            trainResult= train_epoch(train_loader, model, optimizer, epoch)
            # validation
            vaildResult = validate_epoch(val_loader, model)
            # remember best accu and save checkpoint
            acc_new = vaildResult[0]
            is_best = acc_new >= best_accu
            best_accu = max(acc_new, best_accu)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': acc_new,
                'optimizer': optimizer.state_dict(),
            }, is_best, args, filename=args.savename)
        print('\nBest Accu: %f\n' % best_accu)
        logging.info('\nBest Accu: %f\n' % best_accu)


def train_epoch(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    l1_losses = AverageMeter()
    GIoU_losses = AverageMeter()
    acc5 = AverageMeter()
    acc6 = AverageMeter()
    acc7 = AverageMeter()
    acc8 = AverageMeter()
    acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter()
    union_area = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks[:, :, :, 0] == 255
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        gt_bbox = gt_bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        gt_bbox = Variable(gt_bbox)
        gt_bbox = torch.clamp(gt_bbox, min=0, max=args.size - 1)

        pred_bbox = model(image, masks, word_id, word_mask)

        # compute loss
        loss = 0.
        GIoU_loss = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)
        loss += GIoU_loss
        gt_bbox_ = xyxy2xywh(gt_bbox)
        l1_loss = Reg_Loss(pred_bbox, gt_bbox_ / (args.size - 1))
        loss += l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        l1_losses.update(l1_loss.item(), imgs.size(0))
        GIoU_losses.update(GIoU_loss.item(), imgs.size(0))

        # box iou
        pred_bbox = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2), pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)],dim=1)
        pred_bbox = pred_bbox * (args.size - 1)
        iou, interArea, unionArea = bbox_iou(pred_bbox.data.cpu(), gt_bbox.data.cpu(), x1y1x2y2=True)
        cumInterArea = np.sum(np.array(interArea.data.cpu().numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.cpu().numpy()))
        # accuracy
        accu5 = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size
        accu6 = np.sum(np.array((iou.data.cpu().numpy() > 0.6), dtype=float)) / args.batch_size
        accu7 = np.sum(np.array((iou.data.cpu().numpy() > 0.7), dtype=float)) / args.batch_size
        accu8 = np.sum(np.array((iou.data.cpu().numpy() > 0.8), dtype=float)) / args.batch_size
        accu9 = np.sum(np.array((iou.data.cpu().numpy() > 0.9), dtype=float)) / args.batch_size

        # 7 metrics
        meanIoU.update(torch.mean(iou).item(), imgs.size(0))
        inter_area.update(cumInterArea)
        union_area.update(cumUnionArea)
        acc5.update(accu5, imgs.size(0))
        acc6.update(accu6, imgs.size(0))
        acc7.update(accu7, imgs.size(0))
        acc8.update(accu8, imgs.size(0))
        acc9.update(accu9, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'acc@0.5: {acc5.avg:.4f}\t' \
                        'acc@0.6: {acc6.avg:.4f}\t'\
                        'acc@0.7: {acc7.avg:.4f}\t'\
                        'acc@0.8: {acc8.avg:.4f}\t'\
                        'acc@0.9: {acc9.avg:.4f}\t'\
                        'meanIoU: {meanIoU.avg:.4f}\t'\
                        'cumuIoU: {cumuIoU:.4f}\t' \
                        'Loss: {loss.avg:.4f}\t'\
                        'L1_Loss: {l1_loss.avg:.4f}\t'\
                        'GIoU_Loss: {GIoU_loss.avg:.4f}\t'\
                        'vis_lr {vis_lr:.8f}\t'\
                        'lang_lr {lang_lr:.8f}\t'\
                .format( \
                epoch, batch_idx, len(train_loader),\
                acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9,\
                meanIoU=meanIoU,cumuIoU=inter_area.sum/union_area.sum,\
                loss=losses,l1_loss=l1_losses,GIoU_loss=GIoU_losses,\
                vis_lr=optimizer.param_groups[0]['lr'], lang_lr=optimizer.param_groups[2]['lr'])
            print(print_str)
            logging.info(print_str)
    return acc5.avg,acc6.avg,acc7.avg,acc8.avg,acc9.avg,meanIoU.avg,inter_area.sum/union_area.sum, losses.avg


def validate_epoch(val_loader, model, mode='val'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    l1_losses = AverageMeter()
    GIoU_losses = AverageMeter()
    acc5 = AverageMeter()
    acc6 = AverageMeter()
    acc7 = AverageMeter()
    acc8 = AverageMeter()
    acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter()
    union_area = AverageMeter()

    model.eval()
    end = time.time()
    print(datetime.datetime.now())

    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks[:, :, :, 0] == 255
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        with torch.no_grad():
            pred_bbox = model(image, masks, word_id, word_mask)
        gt_bbox = bbox

        # compute loss
        loss = 0.
        GIoU_loss = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)
        loss += GIoU_loss
        gt_bbox_ = xyxy2xywh(gt_bbox)
        l1_loss = Reg_Loss(pred_bbox, gt_bbox_ / (args.size - 1))
        loss += l1_loss

        losses.update(loss.item(), imgs.size(0))
        l1_losses.update(l1_loss.item(), imgs.size(0))
        GIoU_losses.update(GIoU_loss.item(), imgs.size(0))

        # box iou
        pred_bbox = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2), pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)],dim=1)
        pred_bbox = pred_bbox * (args.size - 1)
        iou, interArea, unionArea = bbox_iou(pred_bbox.data.cpu(), gt_bbox.data.cpu(), x1y1x2y2=True)
        cumInterArea = np.sum(np.array(interArea.data.cpu().numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.cpu().numpy()))
        # accuracy
        accu5 = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size
        accu6 = np.sum(np.array((iou.data.cpu().numpy() > 0.6), dtype=float)) / args.batch_size
        accu7 = np.sum(np.array((iou.data.cpu().numpy() > 0.7), dtype=float)) / args.batch_size
        accu8 = np.sum(np.array((iou.data.cpu().numpy() > 0.8), dtype=float)) / args.batch_size
        accu9 = np.sum(np.array((iou.data.cpu().numpy() > 0.9), dtype=float)) / args.batch_size

        # 7 metrics
        meanIoU.update(torch.mean(iou).item(), imgs.size(0))
        inter_area.update(cumInterArea)
        union_area.update(cumUnionArea)
        acc5.update(accu5, imgs.size(0))
        acc6.update(accu6, imgs.size(0))
        acc7.update(accu7, imgs.size(0))
        acc8.update(accu8, imgs.size(0))
        acc9.update(accu9, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.avg:.3f}\t' \
                        'acc@0.5: {acc5.avg:.4f}\t' \
                        'acc@0.6: {acc6.avg:.4f}\t' \
                        'acc@0.7: {acc7.avg:.4f}\t' \
                        'acc@0.8: {acc8.avg:.4f}\t' \
                        'acc@0.9: {acc9.avg:.4f}\t' \
                        'meanIoU: {meanIoU.avg:.4f}\t' \
                        'cumuIoU: {cumuIoU:.4f}\t' \
                        'Loss: {loss.avg:.4f}\t' \
                .format( \
                batch_idx, len(val_loader), batch_time=batch_time,\
                acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9,\
                meanIoU=meanIoU,cumuIoU=inter_area.sum/union_area.sum, loss=losses)
            print(print_str)
            logging.info(print_str)
    final_str = 'acc@0.5: {acc5.avg:.4f}\t' 'acc@0.6: {acc6.avg:.4f}\t' 'acc@0.7: {acc7.avg:.4f}\t' \
                'acc@0.8: {acc8.avg:.4f}\t' 'acc@0.9: {acc9.avg:.4f}\t' \
                'meanIoU: {meanIoU.avg:.4f}\t' 'cumuIoU: {cumuIoU:.4f}\t' \
        .format(acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9, \
                meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
    print(final_str)
    logging.info(final_str)
    return acc5.avg,acc6.avg,acc7.avg,acc8.avg,acc9.avg,meanIoU.avg,inter_area.sum/union_area.sum, losses.avg


def Mytest_epoch(test_loader, model, mode='test'):
    batch_time = AverageMeter()
    acc5 = AverageMeter()
    acc6 = AverageMeter()
    acc7 = AverageMeter()
    acc8 = AverageMeter()
    acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter()
    union_area = AverageMeter()

    model.eval()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, bbox, ratio, dw, dh, im_id, phrase) in enumerate(test_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks[:, :, :, 0] == 255
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        with torch.no_grad():
            pred_bbox = model(image, masks, word_id, word_mask)
        pred_bbox = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2), pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)],dim=1)
        pred_bbox = pred_bbox * (args.size - 1)

        pred_bbox = pred_bbox.data.cpu()
        target_bbox = bbox.data.cpu()
        pred_bbox[:, 0], pred_bbox[:, 2] = (pred_bbox[:, 0] - dw) / ratio, (pred_bbox[:, 2] - dw) / ratio
        pred_bbox[:, 1], pred_bbox[:, 3] = (pred_bbox[:, 1] - dh) / ratio, (pred_bbox[:, 3] - dh) / ratio
        target_bbox[:, 0], target_bbox[:, 2] = (target_bbox[:, 0] - dw) / ratio, (target_bbox[:, 2] - dw) / ratio
        target_bbox[:, 1], target_bbox[:, 3] = (target_bbox[:, 1] - dh) / ratio, (target_bbox[:, 3] - dh) / ratio

        # convert pred, gt box to original scale with meta-info
        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = imgs[0, :, top:bottom, left:right].data.cpu().numpy().transpose(1, 2, 0)

        ratio = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
        # also revert image for visualization
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = Variable(torch.from_numpy(img_np.transpose(2, 0, 1)).cuda().unsqueeze(0))

        pred_bbox[:, :2], pred_bbox[:, 2], pred_bbox[:, 3] = \
            torch.clamp(pred_bbox[:, :2], min=0), torch.clamp(pred_bbox[:, 2], max=img_np.shape[3]), torch.clamp(
                pred_bbox[:, 3], max=img_np.shape[2])
        target_bbox[:, :2], target_bbox[:, 2], target_bbox[:, 3] = \
            torch.clamp(target_bbox[:, :2], min=0), torch.clamp(target_bbox[:, 2], max=img_np.shape[3]), torch.clamp(
                target_bbox[:, 3], max=img_np.shape[2])

        # box iou
        iou, interArea, unionArea = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
        cumInterArea = np.sum(np.array(interArea.data.cpu().numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.cpu().numpy()))
        # accuracy
        accu5 = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / 1
        accu6 = np.sum(np.array((iou.data.cpu().numpy() > 0.6), dtype=float)) / 1
        accu7 = np.sum(np.array((iou.data.cpu().numpy() > 0.7), dtype=float)) / 1
        accu8 = np.sum(np.array((iou.data.cpu().numpy() > 0.8), dtype=float)) / 1
        accu9 = np.sum(np.array((iou.data.cpu().numpy() > 0.9), dtype=float)) / 1

        # 7 metrics
        meanIoU.update(torch.mean(iou).item(), imgs.size(0))
        inter_area.update(cumInterArea)
        union_area.update(cumUnionArea)
        acc5.update(accu5, imgs.size(0))
        acc6.update(accu6, imgs.size(0))
        acc7.update(accu7, imgs.size(0))
        acc8.update(accu8, imgs.size(0))
        acc9.update(accu9, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.avg:.3f}\t' \
                        'acc@0.5: {acc5.avg:.4f}\t' \
                        'acc@0.6: {acc6.avg:.4f}\t' \
                        'acc@0.7: {acc7.avg:.4f}\t' \
                        'acc@0.8: {acc8.avg:.4f}\t' \
                        'acc@0.9: {acc9.avg:.4f}\t' \
                        'meanIoU: {meanIoU.avg:.4f}\t' \
                        'cumuIoU: {cumuIoU:.4f}\t' \
                .format( \
                batch_idx, len(test_loader), batch_time=batch_time,\
                acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9,\
                meanIoU=meanIoU,cumuIoU=inter_area.sum/union_area.sum)
            print(print_str)
            logging.info(print_str)
    final_str = 'acc@0.5: {acc5.avg:.4f}\t' 'acc@0.6: {acc6.avg:.4f}\t' 'acc@0.7: {acc7.avg:.4f}\t' \
                'acc@0.8: {acc8.avg:.4f}\t' 'acc@0.9: {acc9.avg:.4f}\t' \
                'meanIoU: {meanIoU.avg:.4f}\t' 'cumuIoU: {cumuIoU:.4f}\t' \
        .format(acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9, \
                meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
    print(final_str)
    logging.info(final_str)

if __name__ == "__main__":
    main()