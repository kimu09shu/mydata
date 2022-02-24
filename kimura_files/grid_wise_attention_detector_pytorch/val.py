import os
import time
import torch
import json
import matplotlib.pyplot as plt
import torch.optim as optim
from copy import deepcopy
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.network import Network
from config import Config
from dataloader.loader import *
from eval_city.eval_script.eval_demo import validate
from utils.nms_wrapper import nms
from utils.functions import parse_det_offset


parser = argparse.ArgumentParser(description='GWA Training')
parser.add_argument('--detection_model', default='retina', type=str, help='use detection( anchor or csp or retina)')
parser.add_argument('--detection_scale', default='single', type=str, help='single or multi')
parser.add_argument('--model_dir', default='./ckpt', type=str, help='weight dir')
parser.add_argument('--model_path', default='Network-100.pth', type=str, help='weight path')
args = parser.parse_args()

config = Config()
config.train_path = './data/citypersons'
config.test_path = './data/citypersons'
config.gpu_ids = [0, 1]
config.onegpu = 1
config.size_train = [512, 1024]
config.size_test = [512, 1024]#(1024, 2048)
config.init_lr = 1e-5
config.num_epochs = 300
config.offset = True
config.val = True
config.val_frequency = 1
config.rois_thresh = 0.3
config.num_classes = 1

# dataset
print('Dataset...')

if config.val:
    testtransform = Compose([ToTensor()])
    testdataset = CityPersons(path=config.train_path, type='val', config=config,
                              transform=testtransform, preloaded=False)
    testloader = DataLoader(testdataset, batch_size=1)

# net
print('Net...')
net = Network(input_size=config.size_test, phase='val', model=args.detection_model, num_classes=config.num_classes).cuda()

if config.teacher:
    print('I found this teacher model is useless, I disable this training option')
    exit(1)
    teacher_dict = net.state_dict()

def val_epochs():

    best_loss = np.Inf
    best_loss_epoch = 0

    best_mr = 100
    best_mr_epoch = 0

    max_epoch = config.num_epochs

    train_loss_list = np.zeros(max_epoch+1)
    val_loss_list = np.zeros(max_epoch+1)
    train_acc_list = np.zeros(max_epoch+1)
    val_acc_list = np.zeros(max_epoch+1)


    for epoch in range(max_epoch):
        print('----------')
        print('Epoch %d begin' % (epoch + 1))
        net.load_state_dict(torch.load('./ckpt/CSPNet-' + str(epoch+1) + '.pth'))
        #net = nn.DataParallel(net, device_ids=config.gpu_ids)

        t1 = time.time()
        '''
        net.eval()

        if config.teacher:
            print('Load teacher params')
            student_dict = net.module.state_dict()
            net.module.load_state_dict(teacher_dict)

        print('Perform validation...')
        res = []
        t3 = time.time()
        batch_loss = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels, gtbox = data
            inputs = inputs.cuda()
            labels = [l.cuda().float() for l in labels]
            gtbox = gtbox.cuda()
            with torch.no_grad():
                # pos, height, offset = net(inputs)
                outputs = net(inputs)
                # outputs, attmap = net(inputs)

                pos, height, offset = outputs[0], outputs[1], outputs[2]

            boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test,
                                     score=0.1, down=4, nms_thresh=0.5)
            if len(boxes) > 0:
                boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                # boxes = np.array(boxes, dtype=int)

                for box in boxes:
                    temp = dict()
                    temp['image_id'] = i + 1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = float(box[4])
                    res.append(temp)

            print('\r%d/%d' % (i + 1, len(testloader))),
            sys.stdout.flush()

            cls_loss, reg_loss, off_loss = criterion(outputs, labels)
            # attloss_1, attloss_2 = attention(attmap, gtbox)
            loss = cls_loss + reg_loss + off_loss  # + attloss_1 + attloss_2
            cls_loss, reg_loss, off_loss, loss = cls_loss.item(), reg_loss.item(), off_loss.item(), loss.item()
            print ("<Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f " % (loss, cls_loss, reg_loss, off_loss))
            # print ("<Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, gam_gts: %.6f, gam_all: %.6f" % (loss, cls_loss, reg_loss, off_loss, attloss_1, attloss_2))
            batch_loss += loss

        print('')

        if config.teacher:
            print('Load back student params')
            net.module.load_state_dict(student_dict)

        with open('./_temp_val.json', 'w') as f:
            json.dump(res, f)

        MRs = validate('./eval_city/val_gt.json', './_temp_val.json')
        t4 = time.time()
        print('Summerize: [Reasonable: %.2f%%], [Large: %.2f%%], [Middle: %.2f%%], [Small: %.2f%%]'
              # print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
              % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))
        val_loss = batch_loss / len(testloader)
        

        print('Validation time used: %.3f' % (t4 - t3))

        cur_mr = MRs[0]
        '''

        if config.val and (epoch + 1) % config.val_frequency == 0: # and epoch + 1 > 10
            cur_mr, val_loss = val(epoch=epoch)
            val_loss_list[epoch+1] = val_loss
            val_acc_list[epoch+1] = cur_mr
            if cur_mr < best_mr:
                best_mr = cur_mr
                best_mr_epoch = epoch + 1
            print('Epoch %d has lowest MR: %.7f' % (best_mr_epoch, best_mr))

        print ("Make graph...")
        plt.plot(range(1, epoch+1), val_loss_list[1:epoch+1], 'b-', label='val_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        #plt.grid()
        plt.savefig("data/loss.png")
        plt.close()
        print ("----  saved loss graph  ----")
        plt.plot(range(1, epoch+1), val_acc_list[1:epoch+1], 'b-', label='val_MR')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('MR')
        # plt.grid()
        plt.savefig("data/acc.png")
        plt.close()
        print ("----  saved acc graph ----")


    '''
    log.close()
    if config.val:
        vallog.close()
    '''


def val(log=None, epoch=None):
    net.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_path)))
    net.eval()

    if config.teacher:
        print('Load teacher params')
        student_dict = net.module.state_dict()
        net.module.load_state_dict(teacher_dict)

    print('Perform validation...')
    res = []
    res_gt = []
    t3 = time.time()
    batch_loss = 0
    count = 0
    for i, data in enumerate(testloader, 0):
        inputs, gtbox, labels = data
        #print(gts)
        inputs = inputs.cuda()
        labels = [l.cuda().float() for l in labels]
        gtbox = gtbox.cuda()
        if args.detection_model == 'anchor':
            with torch.no_grad():
                rois = net(inputs, gtbox, [config.size_test[0], config.size_test[1], config.size_test[0] / 1024])
                inds = (rois[:, :, 4] > config.rois_thresh)
                roi_keep = rois[:, inds[0], :]

                # unscale back
                if roi_keep.dim() == 1:
                    roi_keep = rois

                roi_keep[:, :, 0:4] /= (config.size_train[0] / 1024)

                roi_single = roi_keep[0]

                roi_single =roi_single.cpu().numpy()

                #print(roi_single)
                nms_keep = nms(roi_single, 0.3, usegpu=False, gpu_id=0)
                cls_dets = roi_single[nms_keep, :]

                for box in cls_dets:
                    temp = dict()
                    temp['image_id'] = i + 1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = float(box[4])
                    res.append(temp)

        elif args.detection_model == 'retina':
            with torch.no_grad():
                scores, classification, transformed_anchors = net(inputs, None, [config.size_test[0], config.size_test[1], config.size_test[0] / 1024])
                #print(scores, classification, transformed_anchors)
                idxs = np.where(scores.cpu() > config.rois_thresh)
                count += 1
                print(count)
                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :].cpu().numpy()
                    #x = int(bbox[0]/(config.size_test[0] / 1024))
                    #y = int(bbox[1]/(config.size_test[0] / 1024))
                    #w = int(bbox[2]/(config.size_test[0] / 1024))
                    #h = int(bbox[3]/(config.size_test[0] / 1024))
                    x = int(bbox[0] /(config.size_test[0] / 1024))
                    y = int(bbox[1] /(config.size_test[0] / 1024))
                    w = int(bbox[2] /(config.size_test[0] / 1024))
                    h = int(bbox[3] /(config.size_test[0] / 1024))
                    temp = dict()
                    temp['image_id'] = count
                    temp['category_id'] = classification[j].item() + 1
                    temp['bbox'] = [x,y,w-x,h-y] #bbox.tolist()
                    temp['score'] = float(scores[j].item())
                    res.append(temp)
        elif args.detection_model == 'csp':
            with torch.no_grad():
                outputs = net(inputs)
                pos, height, offset = outputs[0], outputs[1], outputs[2]

            # print (pos.size())
            boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test,
                                     score=0.3, down=4, nms_thresh=0.5)
            if len(boxes) > 0:
                boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                for box in boxes:
                    temp = dict()
                    temp['image_id'] = i + 1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = float(box[4])
                    res.append(temp)
                    # cv2.rectangle(img, (int(box[0] / img_scale), int(box[1] / img_scale)),
                    #              (int(box[2] + box[0] / img_scale), int(box[3] + box[1] / img_scale)), [0, 0, 255], 2)
                # cv2.imwrite('./data/output/detection/' + str(i)+ '.jpg', img)
            print('\r%d/%d' % (i + 1, len(testloader)))
            # print (res)
            sys.stdout.flush()

            # cls_loss, reg_loss, off_loss = criterion(outputs, labels)
            # loss = cls_loss + reg_loss + off_loss
            # cls_loss, reg_loss, off_loss, loss = cls_loss.item(), reg_loss.item(), off_loss.item(), loss.item()
            # print ("<Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f " % (loss, cls_loss, reg_loss, off_loss))
            # batch_loss += loss

        #MRs = validate('./_temp_val_gt.json', './_temp_val.json')


    print('')
    with open('./_temp_val.json', 'w') as f:
        json.dump(res, f)

    '''
    if config.teacher:
        print('Load back student params')
        net.module.load_state_dict(student_dict)

    with open('./_temp_val.json', 'w') as f:
        json.dump(res, f)
    '''

    MRs = validate('./eval_city/val_gt.json', './_temp_val.json')
    t4 = time.time()
    print('Summerize: [Reasonable: %.2f%%], [Large: %.2f%%], [Middle: %.2f%%], [Small: %.2f%%]'
    #print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    val_loss = batch_loss / len(testloader)
    if log is not None:
        log.write("%.7f %.7f %.7f %.7f\n" % tuple(MRs))
        log.write('%d %.7f\n' % (epoch ,val_loss))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs[0]  #, val_loss

if __name__ == '__main__':
    val()
    #val_epochs()
