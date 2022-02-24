import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import torch
import cv2
import json
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network import CSPNet, CSPNet_mod
from config import Config
from dataloader.loader import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate

parser = argparse.ArgumentParser(description='CSP Training')
parser.add_argument('--ABN_model', default='LAP', type=str, help='use ABN model ( None or GAM or LAP )')
parser.add_argument('--LAP_type', default='sliding_window', type=str, help='use LAP type ( sliding_window or fixed area )')
parser.add_argument('--model_dir', default='./ckpt', type=str, help='weight dir')
parser.add_argument('--model_path', default='CSPNet-100.pth', type=str, help='weight path')
args = parser.parse_args()


config = Config()
config.test_path = './data/citypersons'
config.gpu_ids = [3]
config.onegpu = 1
config.size_test = (1024, 2048)
config.init_lr = 1e-5
config.num_epochs = 75
config.offset = True
config.val_frequency = 1

# dataset
print('Dataset...')

testtransform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
testdataset = CityPersons(path=config.train_path, type='val', config=config,
                              transform=testtransform, preloaded=True)
testloader = DataLoader(testdataset, batch_size=1)

# net
print('Net...')
net = CSPNet(phase='val', model=args.ABN_model, lap_type=args.LAP_type).cuda()

# position
center = cls_pos().cuda()
height = reg_pos().cuda()
offset = offset_pos().cuda()
attention = AttLoss()

if not os.path.exists('./data/output'):
    os.mkdir('./data/output')
if not os.path.exists('./data/output/attmap'):
    os.mkdir('./data/output/attmap')
if not os.path.exists('./data/output/detection'):
    os.mkdir('./data/output/detection')

if config.teacher:
    print('I found this teacher model is useless, I disable this training option')
    exit(1)
    teacher_dict = net.state_dict()

#config.print_conf()

def criterion(output, label):
    cls_loss = center(output[0], label[0])
    reg_loss = height(output[1], label[1])
    off_loss = offset(output[2], label[2])
    return cls_loss, reg_loss, off_loss

def val_epochs():
    best_mr = 100
    best_mr_epoch = 0

    max_epoch = config.num_epochs

    val_loss_list = np.zeros(max_epoch+1)
    val_acc_list = np.zeros(max_epoch+1)


    for epoch in range(max_epoch):
        print('----------')
        print('Epoch %d begin' % (epoch + 1))
        net.load_state_dict(torch.load(os.path.join(args.model_dir, 'CSPNet-' + str(epoch+1) + '.pth')))

        t1 = time.time()

        if (epoch + 1) % config.val_frequency == 0: # and epoch + 1 > 10
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

def val(log=None, epoch=None):
    net.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_path)))
    net.eval()

    if config.teacher:
        print('Load teacher params')
        student_dict = net.module.state_dict()
        net.module.load_state_dict(teacher_dict)

    print('Perform validation...')
    res = []
    t3 = time.time()
    batch_loss = 0
    if args.ABN_model == 'GAM':
        for i, data in enumerate(testloader, 0):
            inputs, labels, gtbox = data
            #img = np.array(inputs[0])
            #img_scale = inputs.shape[1] / config.size_test[0]
            inputs = inputs.cuda()
            labels = [l.cuda().float() for l in labels]
            gtbox = gtbox.cuda()
            with torch.no_grad():
                outputs, attmap = net(inputs)
                pos, height, offset = outputs[0], outputs[1], outputs[2]

            boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.1, down=4, nms_thresh=0.5)
            if len(boxes) > 0:
                boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                for box in boxes:
                    temp = dict()
                    temp['image_id'] = i+1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = float(box[4])
                    res.append(temp)
                    #cv2.rectangle(img, (int(box[0] / img_scale), int(box[1] / img_scale)),
                    #              (int(box[2] + box[0] / img_scale), int(box[3] + box[1] / img_scale)), [0, 0, 255], 2)
                #cv2.imwrite('./data/output/detection/' + img_path[-4] + '.jpg', img)

            print('\r%d/%d' % (i + 1, len(testloader))),
            sys.stdout.flush()

            cls_loss, reg_loss, off_loss = criterion(outputs, labels)
            attloss_1, attloss_2 = attention(attmap, gtbox)
            loss = cls_loss + reg_loss + off_loss + attloss_1 + attloss_2
            cls_loss, reg_loss, off_loss, loss = cls_loss.item(), reg_loss.item(), off_loss.item(), loss.item()
            print ("<Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, gam_gts: %.6f, gam_all: %.6f" % (loss, cls_loss, reg_loss, off_loss, attloss_1, attloss_2))
            batch_loss += loss
    else:
        # LAP or normal CSP
        for i, data in enumerate(testloader, 0):
            inputs, labels, _ = data
            #img = np.array(inputs[0])
            #img_scale = inputs.shape[1] / config.size_test[0]
            inputs = inputs.cuda()
            labels = [l.cuda().float() for l in labels]
            with torch.no_grad():
                outputs= net(inputs)
                pos, height, offset = outputs[0], outputs[1], outputs[2]

            boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test,
                                     score=0.1, down=4, nms_thresh=0.5)
            if len(boxes) > 0:
                boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                for box in boxes:
                    temp = dict()
                    temp['image_id'] = i + 1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist()
                    temp['score'] = float(box[4])
                    res.append(temp)
                    #cv2.rectangle(img, (int(box[0] / img_scale), int(box[1] / img_scale)),
                    #              (int(box[2] + box[0] / img_scale), int(box[3] + box[1] / img_scale)), [0, 0, 255], 2)
                #cv2.imwrite('./data/output/detection/' + str(i)+ '.jpg', img)
            print('\r%d/%d' % (i + 1, len(testloader)))
            sys.stdout.flush()

            cls_loss, reg_loss, off_loss = criterion(outputs, labels)
            loss = cls_loss + reg_loss + off_loss
            cls_loss, reg_loss, off_loss, loss = cls_loss.item(), reg_loss.item(), off_loss.item(), loss.item()
            print ("<Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f " % (loss, cls_loss, reg_loss, off_loss))
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
    #print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    val_loss = batch_loss / len(testloader)
    if log is not None:
        log.write("%.7f %.7f %.7f %.7f\n" % tuple(MRs))
        log.write('%d %.7f\n' % (epoch ,val_loss))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs[0], val_loss


if __name__ == '__main__':
    val()
    #val_epochs()
