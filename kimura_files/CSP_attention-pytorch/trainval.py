import os
import time
import torch
import cv2
import json
import argparse
import matplotlib.pyplot as plt
import torch.optim as optim
from copy import deepcopy
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
parser.add_argument('--validation', default=True, help='validation in training')
args = parser.parse_args()

config = Config()
config.train_path = './data/citypersons'
config.test_path = './data/citypersons'
config.gpu_ids = [0, 1, 2, 3]
config.onegpu = 1
config.size_train = (1024, 2048)#(320, 640)#
config.size_test = (1024, 2048)
config.init_lr = 1e-5
config.num_epochs = 300
config.offset = True
config.val = False #args.validation
config.val_frequency = 1

# dataset
print('Dataset...')
traintransform = Compose(
    [ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
traindataset = CityPersons(path=config.train_path, type='train', config=config,
                           transform=traintransform)
trainloader = DataLoader(traindataset, batch_size=config.onegpu*len(config.gpu_ids))

if config.val:
    testtransform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testdataset = CityPersons(path=config.train_path, type='val', config=config,
                              transform=testtransform, preloaded=True)
    testloader = DataLoader(testdataset, batch_size=1)

# net
print('Net...')
net = CSPNet(phase='train', model=args.ABN_model, lap_type=args.LAP_type).cuda()
# To continue training or val
#net.load_state_dict(torch.load('./ckpt/CSPNet-150.pth'))

# position
center = cls_pos().cuda()
height = reg_pos().cuda()
offset = offset_pos().cuda()
attention = AttLoss()

# optimizer
params = []
for n, p in net.named_parameters():
    if p.requires_grad:
        params.append({'params': p})
    else:
        print(n)

if config.teacher:
    print('I found this teacher model is useless, I disable this training option')
    exit(1)
    teacher_dict = net.state_dict()

#if len(config.gpu_ids) > 1:
net = nn.DataParallel(net, device_ids=config.gpu_ids)

optimizer = optim.Adam(params, lr=config.init_lr)

batchsize = config.onegpu * len(config.gpu_ids)
train_batches = len(trainloader)

#config.print_conf()

def criterion(output, label):
    cls_loss = center(output[0], label[0])
    reg_loss = height(output[1], label[1])
    off_loss = offset(output[2], label[2])
    return cls_loss, reg_loss, off_loss


def train():

    print('Training start')
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists('./output'):
        os.mkdir('./output')

    # open log file
    log_file = './log/' + time.strftime('%Y%m%d', time.localtime(time.time()))+'.log'
    log = open(log_file, 'w')
    if config.val:
        vallog_file = log_file + '.val'
        vallog = open(vallog_file, 'w')

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
        t1 = time.time()

        res = []

        epoch_loss = 0.0
        net.train()

        if args.ABN_model == 'GAM':
            for i, data in enumerate(trainloader, 0):

                t3 = time.time()
                # get the inputs
                inputs, labels, boxes = data
                inputs = inputs.cuda()
                labels = [l.cuda().float() for l in labels]
                boxes = boxes.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # heat map
                outputs, attmap = net(inputs)

                # loss
                cls_loss, reg_loss, off_loss = criterion(outputs, labels)
                attloss_1, attloss_2 = attention(attmap, boxes)
                loss = cls_loss + reg_loss + off_loss + attloss_1 + attloss_2

                # back-prop
                loss.backward()

                # update param
                optimizer.step()
                if config.teacher:
                    for k, v in net.module.state_dict().items():
                        if k.find('num_batches_tracked') == -1:
                            teacher_dict[k] = config.alpha * teacher_dict[k] + (1 - config.alpha) * v
                        else:
                            teacher_dict[k] = 1 * v

                # print statistics
                batch_loss = loss.item()
                batch_cls_loss = cls_loss.item()
                batch_reg_loss = reg_loss.item()
                batch_off_loss = off_loss.item()
                batch_att_loss_1 = attloss_1.item()
                batch_att_loss_2 = attloss_2.item()

                t4 = time.time()
                print('\r[Epoch %d/300, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, gam_gts: %.6f, gam_all: %.6f, Time: %.3f sec        ' %
                     (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_reg_loss, batch_off_loss, batch_att_loss_1, batch_att_loss_2, t4 - t3))
        else:
            # LAP or normal CSP
            for i, data in enumerate(trainloader, 0):
                t3 = time.time()
                # get the inputs
                inputs, labels, _ = data
                inputs = inputs.cuda()
                labels = [l.cuda().float() for l in labels]

                # zero the parameter gradients
                optimizer.zero_grad()

                # heat map
                outputs = net(inputs)

                # loss
                # cls_loss = center(outputs[0], labels[0]) # center only
                cls_loss, reg_loss, off_loss = criterion(outputs, labels)
                loss = cls_loss + reg_loss + off_loss

                # back-prop
                loss.backward()

                # update param
                optimizer.step()
                if config.teacher:
                    for k, v in net.module.state_dict().items():
                        if k.find('num_batches_tracked') == -1:
                            teacher_dict[k] = config.alpha * teacher_dict[k] + (1 - config.alpha) * v
                        else:
                            teacher_dict[k] = 1 * v

                # print statistics
                batch_loss = loss.item()
                batch_cls_loss = cls_loss.item()
                batch_reg_loss = reg_loss.item()
                batch_off_loss = off_loss.item()
                t4 = time.time()
                print(
                            '\r[Epoch %d/300, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, Time: %.3f sec        ' %
                            (
                            epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_reg_loss, batch_off_loss,
                            t4 - t3))

                epoch_loss += batch_loss

            '''
            # set train score
            pos, height, offset = torch.tensor(outputs[0]), torch.tensor(outputs[1]), torch.tensor(outputs[2])
            for batch in range(pos.size(0)):
                pos_b, height_b, offset_b = pos[batch], height[batch], offset[batch]
                offset_b = torch.unsqueeze(offset_b, 0)
                #print (len(labels[0]))
                #labels_b = labels[batch]
                boxes = parse_det_offset(pos_b.cpu().numpy(), height_b.cpu().numpy(), offset_b.cpu().numpy(), config.size_train,
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
                """
                if len(labels) > 0:
                    for label in labels:
                """

                #print('\r%d/%d' % (i + 1, len(trainloader))),
                sys.stdout.flush()
                '''

        print('')

        t2 = time.time()
        epoch_loss /= len(trainloader)
        train_loss_list[epoch+1] = epoch_loss
        print('Epoch %d end, AvgLoss is %.6f, Time used %.1f sec.' % (epoch+1, epoch_loss, int(t2-t1)))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_loss_epoch = epoch + 1
        print('Epoch %d has lowest loss: %.7f' % (best_loss_epoch, best_loss))

        with open('./_temp_train.json', 'w') as f:
            json.dump(res, f)

        '''
        MRs = validate('./eval_city/train_gt.json', './_temp_train.json')
        print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
              % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))
        train_acc_list[epoch+1] = MRs[0]
        '''


        if config.val and (epoch + 1) % config.val_frequency == 0: # and epoch + 1 > 10
            cur_mr, val_loss = val(epoch=epoch)
            val_loss_list[epoch+1] = val_loss
            val_acc_list[epoch+1] = cur_mr
            if cur_mr < best_mr:
                best_mr = cur_mr
                best_mr_epoch = epoch + 1
            print('Epoch %d has lowest MR: %.7f' % (best_mr_epoch, best_mr))


        #log.write("%.7f %.7f %.7f %.7f\n" % tuple(MRs))

        log.write('%d %.7f\n' % (epoch+1, epoch_loss))
            

        print('Save checkpoint...')
        filename = './ckpt/%s-%d.pth' % (net.module.__class__.__name__, epoch+1)

        torch.save(net.module.state_dict(), filename)
        if config.teacher:
            torch.save(teacher_dict, filename+'.tea')

        print('%s saved.' % filename)


        print ("Make graph...")
        plt.plot(range(1, epoch+1), train_loss_list[1:epoch+1], 'r-', label='train_loss')
        plt.plot(range(1, epoch+1), val_loss_list[1:epoch+1], 'b-', label='val_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        #plt.grid()
        plt.savefig("data/loss.png")
        plt.close()
        print ("----  saved loss graph  ----")

        '''
        #plt.plot(range(1, epoch+1), train_acc_list, 'r-', label='train_MR')
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

def val(log=None, epoch=None):
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
            #img = inputs[0]
            #img_scale = inputs.size[1] / config.size_test[0]
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
                    #cv2.rectangle(inputs, (int(box[0] / img_scale), int(box[1] / img_scale)),
                    #              (int(box[2] + box[0] / img_scale), int(box[3] + box[1] / img_scale)), [0, 0, 255], 2)
                #cv2.imwrite('./data/output/detection/' + str(i) + '.jpg', inputs)

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
                outputs, _ = net(inputs)
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
                    #cv2.rectangle(inputs, (int(box[0] / img_scale), int(box[1] / img_scale)),
                    #              (int(box[2] + box[0] / img_scale), int(box[3] + box[1] / img_scale)), [0, 0, 255], 2)
                #cv2.imwrite('./data/output/detection/' + str(i) + '.jpg', inputs)

            print('\r%d/%d' % (i + 1, len(testloader))),
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
    train()
    #val() # validation only
