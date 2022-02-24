import os
import time
import torch
import json
import argparse
import matplotlib.pyplot as plt
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.network import Network
from net.loss import *
from config import Config
from dataloader.loader import *
from val import val
#from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate

parser = argparse.ArgumentParser(description='GWA Training')
parser.add_argument('--detection_model', default='retina', type=str, help='use detection (anchor or csp or retina)')
parser.add_argument('--detection_scale', default='single', type=str, help='single or multi')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
args = parser.parse_args()

config = Config()
config.train_path = './data/citypersons'
config.test_path = './data/citypersons'
config.gpu_ids = [0, 1]
config.onegpu = 2
config.size_train = [512, 1024] #(768, 1536)
config.size_test = [512, 1024]
config.init_lr = 1e-5
config.num_epochs = 300
config.offset = True
config.val = False
config.val_frequency = 10
config.rois_thresh = 0.5
config.num_classes = 1
config.down = 8

img_imfo = [config.size_train[0], config.size_train[1], config.size_train[0] / 1024]

if args.detection_model == 'csp':
    center = cls_pos().cuda()
    height = reg_pos().cuda()
    offset = offset_pos().cuda()

def criterion(output, label):
    cls_loss = center(output[0], label[0])
    reg_loss = height(output[1], label[1])
    off_loss = offset(output[2], label[2])
    return cls_loss, reg_loss, off_loss

# dataset
print('Dataset...')
traintransform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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
net = Network(input_size=config.size_train, phase='train', model=args.detection_model, scale=args.detection_scale, num_classes=config.num_classes).cuda()
# To continue training or val
if args.resume_epoch > 0:
    net.load_state_dict(torch.load('./ckpt/Network-'+str(args.resume_epoch)+'.pth'))

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

config.print_conf()

def train():

    print('Training start')
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists('./log'):
        os.mkdir('./log')

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
    train_boxloss_list = np.zeros(max_epoch + 1)
    train_classloss_list = np.zeros(max_epoch + 1)
    train_offsetloss_list = np.zeros(max_epoch + 1)
    val_loss_list = np.zeros(max_epoch+1)
    train_acc_list = np.zeros(max_epoch+1)
    val_acc_list = np.zeros(max_epoch+1)


    for epoch in range(max_epoch):
        print('----------------------------')
        print('Epoch %d begin' % (epoch + 1))
        t1 = time.time()

        res = []

        epoch_loss = 0.0
        epoch_boxloss = 0.0
        epoch_classloss = 0.0
        epoch_offsetloss = 0.0
        net.train()

        for i, data in enumerate(trainloader, 0):

            t3 = time.time()
            # get the inputs
            inputs, boxes, labels = data

            inputs = inputs.cuda()
            boxes = boxes.cuda()
            labels = [l.cuda().float() for l in labels]
            # zero the parameter gradients
            optimizer.zero_grad()

            if args.detection_model == 'anchor':
                outputs, anchor_info = net(inputs, boxes, img_imfo)
                if args.detection_scale == 'single':
                    loss_c = clsLoss(outputs[0][0], anchor_info[0][0], config.onegpu * len(config.gpu_ids))
                    loss_b = boxLoss(outputs[0][1], anchor_info[0][1], anchor_info[0][2], anchor_info[0][3], sigma=3, dim=[1, 2, 3])
                else:
                    loss_c_1 = clsLoss(outputs[0][0], anchor_info[0][0], config.onegpu * len(config.gpu_ids))
                    loss_c_2 = clsLoss(outputs[1][0], anchor_info[1][0], config.onegpu * len(config.gpu_ids))
                    loss_c_3 = clsLoss(outputs[2][0], anchor_info[2][0], config.onegpu * len(config.gpu_ids))
                    loss_b_1 = boxLoss(outputs[0][1], anchor_info[0][1], anchor_info[0][2], anchor_info[0][3], sigma=3, dim=[1, 2, 3])
                    loss_b_2 = boxLoss(outputs[1][1], anchor_info[1][1], anchor_info[1][2], anchor_info[1][3], sigma=3, dim=[1, 2, 3])
                    loss_b_3 = boxLoss(outputs[2][1], anchor_info[2][1], anchor_info[2][2], anchor_info[2][3], sigma=3, dim=[1, 2, 3])
                    loss_c = loss_c_1 + loss_c_2 + loss_c_3
                    loss_b = loss_b_1 + loss_b_2 + loss_b_3
                    batch_cls_loss_1 = loss_c_1.item()
                    batch_box_loss_1 = loss_b_1.item()
                    batch_cls_loss_2 = loss_c_2.item()
                    batch_box_loss_2 = loss_b_2.item()
                    batch_cls_loss_3 = loss_c_3.item()
                    batch_box_loss_3 = loss_b_3.item()


                loss = loss_c + loss_b
                if bool(loss == 0):
                    continue

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
                batch_cls_loss = loss_c.item()
                batch_box_loss = loss_b.item()



                t4 = time.time()


                print(
                        '\r[Epoch %d/300, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, box: %.6f, Time: %.3f sec        ' %
                        (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_box_loss, t4 - t3)),

                '''
                print(
                        '\r[Epoch %d/300, Batch %d/%d]$ <Total loss: %.6f> small - cls: %.6f, box: %.6f,middle - cls: %.6f, box: %.6f, large - cls: %.6f, box: %.6f,Time: %.3f sec        ' %
                        (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss_1, batch_box_loss_1, batch_cls_loss_2, batch_box_loss_2,  batch_cls_loss_3, batch_box_loss_3,t4 - t3)),

                '''
                epoch_loss += batch_loss
                epoch_boxloss += batch_box_loss
                epoch_classloss += batch_cls_loss


            elif args.detection_model == 'retina':
                loss_c, loss_b = net(inputs, boxes, img_imfo)
                # loss
                loss_b = loss_b.mean()
                loss_c = loss_c.mean()
                loss = loss_c + loss_b

                if bool(loss == 0):
                    continue

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

                batch_cls_loss = loss_c.item()
                batch_box_loss = loss_b.item()

                t4 = time.time()

                print(
                        '\r[Epoch %d/300, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, box: %.6f, Time: %.3f sec        ' %
                        (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_box_loss, t4 - t3)),

                epoch_loss += batch_loss
                epoch_boxloss += batch_box_loss
                epoch_classloss += batch_cls_loss

            elif args.detection_model == 'csp':
                outputs = net(inputs)
                cls_loss, reg_loss, off_loss = criterion(outputs, labels)
                batch_cls_loss = cls_loss.item()
                batch_reg_loss = reg_loss.item()
                batch_off_loss = off_loss.item()

                print('\r[Epoch %d/300, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, Time: %.3f sec        ' %
                                      (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_reg_loss, batch_off_loss, t4-t3)),

                epoch_loss += batch_loss
                epoch_boxloss += batch_reg_loss
                epoch_classloss += batch_cls_loss
                epoch_offsetloss += batch_off_loss


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
        epoch_boxloss /= len(trainloader)
        epoch_classloss /= len(trainloader)
        epoch_offsetloss /= len(trainloader)
        train_loss_list[epoch+1] = epoch_loss
        train_boxloss_list[epoch+1] = epoch_boxloss
        train_classloss_list[epoch+1]= epoch_classloss
        train_offsetloss_list[epoch + 1] = epoch_offsetloss
        print('Epoch %d end, AvgLoss is %.6f,  AvgboxLoss is %.6f,  AvgclsLoss is %.6f,  AvgoffLoss is %.6f, Time used %.1f sec.' % (epoch+1, epoch_loss,epoch_boxloss, epoch_classloss, epoch_offsetloss,int(t2-t1)))
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
            cur_mr = val(epoch=epoch)
            #val_loss_list[epoch+1] = val_loss
            val_acc_list[epoch+1] = cur_mr
            if cur_mr < best_mr:
                best_mr = cur_mr
                best_mr_epoch = epoch + 1
            print('Epoch %d has lowest MR: %.7f' % (best_mr_epoch, best_mr))


        #log.write("%.7f %.7f %.7f %.7f\n" % tuple(MRs))

        #log.write('%d %.7f\n' % (epoch+1, epoch_loss))
            

        print('Save checkpoint...')
        filename = './ckpt/%s-%d.pth' % (net.module.__class__.__name__, epoch+1)

        torch.save(net.module.state_dict(), filename)
        if config.teacher:
            torch.save(teacher_dict, filename+'.tea')

        print('%s saved.' % filename)


        print ("Make graph...")
        plt.plot(range(1, epoch+1), train_loss_list[1:epoch+1], 'r-', label='train_loss')
        plt.plot(range(1, epoch+1), train_boxloss_list[1:epoch+1], 'g-', label='train_scale_loss')
        plt.plot(range(1, epoch+1), train_classloss_list[1:epoch+1], 'b-', label='train_class_loss')
        plt.plot(range(1, epoch + 1), train_classloss_list[1:epoch + 1], 'y-', label='train_offset_loss')
        #plt.plot(range(1, epoch+1), val_loss_list[1:epoch+1], 'b-', label='val_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        #plt.grid()
        plt.savefig("data/loss.png")
        plt.close()
        print ("----  saved loss graph  ----")

        #plt.plot(range(1, epoch+1), train_acc_list, 'r-', label='train_MR')
        '''
        plt.plot(range(1, epoch+1), val_acc_list[1:epoch+1], 'b-', label='val_MR')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('MR')
        # plt.grid()
        plt.savefig("data/acc.png")
        plt.close()
        print ("----  saved acc graph ----")
        '''



    '''
    log.close()
    if config.val:
        vallog.close()
    '''

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
    count = 0
    for i, data in enumerate(testloader, 0):
        inputs, gtbox = data
        # print(gts)
        inputs = inputs.cuda()
        if args.detection_model == 'retina':
            with torch.no_grad():
                scores, classification, transformed_anchors = net(inputs, None, img_imfo)
                # print(scores, classification, transformed_anchors)
                idxs = np.where(scores.cpu() > config.rois_thresh)
                count += 1
                print(count)
                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :].cpu().numpy()

                    x = int(bbox[0] / (config.size_test[0] / 1024))
                    y = int(bbox[1] / (config.size_test[0] / 1024))
                    w = int(bbox[2] / (config.size_test[0] / 1024))
                    h = int(bbox[3] / (config.size_test[0] / 1024))
                    temp = dict()
                    temp['image_id'] = count
                    temp['category_id'] = classification[j].item() + 1
                    temp['bbox'] = [x, y, w - x, h - y]  # bbox.tolist()
                    temp['score'] = float(scores[j].item())
                    res.append(temp)

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
          # print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
          % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))
    #val_loss = batch_loss / len(testloader)
    if log is not None:
        log.write("%.7f %.7f %.7f %.7f\n" % tuple(MRs))
        #log.write('%d %.7f\n' % (epoch, val_loss))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs[0]  # , val_loss


if __name__ == '__main__':
    train()
    #val()
