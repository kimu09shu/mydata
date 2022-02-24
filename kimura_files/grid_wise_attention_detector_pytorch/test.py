import os
import torch
import cv2
import argparse
from PIL import Image
import torch.optim as optim

from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network import Network
from config import Config
from dataloader.loader import *
from utils.nms_wrapper import nms
from utils.functions import parse_det_offset

parser = argparse.ArgumentParser(description='GWA Test')
parser.add_argument('--detection_model', default='anchor', type=str, help='use detection( anchor or csp or retina)')
parser.add_argument('--detection_scale', default='single', type=str, help='single or multi')
parser.add_argument('--image_dir', default='data/citypersons/images/val', type=str, help='test image dir')
parser.add_argument('--image_path', default='frankfurt_000001_005898_leftImg8bit.png', type=str, help='test image path')
parser.add_argument('--model_dir', default='./ckpt', type=str, help='weight dir')
parser.add_argument('--model_path', default='Network-150.pth', type=str, help='weight path')
args = parser.parse_args()


config = Config()
config.gpu_ids = [0]
config.size_test = (512, 1024)#(1024, 2048)#
config.init_lr = 1e-5
config.offset = True
config.rois_thresh = 0.7
config.num_classes = 1

# dataset
print('Dataset...')
testtransform = Compose(
    [ToTensor()])
img_dir = args.image_dir
img_path = args.image_path
img = Image.open('./' + img_dir + '/' + img_path).convert('RGB')
img = img.resize((config.size_test[1], config.size_test[0]))
inputs = testtransform(img)

# net
print('Net...')
net = Network(input_size=config.size_test, phase='test', model=args.detection_model, num_classes=config.num_classes).cuda()
# To continue training or val
net.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_path)))

if not os.path.exists('./data/output'):
    os.mkdir('./data/output')
if not os.path.exists('./data/output/attmap'):
    os.mkdir('./data/output/attmap')
if not os.path.exists('./data/output/detection'):
    os.mkdir('./data/output/detection')

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

#config.print_conf()

def test():
    net.eval()

    if config.teacher:
        print('Load teacher params')
        net.module.load_state_dict(teacher_dict)

    print('Perform validation...')

    # print(data.size())
    data = inputs.unsqueeze(0)
    input = data
    im = np.array(img)
    img_scale = im.shape[0] / 1024
    input = input.cuda()

    if args.detection_model == 'anchor':
        with torch.no_grad():
            rois = net(input, None, [config.size_test[0], config.size_test[1], config.size_test[0] / 1024])
            inds = (rois[:, :, 4] > config.rois_thresh)
            roi_keep = rois[:, inds[0], :]
    
        if roi_keep.dim() == 1:
            roi_keep = rois
    
        roi_keep[:, :, 0:4] /= (config.size_train[0] / 1024)
    
        roi_single = roi_keep[0]
    
        roi_single = roi_single.cpu().numpy()
    
        # print(roi_single)
        nms_keep = nms(roi_single, 0.3, usegpu=False, gpu_id=0)
        cls_dets = roi_single[nms_keep, :]
        if len(cls_dets) > 0:
            #cls_dets[:, [2, 3]] -= cls_dets[:, [0, 1]]
            for box in cls_dets:
                cv2.rectangle(im, (int(box[0] / img_scale), int(box[1] / img_scale)),(int(box[2]/ img_scale), int(box[3]/ img_scale)), [0, 0, 255], 2)

    elif args.detection_model == 'retina':
        scores, classification, transformed_anchors = net(input, None, [config.size_test[0], config.size_test[1],
                                                                        config.size_test[0] / 1024])
        idxs = np.where(scores.cpu() > config.rois_thresh)
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            #x1 = int(bbox[0])
            #y1 = int(bbox[1])
            #x2 = int(bbox[2])
            #y2 = int(bbox[3])
            cv2.rectangle(im, (int(bbox[0] ), int(bbox[1])),
                          (int(bbox[2] ), int(bbox[3])), [0, 0, 255], 2)

        cv2.imwrite('./data/output/detection/' + img_path[:-4] + '.jpg', im)
        sys.stdout.flush()
    elif args.detection_model == 'csp':
        outputs = net(input)
        pos, height, offset = outputs[0], outputs[1], outputs[2]

        boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test,
                                 score=0.3, down=4, nms_thresh=0.5)

        if len(boxes) > 0:
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            for box in boxes:
                cv2.rectangle(im, (int(box[0] / img_scale), int(box[1] / img_scale)),
                              (int(box[2] + box[0] / img_scale), int(box[3] + box[1] / img_scale)), [255, 255, 0], 2)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./data/output/detection/' + img_path[:-4] + '.jpg', im)
        sys.stdout.flush()

    print('')

if __name__ == '__main__':
    test()
