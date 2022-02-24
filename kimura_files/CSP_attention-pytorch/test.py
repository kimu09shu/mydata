import os
import torch
import cv2
import argparse
from PIL import Image
import torch.optim as optim

from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network import CSPNet, CSPNet_mod
from config import Config
from dataloader.loader import *
from util.functions import parse_det_offset

parser = argparse.ArgumentParser(description='CSP Training')
parser.add_argument('--ABN_model', default='LAP', type=str, help='use ABN model ( None or GAM or LAP )')
parser.add_argument('--LAP_type', default='sliding_window',type=str, help='use LAP type ( sliding_window or fixed area )')
parser.add_argument('--image_dir', default='data/citypersons/images/val', type=str, help='test image dir')
parser.add_argument('--image_path', default='frankfurt_000001_005898_leftImg8bit.png', type=str, help='test image path')
parser.add_argument('--model_dir', default='./ckpt', type=str, help='weight dir')
parser.add_argument('--model_path', default='CSPNet-150.pth', type=str, help='weight path')
args = parser.parse_args()


config = Config()
config.gpu_ids = [0]
config.size_test = (1024, 2048)
config.init_lr = 1e-5
config.offset = True

# dataset
print('Dataset...')
testtransform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
img_dir = args.image_dir
img_path = args.image_path
img = Image.open('./' + img_dir + '/' + img_path).convert('RGB')
inputs = testtransform(img)

# net
print('Net...')
net = CSPNet(phase='test', model=args.ABN_model, lap_type=args.LAP_type).cuda()
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
    img_scale = im.shape[0] / config.size_test[0]
    input = input.cuda()
    with torch.no_grad():
        outputs = net(input, img_path[:-4])
        # outputs, attmap = net(inputs)
        pos, height, offset = outputs[0], outputs[1], outputs[2]

    boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.1,
                             down=4, nms_thresh=0.5)

    if len(boxes) > 0:
        boxes[:, [2, 3]] -= boxes[:, [0, 1]]
        for box in boxes:
            cv2.rectangle(im, (int(box[0] / img_scale), int(box[1] / img_scale)),(int(box[2] + box[0] / img_scale), int(box[3] + box[1] / img_scale)), [0, 0, 255], 2)

    cv2.imwrite('./data/output/detection/' + img_path[:-4] + '.jpg', im)
    sys.stdout.flush()

    print('')

if __name__ == '__main__':
    test()
