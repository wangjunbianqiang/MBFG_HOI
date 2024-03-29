"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist

from torch.utils.data import Dataset
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import hflip

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.models import mask_rcnn
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, HandyTimer, BoxPairAssociation, all_gather

def custom_collate(batch):  # 将一个批次的图片，检测，以及目标分别封装成一个列表
    images = []
    detections = []
    targets = []
    for im, det, tar in batch:
        images.append(im)
        detections.append(det)
        targets.append(tar)
    # print('utils里面images： ', images)
    return images, detections, targets


def calukeypointsfeatures():

    images, _, _ = custom_collate()
    model = mask_rcnn.keypointrcnn_resnet50_fpn(pretrained=pretrained)
    model.eval()
    pred_dict = model(images)
    keypointfeature = pred_dict.get('keypointrcnn').get('keypoints')
    keypointfeature_re = keypointfeature[:, :, :2]
    print('1 keypointfeature_re.shape', keypointfeature_re.shape)
    return keypointfeature_re


class DataFactory(Dataset):  # 验证是否是HICO-DET和Vcoco数据集
    def __init__(self,
            name, partition,
            data_root, detection_root,
            flip=False,
            box_score_thresh_h=0.2,
            box_score_thresh_o=0.2
            ):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.human_idx = 49   # hicodet的数据集中，人类别的编号索引是49
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.human_idx = 1   # vcoco的数据集中，人类别的编号索引是1

        self.name = name
        self.detection_root = detection_root

        self.box_score_thresh_h = box_score_thresh_h
        self.box_score_thresh_o = box_score_thresh_o
        self._flip = torch.randint(0, 2, (len(self.dataset),)) if flip \
            else torch.zeros(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def filter_detections(self, detection):  # 执行NMS，删除置信度低的人和物体的检测框
        """Perform NMS and remove low scoring examples"""

        boxes = torch.as_tensor(detection['boxes'])
        labels = torch.as_tensor(detection['labels'])
        scores = torch.as_tensor(detection['scores'])

        # Filter out low scoring human boxes
        idx = torch.nonzero(labels == self.human_idx).squeeze(1)
        keep_idx = idx[torch.nonzero(scores[idx] >= self.box_score_thresh_h).squeeze(1)]

        # Filter out low scoring object boxes
        idx = torch.nonzero(labels != self.human_idx).squeeze(1)
        keep_idx = torch.cat([
            keep_idx,
            idx[torch.nonzero(scores[idx] >= self.box_score_thresh_o).squeeze(1)]
        ])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        return dict(boxes=boxes, labels=labels, scores=scores)

    def flip_boxes(self, detection, target, w):
        detection['boxes'] = pocket.ops.horizontal_flip_boxes(w, detection['boxes'])
        target['boxes_h'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_h'])
        target['boxes_o'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_o'])

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        detection_path = os.path.join(
            self.detection_root,
            self.dataset.filename(i).replace('jpg', 'json')
        )
        with open(detection_path, 'r') as f:
            detection = pocket.ops.to_tensor(json.load(f),
                input_format='dict')

        if self._flip[i]:
            image = hflip(image)
            w, _ = image.size
            self.flip_boxes(detection, target, w)
        image = pocket.ops.to_tensor(image, 'pil')  # 图片转换成向量的形式

        # 每一张图片的处理输出结果
        # print('image是： ', image.shape)
        # print('image是： ', image)  # 输出的是tensor，也就是image = pocket.ops.to_tensor(image, 'pil')，将图片中的特征转换成特征向量的形式
        # print('detection是： ', detection)  # 输出的是boxex,labels,scores，也就是检测到的人或者物体以及对应的标签和置信度
        # print('target是： ', target)  # 输出的是可能存在人物交互的人物对。输出图片可能存在交互中人检测框的坐标，物体检测框的坐标，交互的人物对编号，物体类别，行为类别，以及类别编号。verb和label输出一样
        # # {'boxes_h': tensor([[409.,   3., 639., 271.]]), 'boxes_o': tensor([[ 98.,  39., 598., 405.]]), 'hoi': tensor([266]), 'object': tensor([17]), 'verb': tensor([15]), 'labels': tensor([15])}
        return image, detection, target

def test(net, test_loader):        # 利用训练好的代码进行模型的测试
    testset = test_loader.dataset.dataset
    associate = BoxPairAssociation(min_iou=0.5)

    meter = DetectionAPMeter(
        600, nproc=1,
        num_gt=testset.anno_interaction,
        algorithm='11P'
    )
    net.eval()
    for batch in tqdm(test_loader):
        inputs = pocket.ops.relocate_to_cuda(batch[:-1])
        with torch.no_grad():
            output = net(*inputs)
        if output is None:
            continue

        # Batch size is fixed as 1 for inference
        assert len(output) == 1, "Batch size is not 1"
        output = pocket.ops.relocate_to_cpu(output[0])
        target = batch[-1][0]
        # Format detections
        box_idx = output['index']
        boxes_h = output['boxes_h'][box_idx]
        boxes_o = output['boxes_o'][box_idx]
        objects = output['object'][box_idx]
        scores = output['scores']
        verbs = output['prediction']
        interactions = torch.tensor([
            testset.object_n_verb_to_interaction[o][v]
            for o, v in zip(objects, verbs)
        ])
        # Associate detected pairs with ground truth pairs
        labels = torch.zeros_like(scores)
        unique_hoi = interactions.unique()
        for hoi_idx in unique_hoi:
            gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
            det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
            if len(gt_idx):
                labels[det_idx] = associate(
                    (target['boxes_h'][gt_idx].view(-1, 4),
                    target['boxes_o'][gt_idx].view(-1, 4)),
                    (boxes_h[det_idx].view(-1, 4),
                    boxes_o[det_idx].view(-1, 4)),
                    scores[det_idx].view(-1)
                )

        meter.append(scores, interactions, labels)

    return meter.eval()

# 这个就是训练的类，继承的是pocket.core中的DistributedLearningEngine
class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, train_loader, val_loader, num_classes=117, **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
        self.val_loader = val_loader
        self.num_classes = num_classes

    def _on_start(self):
        self.meter = DetectionAPMeter(self.num_classes, algorithm='11P')
        self.hoi_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)
        self.intr_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)

    def _on_each_iteration(self):
        self._state.optimizer.zero_grad()
        output = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        # print('这是输入的值。。。。。。。。。。。。。。。。。。。。。。。。。。。', self._state.inputs)

        # print('每一轮的输出', output)

        loss_dict = output.pop()
        # print('输出的pop', loss_dict)  # {'hoi_loss': tensor(2.1630, device='cuda:0', grad_fn=<DivBackward0>), 'interactiveness_loss': tensor(0.4872, device='cuda:0', grad_fn=<DivBackward0>)}
        if loss_dict['hoi_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.loss.backward()
        self._state.optimizer.step()

        self.hoi_loss.append(loss_dict['hoi_loss'])
        self.intr_loss.append(loss_dict['interactiveness_loss'])

        self._synchronise_and_log_results(output, self.meter)

    def _on_end_epoch(self):    # 重写了pocket.core中的DistributedLearningEngine中的_on_end_epoch的方法，每一轮训练结束，打印一次输出
        timer = HandyTimer(maxlen=2)
        # Compute training mAP
        if self._rank == 0:
            with timer:
                ap_train = self.meter.eval()
        # Run validation and compute mAP
        with timer:
            ap_val = self.validate()
        # Print performance and time
        if self._rank == 0:
            print("Epoch: {} | training mAP: {:.4f}, evaluation time: {:.2f}s |"
                "validation mAP: {:.4f}, total time: {:.2f}s\n".format(
                    self._state.epoch, ap_train.mean().item(), timer[0],
                    ap_val.mean().item(), timer[1]
            ))
            self.meter.reset()
        super()._on_end_epoch()

    def _print_statistics(self):
        super()._print_statistics()
        hoi_loss = self.hoi_loss.mean()
        intr_loss = self.intr_loss.mean()
        if self._rank == 0:
            print(f"=> HOI classification loss: {hoi_loss:.4f},",
            f"interactiveness loss: {intr_loss:.4f}")
        self.hoi_loss.reset()
        self.intr_loss.reset()

    def _synchronise_and_log_results(self, output, meter):
        scores = []; pred = []; labels = []
        # Collate results within the batch
        for result in output:
            scores.append(result['scores'].detach().cpu().numpy())
            pred.append(result['prediction'].cpu().float().numpy())
            labels.append(result["labels"].cpu().numpy())
        # Sync across subprocesses
        all_results = np.stack([
            np.concatenate(scores),
            np.concatenate(pred),
            np.concatenate(labels)
        ])
        all_results_sync = all_gather(all_results)
        # Collate and log results in master process
        if self._rank == 0:
            scores, pred, labels = torch.from_numpy(
                np.concatenate(all_results_sync, axis=1)
            ).unbind(0)
            meter.append(scores, pred, labels)

    @torch.no_grad()
    def validate(self):
        meter = DetectionAPMeter(self.num_classes, algorithm='11P')
        
        self._state.net.eval()
        for batch in self.val_loader:
            inputs = pocket.ops.relocate_to_cuda(batch)
            results = self._state.net(*inputs)

            self._synchronise_and_log_results(results, meter)

        # Evaluate mAP in master process
        if self._rank == 0:
            return meter.eval()
        else:
            return None
