"""
Transforms

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from torch import nn
from torchvision.models.detection import transform

class HOINetworkTransform(transform.GeneralizedRCNNTransform):
    """
    Transformations for input image and target (box pairs)

    Arguments(Positional):
        min_size(int)
        max_size(int)
        image_mean(list[float] or tuple[float])
        image_std(list[float] or tuple[float])

    Refer to torchvision.models.detection for more details
    """
    def __init__(self, *args):
        super().__init__(*args)

    def resize(self, image, target):
        """
        Override method to resize box pairs
        """
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        scale_factor = min(
            self.min_size[0] / min_size,
            self.max_size / max_size
        )

        image = nn.functional.interpolate(
            image[None], scale_factor=scale_factor,
            mode='bilinear', align_corners=False,
            recompute_scale_factor=True
        )[0]
        if target is None:
            return image, target

        target['boxes_h'] = transform.resize_boxes(target['boxes_h'],  # 改变检测框的大小，也就是高和宽
            (h, w), image.shape[-2:])
        target['boxes_o'] = transform.resize_boxes(target['boxes_o'],
            (h, w), image.shape[-2:])

        return image, target

    def postprocess(self, results, image_shapes, original_image_sizes):
        if self.training:
            loss = results.pop()  # 因为损失是加在最后面的，如果不是训练的话，就不需要把损失加上了。如果是训练过程的话，后面的代码就把损失加上了。

        for pred, im_s, o_im_s in zip(results, image_shapes, original_image_sizes):
            boxes_h, boxes_o = pred['boxes_h'], pred['boxes_o']
            boxes_h = transform.resize_boxes(boxes_h, im_s, o_im_s)
            boxes_o = transform.resize_boxes(boxes_o, im_s, o_im_s)
            pred['boxes_h'], pred['boxes_o'] = boxes_h, boxes_o

        if self.training:
            results.append(loss)
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！训练完或者过程中才能看见打印的输出
        # print('这是transform.py里面的输出', results)   # 训练的话，就会加上loss
        return results

'''
results: List[dict]
            Results organised by images, with keys as below
            `boxes_h`: Tensor[M, 4]
            `boxes_o`: Tensor[M, 4]
            `index`: Tensor[L]
                Expanded indices of box pairs for each predicted action
            `prediction`: Tensor[L]
                Expanded indices of predicted actions
            `scores`: Tensor[L]
                Scores for each predicted action
            `object`: Tensor[M]
                Object indices for each pair
            `prior`: Tensor[2, L]
                Prior scores for expanded pairs
            `weights`: Tensor[M]
                Unary weights for each box pair
            `labels`: Tensor[L], optional
                Binary labels on each action
            `unary_labels`: Tensor[M], optional
                Labels for the unary weights
'''