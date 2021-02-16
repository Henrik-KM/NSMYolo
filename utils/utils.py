from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def rescale_boxes_custom(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was addedof original image
    boxes[:, 0] = boxes[:,0]*(orig_w/current_dim)
    boxes[:, 1] = boxes[:,1]*(orig_h/current_dim)
    boxes[:, 2] = boxes[:,2]*(orig_w/current_dim)
    boxes[:, 3] = boxes[:,3]*(orig_h/current_dim)
    
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    # print(wh1.shape)
    # print(wh2.shape)
    # try:
    wh2 = wh2.t()
   # except:
   #     wh2 = wh2.transpose(0,1)
        
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        try:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        except:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:
        # Get the coordinates of bounding boxes
        try:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        except:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    try:
        #Check if box completely inside other box 
        if (b1_x1 >= b2_x1 and b1_y1 >= b2_y1 and b1_x2 <= b2_x2 and b1_y2 <= b2_y2) or (b1_x1 <= b2_x1 and b1_y1 <= b2_y1 and b1_x2 >= b2_x2 and b1_y2 >= b2_y2):
            return torch.from_numpy(np.array(1))
        if inter_rect_x2 < inter_rect_x1 or inter_rect_y2 < inter_rect_y1: #If boxes are not overlapping, return 0
            return torch.from_numpy(np.array(0))
    except:
        pass
     
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)


    return iou




def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def YOLOLabelSingleParticleToMultiple(YOLOLabels,overlap_thres=0.5,xdim=128,ydim=128):
        continueFlag=True
        while continueFlag:
            continueFlag = False
            labelCombos = np.array(list(itertools.combinations(YOLOLabels,2)))

            for labelCombo in labelCombos:
                label1 = labelCombo[0,:]
                label2 = labelCombo[1,:]
                overlap = bbox_iou(torch.from_numpy(label1[1:]),torch.from_numpy(label2[1:]), x1y1x2y2=False)
                
                
                if overlap > overlap_thres:
                   newLabel = mergeLabels(label1.reshape(1,5),label2.reshape(1,5),xdim,ydim).reshape(1,5)

                   YOLOLabels =  YOLOLabels[~(label1 == YOLOLabels).all(1)]
                   YOLOLabels = YOLOLabels[~(label2 == YOLOLabels).all(1)]
                   YOLOLabels = np.append(YOLOLabels,newLabel,0)
                   if len(YOLOLabels > 1):
                       continueFlag = True
                       
                   else: 
                       continueFlag = False
                       
                   break
                   
        return YOLOLabels
            
def ConvertYOLOLabelsToCoord(YOLOLabels,xdim,ydim):
    #image,x1,y1,x2,y2
    coordinates = np.zeros(YOLOLabels.shape)
    coordinates[:,0] = YOLOLabels[:,0]
    coordinates[:,1] = (xdim-1)*YOLOLabels[:,1]-(xdim-1)/2*YOLOLabels[:,3]
    coordinates[:,2] = (ydim-1)*YOLOLabels[:,2]-(ydim-1)/2*YOLOLabels[:,4]
    coordinates[:,3] = (xdim-1)*YOLOLabels[:,1]+(xdim-1)/2*YOLOLabels[:,3]
    coordinates[:,4] = (ydim-1)*YOLOLabels[:,2]+(ydim-1)/2*YOLOLabels[:,4]
    
    return coordinates
    
def ConvertCoordToYOLOLabels(coordinates,xdim,ydim):
    #image,xc,yc,xw,yw
    labels = np.zeros(coordinates.shape)
    labels[:,0] = coordinates[:,0]
    labels[:,1] = (coordinates[:,1]+coordinates[:,3])/2/(xdim-1)
    labels[:,2] = (coordinates[:,2]+coordinates[:,4])/2/(ydim-1)
    labels[:,3] = (coordinates[:,3]-coordinates[:,1])/(xdim-1)
    labels[:,4] = (coordinates[:,4]-coordinates[:,2])/(ydim-1)
    
    return labels
    
def mergeLabels(label1,label2,xdim,ydim):
    coord1 = ConvertYOLOLabelsToCoord(label1,xdim,ydim)
    coord2 = ConvertYOLOLabelsToCoord(label2,xdim,ydim)
    
    try:
        particleNr = np.max([coord1[0],coord2[0]])
        x1 = np.min([coord1[1],coord2[1]])
        y1 = np.min([coord1[2],coord2[2]])
        x2 = np.max([coord1[3],coord2[3]])
        y2 = np.max([coord1[4],coord2[4]])
    except IndexError:
        coord1 = coord1[0]
        coord2 = coord2[0]
        particleNr = coord1[0]+coord2[0]
        x1 = np.min([coord1[1],coord2[1]])
        y1 = np.min([coord1[2],coord2[2]])
        x2 = np.max([coord1[3],coord2[3]])
        y2 = np.max([coord1[4],coord2[4]])
    
    newLabel = np.array([np.min([particleNr+1,2]),x1,y1,x2,y2])
    newLabel = ConvertCoordToYOLOLabels(newLabel.reshape(1,5),xdim,ydim)
    return newLabel


def remove_traj_overlap(prediction, overlap_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
   #  output = []
   #  while len(prediction[0]) > 1:
   #      for i in range(len(prediction[0])-1):
   #         overlap = bbox_iou(prediction[0][0],prediction[0][i+1], x1y1x2y2=True)
   #          if overlap > overlap_thres:
   #              prediction[0] = prediction[0][2:]
   #              break
   #        #  else: 
   #         #     prediction[0] = prediction[]
    
   # # output = 
   #  if len(prediction[0]) == 0:
   #      prediction = None
        
   #  return prediction
   
    #  output = []
    #  while len(prediction[0]) > 1:
    #         overlap = bbox_iou(prediction[0][0],prediction[0][1], x1y1x2y2=True)
    #          if overlap > overlap_thres:
    #              prediction[0] = prediction[0][2:]
    #          else: 
    #              output.extend(prediction[0][0])
    #              prediction[0] = prediction[0][2:]
    
    # # output = 
    #  if len(prediction[0]) == 0:
    #      prediction = None
        
    #  return prediction
    
    output = [None for _ in range(len(prediction[0]))]
    while len(prediction[0]) > 0:
        success = True
        for i in range(len(prediction[0])-1):   
           overlap = bbox_iou(prediction[0][0],prediction[0][i+1], x1y1x2y2=True)
           if overlap > overlap_thres:
                prediction[0] = prediction[0][1:]
                prediction[0] = torch.cat((prediction[0][:i], prediction[0][i+1:]))
                success = False
                break
            
        if success:
            if output[0] == None:
                counter = 0
                output[0] = prediction[0][0]
                prediction[0] = prediction[0][1:]
                counter+=1
            else:
               output[counter] = prediction[0][0]#torch.cat((output,prediction[0][0]),axis=1)
               prediction[0] = prediction[0][1:]

    output =[ o for o in output if o is not None]
    #if output[0] is not None:
     #  output = torch.stack(output)
    try:
         output =[output[0].reshape(-1,7)]     
    except:
        output = [None]
    return output

    # for i in range(len(prediction[0])-1):
    #     for j in range(len(prediction[0])-1):
    #         if prediction[0][0] is not None and prediction[0][i+1] is not None:
    #             overlap = bbox_iou(prediction[0][0],prediction[0][i+1], x1y1x2y2=True)
    #             if overlap > overlap_thres:
    #                 prediction[0][0] = None
    #                 prediction[0][i+1] = None
                    
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
