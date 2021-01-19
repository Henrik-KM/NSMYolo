from __future__ import division
# runfile('C:/Users/ccx55/OneDrive/Documents/GitHub/Phd/Biosensing---nanochannel-project/NSMYOLO/train.py',args='--data_config config/customNSM.data --model_def config/yolov3-customNSM.cfg --n_cpu 0 --batch_size=2')
# runfile('C:/Users/ccx55/OneDrive/Documents/GitHub/NSMYOLO/train.py',args='--data_config config/customNSM.data --model_def config/yolov3-customNSM.cfg --n_cpu 0 --batch_size=32 --pretrained_weights weights/yolov3_ckpt_8.pth --epochs 9')
from models import *
from utils.logger import *
from utils.utils import *
from utils.datasetsNSMTest import *
from utils.parse_config import *


from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import tensorflow as tf
config = tf.compat.v1.ConfigProto() #Use to fix OOM problems with unet
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
unet = tf.keras.models.load_model('../../input/network-weights/unet-1-dec-1415.h5',compile=False)

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False,totalData=500)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        if len(imgs) == batch_size:
            imgs = torch.stack(imgs)
        try:
            imgs = Variable(imgs.type(Tensor), requires_grad=False)
        except:
            imgs = torch.stack(imgs)
            imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=23, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=4, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=False, multiscale=opt.multiscale_training,totalData = 500,unet=unet)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]


    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            # plt.imshow(imgs[0,0,:,:],aspect='auto')
            
            # if targets is not None:
            #      # Rescale boxes to original image
            #      targets = rescale_boxes(targets, opt.img_size, img.shape[:2])
            #      unique_labels = targets[:, -1].cpu().unique()
            #      n_cls_preds = len(unique_labels)
            #      bbox_colors = random.sample(colors, n_cls_preds)
            #      for x1, y1, x2, y2, conf, cls_conf, cls_pred in targets:
            #             print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
        
            #             box_w = x2 - x1
            #             box_h = y2 - y1
        
            #             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            #             # Create a Rectangle patch
            #             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            #             # Add the bbox to the plot
            #             ax.add_patch(bbox)
                     
            batches_done = len(dataloader) * epoch + batch_i

           # if len(imgs) == opt.batch_size:
           #     imgs = torch.stack(imgs)
            try:
                imgs = Variable(imgs.to(device))
            except:
                imgs = torch.stack(imgs) 
                imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            #model=model.float()
            #print(targets)
            #plt.imshow(imgs[0,0,:,:].cpu(),aspect='auto')
            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if False:#epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            # for i, c in enumerate(ap_class):
            #     ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"weights/yolov3_ckpt_%d.pth" % epoch)
