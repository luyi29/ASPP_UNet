import os
import time
import datetime
from src11 import deeplabv3_resnet50,deeplabv3_effi
import torch
import matplotlib.pyplot as plt
from src import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T
import model_
import SimpleITK as sitk
import cv2

def create_model(num_classes):
    if args.model_name == 'unet':
        #可以更换一些网络
        model = UNet(in_channels=1, num_classes=num_classes, base_c=32)
    else:
        # model = model_.efficientnetv2_base(num_classes)
        model = deeplabv3_effi(aux=False, num_classes=num_classes)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1


    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DriveDataset(args.data_path,
                                 train=True)

    val_dataset = DriveDataset(args.data_path,
                               train=False)

    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn,drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                            collate_fn=val_dataset.collate_fn,drop_last=True)

    model = create_model(num_classes=num_classes)
    model.to(device)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print('-'*10)
        print('[epoch]: ',str(epoch))
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice ,loss_test= evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print('test_loss',loss_test)
        print('mean_loss',mean_loss)
        print('lr',lr)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"test_loss: {loss_test:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    model.eval()
    with torch.no_grad():
        i = 0
        
        for val_data_img, val_data_label in val_loader:
            i = i+1
            val_data_img, val_data_label = val_data_img.to(device).float(), val_data_label.to(device).long()
            if args.model_name == 'unet':
                val_data_img_1 = val_data_img.unsqueeze(1)
            val_outputs = model(val_data_img_1.to(device))

            b =val_data_label.cpu()[0]
            b = torch.unsqueeze(b, 0)
            c = torch.argmax(val_outputs['out'], dim=1).detach().cpu()[0]
            c = torch.unsqueeze(c, 0)
            b=dan_to_duo(b)
            c=dan_to_duo(c)
            b = torch.permute( b,(1,2,0))
            c = torch.permute( c,(1,2,0))
            cv2.imwrite('E:\\file\\yansan\\1\\result\\'+str(i)+'b.jpg', b.numpy())
            cv2.imwrite('E:\\file\\yansan\\1\\result\\'+str(i)+'c.jpg', c.numpy())
def dan_to_duo(data_img):
    #控制输出图片的颜色
    r = [0,15,96,163,63,195,164,222,127]
    g = [0,15,71,47,137,114,166,165,240]
    b = [0,143,31,151,188,79,95,190,255]
    r_d = torch.zeros((464,880))
    g_d = torch.zeros((464,880))
    b_d = torch.zeros((464,880)) 
    for i in range(9):      
        r_d[data_img[0]==i] =r[i]
        g_d[data_img[0]==i] =g[i]
        b_d[data_img[0]==i] =b[i]
    r_d = torch.unsqueeze(r_d, 0)
    g_d = torch.unsqueeze(g_d, 0)
    b_d = torch.unsqueeze(b_d, 0)
    rgb = torch.cat((r_d,g_d,b_d),0)
    return rgb

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default='E:\\file\\start_data_5_22\\image', help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=8, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--model-name", default='unet', type=str,
                        help="unet or effi")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
