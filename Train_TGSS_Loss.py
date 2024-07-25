import os
import argparse

from tqdm import tqdm
import json

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    CropForegroundd,
    RandAffined,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureChannelFirstd,
    SpatialPadd,
    ToTensord,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianSmoothd,
    RandRotated,
    AsDiscreted,
    ResampleToMatchd,
    ConcatItemsd,
    ConcatItemsd,
    ScaleIntensityd
)
import monai
from monai.metrics import DiceMetric


from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset,
    decollate_batch,
    set_track_meta,
)
from monai.visualize import matshow3d
import matplotlib.pyplot as plt
import pandas as pd


import torch
import numpy as np
from time import sleep

from datetime import datetime

from models import CustomUNet
from TGSS_Loss import TcgeConfig_pytorch_3D, TcgeSs_pytorch_3D

def custom_print(*args, **kwargs):
    # Get the current datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Construct the message with the datetime
    formatted_message = f"{current_time} - {' '.join(map(str, args))}"
    # Use the built-in print function to display the message
    print(formatted_message, **kwargs)

# Train Function
def train_Multiple_Annotators(
    model,
    data_in,
    loss,
    optim,
    max_epochs,
    model_dir,
    device,
    name,
    test_interval=1,
    start_epoch=0,
):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    save_metric_test_per_class = []
    save_iou_test_per_class = []
    train_loader, test_loader = data_in

    data_classes = [
        "csPCa"
    ]

    for epoch in range(start_epoch, max_epochs):
        print("-" * 50)
        custom_print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tepoch.set_description(f"{current_time} - Epoch {epoch+1}")
                train_step += 1

                volume = batch_data["image"]
                label = batch_data["label"]
                # print(volume.shape)
                #print(label.shape)
                volume, labels = (volume.to(device), label.to(device))

                optim.zero_grad()
                outputs = model(volume)
                #print(outputs.shape)
                #print(labels.shape)
                train_loss = loss(outputs, labels,device)
                #print(train_loss)
                train_loss.backward()
                optim.step()

                train_epoch_loss += train_loss.item()
                #print(labels.shape)
                #labels_convert = labels[:,0:2]
                labels_list = decollate_batch(labels)
                labels_convert = [
                   label_tensor[0:2] for label_tensor in labels_list
                ]
                
                outputs = outputs[:,0:2] #First annotator 
                output_list = decollate_batch(outputs)
                output_convert = [
                    post_pred(output_tensor) for output_tensor in output_list
                ]
                # print(len(output_convert))
                # print(output_convert[0].shape)
                # print(len(labels_convert))
                # print(labels_convert[0].shape)


                dice_metric(y_pred=output_convert, y=labels_convert)
                iou_metric(y_pred=output_convert, y=labels_convert)

                tepoch.set_postfix(
                    loss=train_loss.item(),
                    dice_score=dice_metric.aggregate(reduction="mean").item(),
                )
                sleep(0.01)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(model_dir, name + "_last_checkpoint.pth"),
            )

            print("-" * 20)

            train_epoch_loss /= train_step
            print(f"Epoch_loss: {train_epoch_loss:.4f}")
            save_loss_train.append(train_epoch_loss)
            np.save(os.path.join(model_dir, name + "_loss_train.npy"), save_loss_train)

            epoch_metric_train = dice_metric.aggregate(reduction="mean").item()
            dice_metric.reset()

            print(f"Epoch_metric: {epoch_metric_train:.4f}")

            iou_metric_train = iou_metric.aggregate(reduction="mean").item()
            iou_metric.reset()

            print(f"IoU_metric: {iou_metric_train:.4f}")

            save_metric_train.append(epoch_metric_train)
            np.save(
                os.path.join(model_dir, name + "_metric_train.npy"), save_metric_train
            )

            if (epoch + 1) % test_interval == 0:

                model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0
                    test_metric = 0
                    epoch_metric_test = 0
                    test_step = 0

                    for test_data in tqdm(test_loader):

                        test_step += 1

                        test_volume = test_data["image"]
                        test_label = test_data["label"]

                        test_volume, test_label = (
                            test_volume.to(device),
                            test_label.to(device),
                        )

                        test_outputs = sliding_window_inference(
                            test_volume,patch_size, 8, model, overlap=0.5
                        )

                        test_loss = loss(test_outputs, test_label,device)
                        test_epoch_loss += test_loss.item()

                        labels_list = decollate_batch(test_label)
                        labels_convert = [
                           label_tensor[0:2] for label_tensor in labels_list
                        ]

                        test_outputs = test_outputs[:,0:2]
                        output_list = decollate_batch(test_outputs)
                        output_convert = [
                            post_pred(output_tensor) for output_tensor in output_list
                        ]

                        dice_metric(y_pred=output_convert, y=labels_convert)
                        iou_metric(y_pred=output_convert, y=labels_convert)

                    test_epoch_loss /= test_step
                    print(f"test_loss_epoch: {test_epoch_loss:.4f}")
                    save_loss_test.append(test_epoch_loss)
                    np.save(
                        os.path.join(model_dir, name + "_loss_test.npy"), save_loss_test
                    )

                    epoch_metric_test = dice_metric.aggregate(reduction="mean").item()

                    print(f"test_dice_epoch: {epoch_metric_test:.4f}")

                    dice_scores = {
                        key: value
                        for key, value in zip(
                            data_classes,
                            dice_metric.aggregate(reduction="mean_batch").tolist(),
                        )
                    }
                    print("test_dice_epoch_per_class:")
                    for key, value in dice_scores.items():
                        print(f"\t{key}: {value}")
                    save_metric_test_per_class.append(
                        np.array(list(dice_scores.values()))
                    )
                    np.save(
                        os.path.join(model_dir, name + "_metric_test_per_class.npy"),
                        save_metric_test_per_class,
                    )

                    iou_metric_test = iou_metric.aggregate(reduction="mean").item()

                    print(f"test_iou_epoch: {iou_metric_test:.4f}")

                    iou_scores = {
                        key: value
                        for key, value in zip(
                            data_classes,
                            iou_metric.aggregate(reduction="mean_batch").tolist(),
                        )
                    }
                    print("test_iou_epoch_per_class:")
                    for key, value in iou_scores.items():
                        print(f"\t{key}: {value}")
                    save_iou_test_per_class.append(np.array(list(iou_scores.values())))
                    np.save(
                        os.path.join(model_dir, name + "_iou_test_per_class.npy"),
                        save_iou_test_per_class,
                    )

                    iou_metric.reset()
                    save_metric_test.append(epoch_metric_test)
                    np.save(
                        os.path.join(model_dir, name + "_metric_test.npy"),
                        save_metric_test,
                    )
                    dice_metric.reset()
                    if epoch_metric_test > best_metric:
                        best_metric = epoch_metric_test
                        best_metric_epoch = epoch + 1
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(model_dir, name + "_best_metric_model.pth"),
                        )

                    print(
                        f"current epoch: {epoch + 1} current mean dice: {epoch_metric_test:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
    )

path = "./prostate158_train/"
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "valid.csv"))

# Keep Only images with 2 annotators
train_df = train_df.dropna(subset=["adc_tumor_reader2"])
test_df = test_df.dropna(subset=["adc_tumor_reader2"])

# Add path to all images
train_df["t2"] = train_df["t2"].apply(lambda x: os.path.join(path, x))
train_df["adc"] = train_df["adc"].apply(lambda x: os.path.join(path, x))
train_df["dwi"] = train_df["dwi"].apply(lambda x: os.path.join(path, x))
train_df["adc_tumor_reader1"] = train_df["adc_tumor_reader1"].apply(lambda x: os.path.join(path, x))
train_df["adc_tumor_reader2"] = train_df["adc_tumor_reader2"].apply(lambda x: os.path.join(path, x))

test_df["t2"] = test_df["t2"].apply(lambda x: os.path.join(path, x))
test_df["adc"] = test_df["adc"].apply(lambda x: os.path.join(path, x))
test_df["dwi"] = test_df["dwi"].apply(lambda x: os.path.join(path, x))
test_df["adc_tumor_reader1"] = test_df["adc_tumor_reader1"].apply(lambda x: os.path.join(path, x))
test_df["adc_tumor_reader2"] = test_df["adc_tumor_reader2"].apply(lambda x: os.path.join(path, x))


img_columns = ["t2", "dwi", "adc"]  # ,"adc","dwi"]
label_columns = ["label_1","label_2"]

mode = [
        "bilinear",
        "bilinear",
        "bilinear",
        "nearest",
        "nearest"
    ]


train_files = [
        {"t2": t2, "adc": adc, "dwi": dwi, "label_1": label_1, "label_2": label_2}
        for t2, adc, dwi, label_1, label_2 in zip(
            train_df["t2"].values,
            train_df["adc"].values,
            train_df["dwi"].values,
            train_df["adc_tumor_reader1"].values,
            train_df["adc_tumor_reader2"].values
        )
    ]
test_files = [
    {"t2": t2, "adc": adc, "dwi": dwi, "label_1": label_1, "label_2": label_2}
        for t2, adc, dwi, label_1, label_2 in zip(
            test_df["t2"].values,
            test_df["adc"].values,
            test_df["dwi"].values,
            test_df["adc_tumor_reader1"].values,
            test_df["adc_tumor_reader2"].values
        )
    ]

#Define Image transforms
pixdim = (0.5, 0.5, 1)
patch_size = (32,32,32)

def add_new_axis(data):
    return torch.unsqueeze(data, 0)


train_transforms = Compose(
    [
        LoadImaged(keys=img_columns + label_columns, reader="NibabelReader", image_only=True),
        EnsureChannelFirstd(keys=img_columns + label_columns),
        AsDiscreted(
            keys=label_columns, threshold=1,to_onehot=2
        ),  # Convert values greater than 1 to 1
        Spacingd(keys=img_columns + label_columns, pixdim=pixdim, mode=mode),
        ResampleToMatchd(
            keys=["adc", "dwi", "label_1","label_2"],
            key_dst="t2",
            mode=("bilinear", "bilinear", "nearest","nearest"),
        ),
        ScaleIntensityd(keys=img_columns, minv=0.0, maxv=1.0),
        Orientationd(keys=img_columns + label_columns, axcodes="RAS"),
        CropForegroundd(keys=img_columns + label_columns, source_key="t2"),
        SpatialPadd(keys=img_columns + label_columns, spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=img_columns + label_columns,
            label_key="label_1",
            spatial_size=patch_size,
            pos=2,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        ConcatItemsd(keys=img_columns, name="image", dim=0),
        ConcatItemsd(keys=label_columns, name="label", dim=0),
        RandRotated(
            keys=["image", "label"],
            prob=0.2,
            range_x=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            range_y=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            range_z=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            mode=["bilinear", "nearest"],
        ),
        RandAffined(keys=["image", "label"], scale_range=(0.7, 1.4), prob=0.2, mode=["bilinear", "nearest"]),
        RandGaussianNoised(keys="image", prob=0.1, mean=0, std=0.1),
        RandGaussianSmoothd(keys="image", prob=0.1, sigma_x=(0.5, 1)),
        RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.75, 1.25)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
        ToTensord(keys=["image", "label"]),
        #Transposed(keys=["label"], indices=[1, 2, 3, 0]),
        #Lambdad(keys=["label"], func=add_new_axis),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=img_columns + label_columns, reader="NibabelReader", image_only=True),
        EnsureChannelFirstd(keys=img_columns + label_columns),
        AsDiscreted(
            keys=label_columns, threshold=1,to_onehot=2
        ),  # Convert values greater than 1 to 1
        Spacingd(keys=img_columns + label_columns, pixdim=pixdim, mode=mode),
        ResampleToMatchd(
            keys=["adc", "dwi", "label_1","label_2"],
            key_dst="t2",
            mode=("bilinear", "bilinear", "nearest","nearest"),
        ),
        ScaleIntensityd(keys=img_columns, minv=0.0, maxv=1.0),
        Orientationd(keys=img_columns + label_columns, axcodes="RAS"),
        CropForegroundd(keys=img_columns + label_columns, source_key="t2"),
        SpatialPadd(keys=img_columns + label_columns, spatial_size=patch_size),
        ConcatItemsd(keys=img_columns, name="image", dim=0),
        ConcatItemsd(keys=label_columns, name="label", dim=0),
        
        ToTensord(keys=["image", "label"]),
        #Transposed(keys=["label"], indices=[1, 2, 3, 0]),
        #Lambdad(keys=["label"], func=add_new_axis),
    ]
)
## Creating Datasets

train_ds = CacheDataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

test_ds = CacheDataset(data=test_files, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print(train_ds[0][1]["label"].shape) 

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Working on device: {device}")

model = CustomUNet(
    spatial_dims=3,
    in_channels=3,
    out_channels=2,
    channels=[16, 32, 64, 128, 256, 512],
    strides=[2, 2, 2, 2, 2],
    num_res_units=4,
    norm="batch",
    act="PReLU",
    dropout=0.15,
    n_scorers=2,
).to(device)
torch.backends.cudnn.benchmark = True
config = TcgeConfig_pytorch_3D(num_annotators=2, num_classes=2)
loss_function = TcgeSs_pytorch_3D(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-5)

data_in = (train_loader, test_loader)
model_dir = "./model_results"


#post_label = monai.transforms.AsDiscrete(to_onehot=2)
post_pred = monai.transforms.AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
iou_metric = monai.metrics.MeanIoU(
    include_background=False, reduction="mean", get_not_nans=False
)

train_Multiple_Annotators(
    model=model,
    data_in=(train_loader, test_loader),
    loss=loss_function,
    optim=optimizer,
    max_epochs=1000,
    model_dir=model_dir,
    device=device,
    name="Prostate158_multiple_labelers",
    test_interval=1,
    start_epoch=0,
)