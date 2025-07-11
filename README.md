[TOC]

# Sign Languages
By: 
Debarghya Barik - 10000738
Dongxu Liu - 10001283

## 1. Data Exploration
- Lack of label 9
- Inverted imges in todo daset.

Check details in data_explore.ipynb.
## 2. Dasetset
Defination in dataset.py.
### 2.1 Add transform for img
```python
        # set trans
        if transform:
            self.transform = transform
        else:
            self.transform = lambda x: x
```
Because the img of todo.pth is inverted, we need to invert img in trian dataset. And to improve the generalization of the model, we add some other random transform.
```python
    transform = v2.Compose(
    [
        # ouput: [1, H, W], get the channel with max intensity
        v2.Lambda(max_channel),
        v2.Lambda(lambda x: 1-x),
        # normalize, ouput = (input-mean)/std, make it zero mean and unit std
        v2.Normalize([0.3992], [0.1779]),

        v2.ToImage(),                                 # convert tensor to image
        # random change brightness, contrast, and saturation
        # change the brightness and contrast
        v2.ColorJitter(brightness=0.5, contrast=0.5),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),          # resize
        # enhance contrast, nonlinear
        v2.RandomEqualize(p=0.8),
        v2.RandomAffine(
            degrees=20,                               # random rotation
        ),
        v2.ToDtype(torch.float32, scale=True),        # convert to tensenor
    ]
)
```

### 2.2 Deal with labels, about the lack 9 issue.
```python
        target_trans = []
        if target_transform:
            target_trans.append(target_transform)
        target_trans.append(Lambda(
            lambda x: x if x <= 8 else x-1))
        self.target_transform = v2.Compose(target_trans)
```
Normally we don't add transform for labels.

## 3. Model
We try on two model.
### 3.1 Simple Convlution
**HandGestureCNN**
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
HandGestureCNN                           [1, 24]                   --
├─Conv2d: 1-1                            [1, 16, 64, 64]           160
├─BatchNorm2d: 1-2                       [1, 16, 64, 64]           32
├─MaxPool2d: 1-3                         [1, 16, 32, 32]           --
├─Conv2d: 1-4                            [1, 32, 32, 32]           4,640
├─BatchNorm2d: 1-5                       [1, 32, 32, 32]           64
├─MaxPool2d: 1-6                         [1, 32, 16, 16]           --
├─Conv2d: 1-7                            [1, 64, 16, 16]           18,496
├─BatchNorm2d: 1-8                       [1, 64, 16, 16]           128
├─MaxPool2d: 1-9                         [1, 64, 8, 8]             --
├─Conv2d: 1-10                           [1, 128, 8, 8]            73,856
├─BatchNorm2d: 1-11                      [1, 128, 8, 8]            256
├─MaxPool2d: 1-12                        [1, 128, 4, 4]            --
├─Conv2d: 1-13                           [1, 256, 4, 4]            295,168
├─BatchNorm2d: 1-14                      [1, 256, 4, 4]            512
├─AdaptiveAvgPool2d: 1-15                [1, 256, 1, 1]            --
├─Linear: 1-16                           [1, 128]                  32,896
├─Dropout: 1-17                          [1, 128]                  --
├─Linear: 1-18                           [1, 24]                   3,096
==========================================================================================
Total params: 429,304
Trainable params: 429,304
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 19.63
==========================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 2.03
Params size (MB): 1.72
Estimated Total Size (MB): 3.77
==========================================================================================
```
### 3.2 Convlution with Residual Block
**EnhancedHandGestureCNN**
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
EnhancedHandGestureCNN                   [1, 24]                   --
├─Sequential: 1-1                        [1, 64, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 64, 64, 64]           640
│    └─BatchNorm2d: 2-2                  [1, 64, 64, 64]           128
│    └─SiLU: 2-3                         [1, 64, 64, 64]           --
│    └─MaxPool2d: 2-4                    [1, 64, 32, 32]           --
├─Sequential: 1-2                        [1, 128, 16, 16]          --
│    └─Conv2d: 2-5                       [1, 128, 32, 32]          73,856
│    └─BatchNorm2d: 2-6                  [1, 128, 32, 32]          256
│    └─SiLU: 2-7                         [1, 128, 32, 32]          --
│    └─MaxPool2d: 2-8                    [1, 128, 16, 16]          --
│    └─ResidualBlock: 2-9                [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-1                  [1, 128, 16, 16]          147,584
│    │    └─BatchNorm2d: 3-2             [1, 128, 16, 16]          256
│    │    └─Conv2d: 3-3                  [1, 128, 16, 16]          147,584
│    │    └─BatchNorm2d: 3-4             [1, 128, 16, 16]          256
├─Sequential: 1-3                        [1, 256, 8, 8]            --
│    └─Conv2d: 2-10                      [1, 256, 16, 16]          295,168
│    └─BatchNorm2d: 2-11                 [1, 256, 16, 16]          512
│    └─SiLU: 2-12                        [1, 256, 16, 16]          --
│    └─MaxPool2d: 2-13                   [1, 256, 8, 8]            --
│    └─ResidualBlock: 2-14               [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-5                  [1, 256, 8, 8]            590,080
│    │    └─BatchNorm2d: 3-6             [1, 256, 8, 8]            512
│    │    └─Conv2d: 3-7                  [1, 256, 8, 8]            590,080
│    │    └─BatchNorm2d: 3-8             [1, 256, 8, 8]            512
├─Sequential: 1-4                        [1, 512, 1, 1]            --
│    └─Conv2d: 2-15                      [1, 512, 8, 8]            1,180,160
│    └─BatchNorm2d: 2-16                 [1, 512, 8, 8]            1,024
│    └─SiLU: 2-17                        [1, 512, 8, 8]            --
│    └─ResidualBlock: 2-18               [1, 512, 8, 8]            --
│    │    └─Conv2d: 3-9                  [1, 512, 8, 8]            2,359,808
│    │    └─BatchNorm2d: 3-10            [1, 512, 8, 8]            1,024
│    │    └─Conv2d: 3-11                 [1, 512, 8, 8]            2,359,808
│    │    └─BatchNorm2d: 3-12            [1, 512, 8, 8]            1,024
│    └─AdaptiveAvgPool2d: 2-19           [1, 512, 1, 1]            --
├─Sequential: 1-5                        [1, 24]                   --
│    └─Flatten: 2-20                     [1, 512]                  --
│    └─Dropout: 2-21                     [1, 512]                  --
│    └─Linear: 2-22                      [1, 256]                  131,328
│    └─SiLU: 2-23                        [1, 256]                  --
│    └─Linear: 2-24                      [1, 24]                   6,168
==========================================================================================
Total params: 7,887,768
Trainable params: 7,887,768
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 682.63
==========================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 10.49
Params size (MB): 31.55
Estimated Total Size (MB): 42.06
==========================================================================================
```
## 4. Train
### 4.1 Dataset init
1. Split annotated dataset to train, val and test with proportion 0.8, 0.15, 0.5.
2. Define and apply the transform of img.
### 4.2 Model Init
Init Model, slect from those two model design. The enhanced one works better.
Deeper network has more ablility to generlize.
### 4.3 Loss Function
**CrossEntropyLoss**
$$
\mathcal{L} = -\log\left(\frac{e^{z_y}}{\sum_{j=1}^{C} e^{z_j}}\right) = -z_y + \log\left( \sum_{j=1}^{C} e^{z_j} \right)
$$
### 4.3 Optimizer
```python
# optimizer, already use dropout, didn't need weight_decay
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE)
```
Update parameters with learning rate.

### 4.4 Learning Rate Schedule
```
# set lr_scheduler, lr = lr * gamma^(epoch//7)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```
Reduce learning rate every 7 epoch.

### 4.5 Validation and save best model
Keep tracking the best accuracy and save the model with best performance.

### 4.6 Early Stop Machnism
Stop learning after the loss didn't change for several times.
See detals in class EarlyStopping, utils.py.

## 5. Inference and Submission
Run model on todo.pth.
inference.py
### 5.1 Transform for todo dataset.
```
transform = v2.Compose(
    [
        v2.Grayscale(),
        v2.Normalize(mean=[0.3896],
                     std=[0.1755]),  # also normalize
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
)
```
Only make image grayscale, and normalize it.
### Post process of label
```python
def idx_to_label(x: int): return x if x <= 8 else x+1

predictions = [idx_to_label(lb) for lb in predictions]
```
Add 1 if the label is bigger than 8.