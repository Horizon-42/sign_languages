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
Because the 

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

## 4. Train

## 5. Inference and Submission