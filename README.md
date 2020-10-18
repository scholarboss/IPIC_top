# IPIC_top
NAIC AI+无线通信参赛作品，队伍名称: IPIC-top

---

### 训练方式:
将训练数据 `H_train` 放于data目录，运行以下命令即可开始训练。为了加速训练，训练前部分将batchsize设置为64，60个epoch之后，将batchsize更改为256进一步精确拟合。

step 1:
```python
python Model_train.py --channel 256 --lr 0.0001 --batchsize 64 --loss mse
```
step 2:
```python
python Model_train.py --channel 256 --lr 0.00001 --batchsize 256 --encoder encoder --decoder decoder  --loss mse
```

### 方案说明

当前方案主要参考基于深度学习的无线通信与基于深度学习的图片压缩相关方法，使用gdn模块结合nonlocal residual attention，multi scale等网络结构做深度编码。






















