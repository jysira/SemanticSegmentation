# PCB 语义分割压缩实践

在 PCB_defect 根目录、industrial_vision 环境运行。数据为 data/pcb_semantic_clahe，权重默认 runs/02_segmentation/clahe/pcb-2/weights/best.pt。先确认这些文件符合本次实验。压缩结果与记录放到 runs/05_compression。

当前验证、测试标签曾依据模型预测修改，重评反映修标后的一致性。要报告独立泛化，另准备人工独立核验且冻结的测试集。校准与调参使用训练/验证数据，不能使用测试集。

## 1. FP32 与 FP16

```cmd
python scripts/semantic_compression.py evaluate --split val --output runs/05_compression/fp32
python scripts/semantic_compression.py fp16 --batch 1 --output runs/05_compression/fp16
```

export.json 中记录实际 engine 路径。导出可能把文件写在权重旁，及时将各精度产物复制到不同命名位置，避免后续 INT8 导出覆盖同名 engine。用实际路径评估。

```cmd
python scripts/semantic_compression.py evaluate --weights "实际FP16.engine路径" --batch 1 --split val --output runs/05_compression/fp16_eval
```

## 2. INT8 训练后量化

```cmd
python scripts/semantic_compression.py int8 --batch 1 --output runs/05_compression/int8
python scripts/semantic_compression.py evaluate --weights "实际INT8.engine路径" --batch 1 --split val --output runs/05_compression/int8_eval
```

TensorRT版本、GPU与量化接口必须兼容。此入口调用安装版本的Ultralytics导出器，报错时保留完整日志检查环境。data参数指向当前CLAHE数据，fraction=1使用全部训练图校准。若框架版本的实际校准集合不同，必须检查导出日志，确保只读取训练集。校准集应覆盖六类缺陷、正常背景、亮度和板型，且使用与实际部署相同的CLAHE参数与颜色处理。

## 3. 非结构化剪枝基线

```cmd
python scripts/semantic_compression.py prune --amount 0.1 --output runs/05_compression/prune10
python scripts/semantic_compression.py evaluate --weights runs/05_compression/prune10/pruned_sparse.pt --output runs/05_compression/prune10_eval
```

脚本按卷积权重绝对值做全局L1剪枝，amount表示目标置零比例。先试10%，记录各类IoU，再比较20%。网络通道与张量尺寸不变，稠密推理通常不能自动提速，文件也未必缩小。这是稀疏化实践，不应写成已实现结构压缩或加速。恢复训练时零权重可能重新生长，本入口不宣称掩码约束微调。

真正结构化剪枝要按依赖关系同步删除卷积、BN与拼接分支通道，并微调、导出和重新测速。当前脚本不直接裁通道，以免破坏YOLO多分支结构。若需要结构化压缩，可先用较小学生模型与蒸馏作为另一条可控路线。

## 4. 教师与学生蒸馏

先训练并验证一个相同类别顺序的语义分割教师，例如yolo26s-sem。教师与学生均只用训练集训练。

```cmd
yolo semantic train model=yolo26s-sem.pt data="D:/pycharm/pythonProject/VisualProjects/SemanticSegmentation/data/pcb_semantic_clahe/dataset.yaml" imgsz=640 batch=2 epochs=100 patience=30 workers=0 project="D:/pycharm/pythonProject/VisualProjects/SemanticSegmentation/runs/05_compression/teacher" name=pcb
python scripts/semantic_compression.py distill --weights yolo26n-sem.pt --teacher runs/05_compression/teacher/pcb/weights/best.pt --epochs 20 --batch 2 --alpha 0.3 --temperature 3 --output runs/05_compression/distill
python scripts/semantic_compression.py evaluate --weights runs/05_compression/distill/student_best.pt --output runs/05_compression/distill_eval
```

当前教师也可以使用你已有pcb-2权重跑通机制，但同规模教师不保证收益。正式对照优先选择验证效果更好的教师。

脚本重建七类学生头并迁移兼容参数，教师冻结。损失由带前景权重的交叉熵与温度KL散度构成，alpha控制软标签影响，温度T让类别分布平滑，KL乘T平方补偿梯度尺度。额外监督辅助头。RGB输入与教师/学生保持一致。训练使用256前景采样图块，验证整图640。对照学生必须使用相同采样与损失流程，可用alpha=0再跑一次，以区分训练流程收益与蒸馏收益。

此为自定义训练循环，尚需完整训练验证效果，不能把它等同官方Ultralytics内置蒸馏支持。学生按排除背景的验证mIoU保存best。不同模型类别顺序必须一致。

## 5. 量化掉精度的原因与对策

1. 校准图像不具代表性。小缺陷和极端光照缺失会让激活范围估计偏差。覆盖真实板型和六类缺陷，检查校准读取的文件列表。
2. 预处理不一致。BGR/RGB、0至1归一化、CLAHE、缩放和padding不同都会引入偏差。用一张固定图片逐阶段检查PyTorch与engine输入。
3. 量化范围被异常值拉大。INT8可表达离散值有限，少量极端激活导致大部分数值分辨率下降。改进校准样本、检查异常图像，必要时采用量化感知训练或混合精度。
4. 敏感层误差累积。分割头、细节特征和上采样前的logits误差可能改变边界。工具支持时对敏感层保留FP16/FP32，并重新评估各类IoU。
5. 小目标更敏感。轻微边界位移对几十像素的区域影响很大。看面积分桶的IoU、漏检和边界，必要时比较更大输入，计入延迟成本。
6. 导出与后处理差异。logits resize、argmax顺序、插值和ignore像素会改变结果。保持相同输入尺寸和最近邻类别图处理，先核对FP32 ONNX，再检查FP16和INT8。
7. 不同硬件与环境。engine的兼容性和内核选择依赖GPU、TensorRT与CUDA。应在目标设备构建并记录环境，避免跨机器直接复用后解释性能差异。
8. 训练模型本身不稳定。标注噪声或泛化不足会掩盖量化收益。先固定可信的FP32基线。PTQ仍不达标时再考虑QAT，QAT需要匹配量化后端的训练工具链，本脚本未实现。

## 6. 统一验收表

每个模型记录权重、数据标签版本、输入、batch、精度模式、每类IoU、缺陷mIoU、文件大小、参数量、峰值显存、P50/P95延迟和吞吐量。用相同硬件与前后处理，先预热，GPU计时同步。压缩率与速度提升分别计算，参数置零不能代表实际延迟下降。

先用验证集选择方案，预先约定可接受的缺陷mIoU下降（例如1个百分点作为起始业务假设）和最差类别下限，再在独立测试集报告。至少单独观察残铜、断路和小面积区域。本文没有填入未经运行验证的量化、剪枝或蒸馏收益数字。
