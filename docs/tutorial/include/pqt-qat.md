
```python
import torch

# 定义浮点模型，其中一些层可以受益于 QAT
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub 将张量从浮点转换为量化
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub 将张量从量化转换为浮点
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x


# 创建模型实例
model_fp32 = M()

# 模型必须设置为训练模式，以便 QAT 逻辑工作
model_fp32.train()

# 附加全局 qconfig，其中包含关于要附加哪种观测器的信息。
# 使用 'fbgemm' 进行服务器端推理，使用 'qnnpack' 进行移动端推理。
# 其他量化配置，如选择对称或非对称量化和 MinMax 或 L2Norm 校准技术，可以在这里指定。
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# 将激活融合到前面的层，在适用的情况下，这需要根据模型体系结构手工完成
model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
                                                   [['conv', 'bn', 'relu']])

# 为 QAT 准备模型。
# 这将在模型中插入观测者和 fake_quants，它们将在校准期间观测权值和激活张量。
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)

# 运行训练循环（没有显示）
training_loop(model_fp32_prepared)

# 将观测到的模型转换为量化模型。这有几件事：
# 量化权重，计算和存储用于每个激活张量的尺度（scale）和偏差（bias）值，
# 在适当的地方融合模块，并用量化实现替换关键算子。
model_fp32_prepared.eval()
model_int8 = torch.quantization.convert(model_fp32_prepared)

# 运行模型，相关的计算将在 int8 中发生
res = model_int8(input_fp32)
```

