# 最佳量化实践（PyTorch）

```{admonition} 模型量化动机
- **更少的存储开销和带宽需求**。即使用更少的比特数存储数据，有效减少应用对存储资源的依赖，但现代系统往往拥有相对丰富的存储资源，这一点已经不算是采用量化的主要动机；
- **更快的计算速度**。即对大多数处理器而言，整型运算的速度一般要比浮点运算更快一些（但不总是）；
- **更低的能耗与占用面积**：FP32 乘法运算的能耗是 INT8 乘法运算能耗的 18.5 倍，芯片占用面积则是 INT8 的 27.3 倍，而对于芯片设计和 FPGA 设计而言，更少的资源占用意味着相同数量的单元下可以设计出更多的计算单元；而更少的能耗意味着更少的发热，和更长久的续航。
- **尚可接受的精度损失**。即量化相当于对模型权重引入噪声，所幸 CNN 本身对噪声不敏感（在模型训练过程中，模拟量化所引入的权重加噪还有利于防止过拟合），在合适的比特数下量化后的模型并不会带来很严重的精度损失。按照 [GluonCV](https://cv.gluon.ai/build/examples_deployment/int8_inference.html) 提供的报告，经过 INT8 量化之后，ResNet50_v1 和 MobileNet1.0_v1 在 ILSVRC2012 数据集上的准确率仅分别从 77.36%、73.28% 下降为 76.86%、72.85%。
- **支持 INT8 是一个大的趋势**。即无论是移动端还是服务器端，都可以看到新的计算设备正不断迎合量化技术。比如 NPU/APU/AIPU 等基本都是支持 INT8（甚至更低精度的 INT4）计算的，并且有相当可观的 TOPs，而 Mali GPU 开始引入 INT8 dot 支持，Nvidia 也不例外。除此之外，当前很多创业公司新发布的边缘端芯片几乎都支持 INT8 类型。
```

量化（包括 PTQ （静态）和 QAT）的整个流程如下：

```{rubric} 模块融合阶段
```

将已有模型 ``float_model`` 改造为可量化模型 ``quantizable_model``，需要做如下工作：
    
1. 替换 ``float_model`` 的部分算子，比如 {func}`torch.add` 替换为 {class}`~torch.nn.quantized.FloatFunctional`.{func}`add`，{func}`torch.cat` 替换为 {class}`~torch.nn.quantized.FloatFunctional`.{func}`cat`
1. 使用 {func}`~torch.ao.quantization.fuse_modules.fuse_modules`（用于静态 PTQ） 或者 {func}`~torch.ao.quantization.fuse_modules.fuse_modules_qat` （用于 QAT）融合如下模块序列：

    - conv, bn
    - conv, bn, relu
    - conv, relu
    - linear, bn
    - linear, relu

1. 在模型 ``float_model`` 的开头和结尾分别插入 {class}`~torch.ao.quantization.stubs.QuantStub` 和 {class}`~torch.ao.quantization.stubs.DeQuantStub`
1. 将 {class}`torch.nn.ReLU6` （如果存在的话）替换为 {class}`torch.nn.ReLU`

```{rubric} QAT 配置和训练阶段
```

将可量化模型 ``quantizable_model`` 转换为 QAT 模型 ``qat_model``：

1. `quantizable_model.qconfig` 赋值为 {class}`~torch.ao.quantization.qconfig.QConfig` 类；
1. 使用 {func}`~torch.ao.quantization.quantize.prepare_qat` 函数，将其转换为 QAT 模型；
1. 像普通的浮点模型一样训练 QAT 模型；
1. {func}`torch.jit.save` 函数保存训练好的 QAT 模型。

## 模块融合

### 算子替换