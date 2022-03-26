# 最佳量化实践（PyTorch）

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