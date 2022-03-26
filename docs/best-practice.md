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
    
1. **算子替换**：替换 ``float_model`` 的部分算子，比如 {func}`torch.add` 替换为 {class}`~torch.nn.quantized.FloatFunctional`.{func}`add`，{func}`torch.cat` 替换为 {class}`~torch.nn.quantized.FloatFunctional`.{func}`cat`
1. **算子融合**：使用 {func}`~torch.ao.quantization.fuse_modules.fuse_modules`（用于静态 PTQ） 或者 {func}`~torch.ao.quantization.fuse_modules.fuse_modules_qat` （用于 QAT）融合如下模块序列：

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

可量化模型与浮点模型的算子总会存在一些差异，为了提供更加通用的接口，需要做如下工作。

### 算子替换

{class}`~torch.nn.quantized.FloatFunctional` 算子比普通的 `torch.` 的运算多了后处理操作，比如：

```python
import torch
from torch import Tensor

class FloatFunctional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_post_process = torch.nn.Identity()

    def forward(self, x):
        raise RuntimeError("FloatFunctional 不打算使用 `forward`。请使用下面的操作")

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        """等价于 ``torch.add(Tensor, Tensor)`` 运算"""
        r = torch.add(x, y)
        r = self.activation_post_process(r)
        return r
    ...
```

由于 `self.activation_post_process = torch.nn.Identity()` 是自映射，所以 {meth}`~torch.nn.quantized.FloatFunctional.add` 等价于 {func}`torch.add`。

```{tip}
猜测 {class}`~torch.nn.quantized.FloatFunctional` 算子提供了自定义算子的官方接口。即只需要对 `self.activation_post_process` 赋值即可添加算子的后处理工作。
```

### 算子融合

{func}`~torchvision.models.quantization.utils._fuse_modules` 提供了 {func}`~torch.ao.quantization.fuse_modules.fuse_modules`（用于 PTQ 静态量化） 和 {func}`~torch.ao.quantization.fuse_modules.fuse_modules_qat` （用于 QAT）的统一接口。

```python
from torch.ao.quantization import fuse_modules_qat, fuse_modules


def _fuse_modules(
    model: nn.Module, 
    modules_to_fuse: list[str] | list[list[str]], 
    is_qat: bool | None, 
    **kwargs: Any
):
    if is_qat is None:
        is_qat = model.training
    method = fuse_modules_qat if is_qat else  fuse_modules
    return method(model, modules_to_fuse, **kwargs)
```

下面介绍几个例子。

### 示例

为了说明模块融合的细节，举例如下。

#### 改造 ResNet

`````{panels}
:column: col-lg-12 px-2 py-2
:header: w3-pale-red w3-round-xxlarge w3-wide
:body: w3-text-blue
:footer: w3-pale-blue w3-round-large w3-opacity

浮点模块
^^^
ResNet 的 `BasicBlock` 中需要被改造的部分：

1. `24-26`、`28-29` 与 `32` 需要被 {func}`~torchvision.models.quantization.utils._fuse_modules` 函数融合；
2. `34-35` 被替换为 {class}`~torch.nn.quantized.FloatFunctional` 的 {func}`add_relu` 函数。

```{eval-rst}
.. code-block:: python
    :linenos:
    :emphasize-lines: 24-26,28-29,32,34-35

    from torch import nn


    class BasicBlock(nn.Module):
        expansion: int = 1

        def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            ) -> None:
            super().__init__()
            ... # 此处省略

        def forward(self, x: Tensor) -> Tensor:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
            return out
```
+++
`torchvision.models.resnet` 中的 {class}`~torchvision.models.resnet.BasicBlock`
---

可量化模块
^^^
1. `7` 和 `22` 实现算子替换工作；
2. `12-14`、`16-17`、`20`、`26-27`、`29-30` 实现算子融合工作。

```{eval-rst}
.. code-block:: python
    :linenos:
    :emphasize-lines: 7,12-14,16-17,20,22,26-27,29-30

    from torchvision.models.resnet import BasicBlock


    class QuantizableBasicBlock(BasicBlock):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.add_relu = torch.nn.quantized.FloatFunctional()

        def forward(self, x: Tensor) -> Tensor:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.add_relu.add_relu(out, identity)
            return out
        
        def fuse_model(self, is_qat: Optional[bool] = None) -> None:
            _fuse_modules(self, [["conv1", "bn1", "relu"],
                                ["conv2", "bn2"]], is_qat, inplace=True)
            if self.downsample:
                _fuse_modules(self.downsample,
                            ["0", "1"], is_qat, inplace=True)
````
+++
`torchvision.models.quantization.resnet` 中的 {class}`~torchvision.models.quantization.resnet.QuantizableBasicBlock`
`````

## 量化配置

## 量化训练

## 实战 MobileNetv2
