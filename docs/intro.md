# 概述

参考：[量化](https://pytorch.org/docs/master/quantization.html#)

**量化** 是指以比浮点精度更低的位宽执行计算和存储张量的技术。量化模型对具有整数而不是浮点值的张量执行部分或全部运算。这允许在许多硬件平台上使用更紧凑的模型表示和高性能矢量化运算。与典型的 FP32 模型相比，PyTorch 支持 INT8 量化，允许模型尺寸减少 4 倍，内存带宽需求减少 4 倍。INT8 计算的硬件支持通常比 FP32 计算快 2 到 4 倍。量化主要是加速推理的技术，只支持量化算子的前向传播。

PyTorch 支持多种方法来量化深度学习模型。在大多数情况下，模型是在 FP32 中训练的，然后将模型转换为 INT8。此外，PyTorch 还支持量化感知训练（quantization aware training，简称 QAT），它使用伪量化模块对正向和反向传播中的量化误差进行建模。请注意，整个计算均是浮点运算。在量化感知训练结束时，PyTorch 提供了转换函数，将训练后的模型转换成精度（precision）较低的模型。

在低层接口，PyTorch 提供了表示量化张量（quantized tensor）及其运算的方法。它们可以用来直接构建模型，以较低的精度执行全部或部分计算。也提供高层 API，结合 FP32 模型转换的典型工作流程，以最小的 accuracy 损失降低精度。

量化要求用户了解三个概念：

- 量化配置（`Qconfig`）：使用 {class}`~torch.ao.quantization.qconfig.QConfig` 指定权重和激活的量化方案。
- 后端：提供支持量化的内核，通常使用不同的数值。
- 量化引擎（`torch.backends.quantization.engine`）：当执行量化模型时，`qengine` 指定执行时使用哪个后端。重要的是要确保 `qengine` 与 `Qconfig` 一致。

## 量化 API 概述

PyTorch 提供了两种不同的量化模式：Eager 模式量化和 FX 图模式量化。

Eager 模式量化是 beta 特性。用户需要进行融合，并手动指定量化和反量化发生的位置，而且它只支持模块而不支持函数。FX 图模式量化是 PyTorch 中新的自动量化框架，目前它是原型（prototype）特性。它通过添加对函数的支持和量化过程的自动化，对 Eager 模式量化进行了改进，尽管人们可能需要重构模型，以使模型与 FX Graph 模式量化兼容（通过 {mod}`torch.fx` 符号可追溯（symbolically traceable））。

```{note}
FX 图模式量化预计不会在任意可能不是 symbolically traceable 的模型工作，将其集成到域库 torchvision 和用户将能够量化模型类似于支持域的库与 FX 图模式量化。对于任意的模型，提供一般的指导方针，但要让它实际工作，用户可能需要熟悉 {mod}`torch.fx`，特别是如何使模型具有符号可追溯性。

新用户的量化鼓励尝试 FX 图模式量化首先，如果它不工作，用户可以尝试遵循[使用 FX 图模式量化](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)的指导方针或回落到 Eager 模式量化。
```

支持三种类型的量化：

1. 动态量化（dynamic quantization）：通过读取/存储在浮点数中的激活（activation）以量化权重（weight），并量化用于计算。
2. 静态量化（static quantization）：权重量化，激活量化，需要训练后校准。
3. 静态量化感知训练（static quantization aware training）：权重量化，激活量化，训练过程中的量化数值建模。

### Eager 模式量化

分为动态量化、PTQ、QAT。

#### 动态量化

这是最简单的量化形式，其中权值（weight）提前量化，但在推理期间动态量化激活（activation）。这用于模型执行时由从内存中加载权重控制而不是计算矩阵乘法的情况。这适用于小批量的 LSTM 和 Transformer 类型模型。

```{rubric} 示意图
```

```
# 原始 model
# 所有的张量和计算都是浮点数
previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                 /
linear_weight_fp32

# 动态量化 model
# linear 和 LSTM 权重是 int8
previous_layer_fp32 -- linear_int8_w_fp32_inp -- activation_fp32 -- next_layer_fp32
                     /
   linear_weight_int8
```

```{rubric} 示例
```

```python
import torch

# 定义浮点模型
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x


# 创建模型实例
model_fp32 = M()
# 创建量化模型实例
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # 原始模型
    {torch.nn.Linear},  # 一组要动态量化的层
    dtype=torch.qint8)  # 量化权重的目标 dtype

# 运行模型
input_fp32 = torch.randn(4, 4, 4, 4)
res = model_int8(input_fp32)
```

要了解更多关于动态量化的信息，请参阅[动态量化教程](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)。

### （原型）FX 图模式量化

FX 图（Graph）模式支持的量化类型可以分为两种方式：

1. PTQ：训练后进行量化，根据样本校准数据计算量化参数
2. QAT：在训练过程中模拟量化，以便利用训练数据与模型一起学习量化参数

然后这两种可能包括以下任何一种或所有的类型：

- 仅权重量化（Weight Only Quantization）：只有权重是静态量化的
- 动态量化（Dynamic Quantization）：权重静态量化，激活动态量化
- 静态量化（Static Quantization）：权重和激活都是静态量化的

这两种分类方式是独立的，所以理论上我们可以有 6 种不同的量化方式。

FX 图模式量化中支持的量化类型有：

- Post Training Quantization

    - Weight Only Quantization
    - Dynamic Quantization
    - Static Quantization

- Quantization Aware Training

    - Static Quantization

在训练后量化中有多种量化类型（仅权重、动态和静态），配置是通过 `qconfig_dict ` （`prepare_fx` 函数的参数）完成的。

```{rubric} 示例
```

```python
import torch.quantization.quantize_fx as quantize_fx
import copy

model_fp = UserModel(...)

#
# post training dynamic/weight_only quantization
#

# we need to deepcopy if we still want to keep model_fp unchanged after quantization since quantization apis change the input model
model_to_quantize = copy.deepcopy(model_fp)
model_to_quantize.eval()
qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}
# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
# no calibration needed when we only have dynamici/weight_only quantization
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)

#
# post training static quantization
#

model_to_quantize = copy.deepcopy(model_fp)
qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}
model_to_quantize.eval()
# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
# calibrate (not shown)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)

#
# quantization aware training for static quantization
#

model_to_quantize = copy.deepcopy(model_fp)
qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('qnnpack')}
model_to_quantize.train()
# prepare
model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_dict)
# training loop (not shown)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)

#
# fusion
#
model_to_quantize = copy.deepcopy(model_fp)
model_fused = quantize_fx.fuse_fx(model_to_quantize)
```

有关 FX 图模式量化的更多信息，请参阅以下教程：

* [User Guide on Using FX Graph Mode Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)
* [FX Graph Mode Post Training Static Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
* [FX Graph Mode Post Training Dynamic Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)

## 量化 API 参考

[量化 API 参考](https://pytorch.org/docs/master/quantization-support.html)包含了量化 API 的文档，例如量化传递、量化张量操作以及支持的量化模块和函数。

## 量化张量

PyTorch 支持逐张量和逐通道的非对称线性量化。逐张量意味着张量内的所有值都以相同的方式缩放。逐通道意味着对于每个维度，通常是张量的通道维度，张量中的值被不同的值缩放和偏移（实际上，缩放和偏移变成了向量）。这使得将张量转换为量化值的误差更小。

映射是通过使用转换浮点张量来执行的：

$$
Q(x, \text{scale}, \text{zero_point}) = \text{round}(\frac{x}{\text{scale}} + \text{zero_point})
$$

请注意，我们确保浮点数中的零点在量化后不会出现错误，从而确保 padding 之类的运算不会导致额外的量化误差。

为了在 PyTorch 中进行量化，需要能够用张量表示量化的数据。量化张量允许存储量化数据（表示为 int8/uint8/int32）以及量化参数，如 `scale` 和 `zero_point`。量化张量（Quantized Tensor）除了允许以量化格式对数据进行序列化外，还允许许多有用的运算使量化算术变得简单。

## 原生支持的后端

今天，PyTorch 支持以下后端来高效地运行量化算子：

- 支持 AVX2 或更高版本的 x86 CPU（如果没有 AVX2，一些运算会低效实现），通过 [fbgemm](https://github.com/pytorch/FBGEMM)
- ARM CPU（通常在移动/嵌入式设备中找到），通过 [qnnpack](https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu/qnnpack)

相应的实现会根据 PyTorch 构建模式自动选择，不过用户可以通过将 `torch.backends.quantization.engine` 设置为 `fbgemm` 或 `qnnpack` 来覆盖这个选项。

```{note}
目前 PyTorch 还没有在 CUDA 上提供量化的算子实现——这是未来工作的方向。将模型移到 CPU 上，以测试量化的功能。

量化感知训练（通过 `FakeQuantize`，它模拟 fp32 中的量化数值）支持 CPU 和 CUDA。
```

在准备量化模型时，必须确保 `qconfig` 和用于量化计算的引擎与将在其上执行模型的后端匹配。`qconfig` 控制量化传递期间使用的观测器类型。当对线性和卷积函数和模块进行权重打包时，`qengine` 控制是使用 `fbgemm` 还是 `qnnpack` 特定的打包函数。例如：

`fbgemm` 的默认设置：

```python
# set the qconfig for PTQ
qconfig = torch.quantization.get_default_qconfig('fbgemm')
# or, set the qconfig for QAT
qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# set the qengine to control weight packing
torch.backends.quantized.engine = 'fbgemm'
```

`qnnpack` 的默认设置：

```python
# set the qconfig for PTQ
qconfig = torch.quantization.get_default_qconfig('qnnpack')
# or, set the qconfig for QAT
qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
# set the qengine to control weight packing
torch.backends.quantized.engine = 'qnnpack'
```

## 量化定制

虽然提供了基于观测到的张量数据选择 scale 因子和 bias 的观测者的默认实现，但开发人员可以提供自己的量化函数。量化可以有选择地应用于模型的不同部分，也可以针对模型的不同部分进行不同的配置。

PyTorch 还为 {func}`conv2d`、{func}`conv3d` 和 {func}`linear` 提供了逐通道量化的支持。

量化工作流通过添加（例如添加观测者作为 `.observer` 子模块）或替换（例如转换 `nn.Conv2d` 为 `nn.quantized.Conv2d`）的子模块。这意味着模型在整个过程中保持常规的基于 `nn.Module` 的实例，因此可以与其他 PyTorch API 一起工作。

### 量化自定义模块 API

Eager 模式和 FX 图模式量化 API 都为用户提供了钩子，用户可以通过自定义的方式指定量化模块，并使用用户定义的逻辑进行观测和量化。用户需要指定：

1. 源 fp32 模块的 Python 类型（存在于模型中）
2. 被观测模块的 Python 类型（由用户提供）。这个模块需要定义 {func}`from_float` 函数，它定义了如何从原始 fp32 模块创建观测到的模块。
3. 量化模块的 Python 类型（由用户提供）。这个模块需要定义 {func}`from_observed` 函数，该函数定义如何从被观测模块创建量化的模块。
4. 上面描述的 (1)、(2)、(3) 配置，传递给量化 API。

然后，框架将执行以下操作：

1. 在 `prepare` 模块交换过程中，它将使用 (2) 中类的 {func}`from_float` 函数将 (1) 中指定的类型的每个模块转换为 (2) 中指定的类型
1. 在 `convert` 模块交换期间，它将使用 (3) 中类的 {func}`from_observed` 函数将 (2) 中指定的类型的每个模块转换为 (3) 中指定的类型。

目前，要求是 `ObservedCustomModule` 将有单个张量输出，并且观测者将由框架（而不是由用户）添加到该输出上。观测者将作为自定义模块实例的属性存储在 `activation_post_process` 键下。放宽这些限制可能会在未来的某个时候实现。

```{rubric} 示例
```

```python
import torch
import torch.nn.quantized as nnq
import torch.quantization.quantize_fx

# 源 fp32 模块被替换
class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)

# 自定义观测者模块，由用户提供
class ObservedCustomModule(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)

    @classmethod
    def from_float(cls, float_module):
        assert hasattr(float_module, 'qconfig')
        observed = cls(float_module.linear)
        observed.qconfig = float_module.qconfig
        return observed

# 自定义量化模块，由用户提供
class StaticQuantCustomModule(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)

    @classmethod
    def from_observed(cls, observed_module):
        assert hasattr(observed_module, 'qconfig')
        assert hasattr(observed_module, 'activation_post_process')
        observed_module.linear.activation_post_process = \
            observed_module.activation_post_process
        quantized = cls(nnq.Linear.from_float(observed_module.linear))
        return quantized

#
# 示例 API 调用（Eager 模式量化）
#


m = torch.nn.Sequential(CustomModule()).eval()

prepare_custom_config_dict = {
    "float_to_observed_custom_module_class": {
        CustomModule: ObservedCustomModule
    }
}
convert_custom_config_dict = {
    "observed_to_quantized_custom_module_class": {
        ObservedCustomModule: StaticQuantCustomModule
    }
}

m.qconfig = torch.quantization.default_qconfig
mp = torch.quantization.prepare(
    m, prepare_custom_config_dict=prepare_custom_config_dict)
# calibration (not shown)
mq = torch.quantization.convert(
    mp, convert_custom_config_dict=convert_custom_config_dict)

#
# 示例 API 调用（FX 图模式量化）
#

m = torch.nn.Sequential(CustomModule()).eval()

qconfig_dict = {'': torch.quantization.default_qconfig}
prepare_custom_config_dict = {
    "float_to_observed_custom_module_class": {
        "static": {
            CustomModule: ObservedCustomModule,
        }
    }
}
convert_custom_config_dict = {
    "observed_to_quantized_custom_module_class": {
        "static": {
            ObservedCustomModule: StaticQuantCustomModule,
        }
    }
}
mp = torch.quantization.quantize_fx.prepare_fx(
    m, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)
# calibration (not shown)
mq = torch.quantization.quantize_fx.convert_fx(
    mp, convert_custom_config_dict=convert_custom_config_dict)
```



## 最佳实践（已废弃）

如果你使用 `fbgemm` 后端，设置观测者的 `reduce_range` 参数为 `True`。该参数通过将量化数据类型的范围减少 1 位来防止某些 int8 指令的溢出。

## 数值调试（原型）

```{warning}
数值调试工具是早期的原型，可能会发生变化。
```

- {func}`torch.ao.ns._numeric_suite` Eager 模式数值套件
- {func}`torch.ao.ns._numeric_suite_fx` FX 数值套件

## 常见错误

### 保存和加载量化模型

当调用 {func}`torch.load` 量化模型，如果你看到如下错误：

```python
AttributeError: 'LinearPackedParams' object has no attribute '_modules'
```

这是因为直接使用 {func}`torch.save` 和 {func}`torch.load` 保存和加载量化模型是不支持的。为了保存/加载量化模型，可以使用以下方法：

1. 保存/加载量化模型 `state_dict`

    ```python
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return x

    m = M().eval()
    prepare_orig = prepare_fx(m, {'' : default_qconfig})
    prepare_orig(torch.rand(5, 5))
    quantized_orig = convert_fx(prepare_orig)

    # Save/load using state_dict
    b = io.BytesIO()
    torch.save(quantized_orig.state_dict(), b)

    m2 = M().eval()
    prepared = prepare_fx(m2, {'' : default_qconfig})
    quantized = convert_fx(prepared)
    b.seek(0)
    quantized.load_state_dict(torch.load(b))
    ```

2. 使用 {func}`torch.jit.save` 和 {func}`torch.jit.load` 保存/加载脚本化量化模型

    ```python
    # Note: using the same model M from previous example
    m = M().eval()
    prepare_orig = prepare_fx(m, {'' : default_qconfig})
    prepare_orig(torch.rand(5, 5))
    quantized_orig = convert_fx(prepare_orig)

    # save/load using scripted model
    scripted = torch.jit.script(quantized_orig)
    b = io.BytesIO()
    torch.jit.save(scripted, b)
    b.seek(0)
    scripted_quantized = torch.jit.load(b)
    ```

### 传递非量化的张量到量化的核

如果你看到类似的错误：

```python
RuntimeError: Could not run 'quantized::some_operator' with arguments from the 'CPU' backend...
```

这意味着你试图传递非量化的张量给量化的核。常见的解决方法是使用 `torch.quantization.QuantStub` 来量化张量。这需要在 Eager 模式量化中手动完成。e2e 的例子：

```python
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        # during the convert step, this will be replaced with a
        # `quantize_per_tensor` call
        x = self.quant(x)
        x = self.conv(x)
        return x
```

### 传递量化的张量到非量化的核

如果你看到类似的错误：

```python
RuntimeError: Could not run 'aten::thnn_conv2d_forward' with arguments from the 'QuantizedCPU' backend.
```

这意味着你试图传递量化的张量给非量化的核。常见的解决方法是使用 {class}`torch.quantization.DeQuantStub` 反量化张量。这需要在 Eager 模式量化中手动完成。e2e 的例子：

```python
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        # this module will not be quantized (see `qconfig = None` logic below)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # during the convert step, this will be replaced with a
        # `quantize_per_tensor` call
        x = self.quant(x)
        x = self.conv1(x)
        # during the convert step, this will be replaced with a
        # `dequantize` call
        x = self.dequant(x)
        x = self.conv2(x)
        return x

m = M()
m.qconfig = some_qconfig
# turn off quantization for conv2
m.conv2.qconfig = None
```