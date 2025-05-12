import onnx
from onnxruntime.training import artifacts
import onnxruntime.training.onnxblock as onnxblock
import torch.onnx
import onnxruntime.training.onnxblock._graph_utils as _graph_utils
import onnxruntime.training.onnxblock.blocks as blocks
import copy
from segmentation_model import EffSegModel

class DiceLoss(onnxblock.Block):
    def __init__(self, reduction: str = "mean"):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise RuntimeError(f"Reduction {reduction} not supported.")

        reduction_blocks = {"mean": blocks.ReduceMean, "sum": blocks.ReduceSum, "none": blocks.PassThrough}
        self._reduce = reduction_blocks[reduction](keepdims=False)
        self._add = onnxblock.blocks.Add()
        self._mul = onnxblock.blocks.Mul()
        self._div = onnxblock.blocks.Div()
        self._sub = onnxblock.blocks.Sub()
        self._sum = reduction_blocks["sum"](keepdims=True)
        self._eps = onnxblock.blocks.Constant(0.0001)

    def build(self, loss_input_name: str, target_name: str | None = "target"):
      if not _graph_utils.node_arg_exists(self.base, target_name):
          # Create a new graph input. This is the target input needed to compare
          # the graph output against to calculate loss.
          target_input = copy.deepcopy(_graph_utils.get_output_from_output_name(self.base, loss_input_name))
          target_input.name = target_name
          self.base.graph.input.append(target_input)

      sub_twos_operand_name = _graph_utils.generate_graph_name("diceloss.sub_twos")
      self.base.graph.initializer.append(
          onnx.helper.make_tensor(sub_twos_operand_name, onnx.TensorProto.FLOAT, [1], [2.0])
      )

      sub_ones_operand_name = _graph_utils.generate_graph_name("diceloss.sub_ones")
      self.base.graph.initializer.append(
          onnx.helper.make_tensor(sub_ones_operand_name, onnx.TensorProto.FLOAT, [1], [1.0])
      )

      print('loss_input_name1', '--------------', loss_input_name)
      intersection = self._mul(sub_twos_operand_name, self._sum(self._mul(loss_input_name, target_name)))      # intersection = (inputs * targets).sum()
      total = self._add(self._sum(target_name), self._sum(loss_input_name))
      dice_score = self._sub(sub_ones_operand_name, self._div(intersection, total))
      return self._reduce(dice_score)

dice_loss = DiceLoss()

device = "cpu"
# Generate a random input.
batch_size = 1
input_size = [3,224,224]
model_inputs = torch.randn(batch_size, 3, 224, 224, device=device)
print(model_inputs.shape)
input_names = ["input"]
output_names = ["output"]
dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
pt_model = EffSegModel()
# torch.onnx.export(pt_model, model_inputs, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

torch.onnx.export(
    pt_model,
    model_inputs,
    "efficientnet_lite0.onnx",
    input_names=input_names,
    output_names=output_names,
    opset_version=17,
    # do_constant_folding=False,
    # export_params=True,
    # keep_initializers_as_inputs=False,
    dynamic_axes=dynamic_axes
)

onnx_model = onnx.load("efficientnet_lite0.onnx")
output_names = ["output"]

requires_grad = [param.name for param in onnx_model.graph.initializer]
# requires_grad = ['model.0.conv.weight', 'model.0.conv.bias']

# frozen_params = [name for name, param in onnx_model.named_parameters() if not param.requires_grad]
frozen_params = [param.name for param in onnx_model.graph.initializer if param.name not in requires_grad]
# frozen_params = ['model.0.bnorm.weight', 'model.0.bnorm.bias', 'model.1.bnorm.weight', 'model.1.bnorm.bias', 'model.2.bnorm.weight', 'model.2.bnorm.bias', 'model.3.bnorm.weight', 'model.3.bnorm.bias', ]
# frozen_params.append()
artifacts.generate_artifacts(
    onnx_model,
    optimizer=artifacts.OptimType.AdamW,
    loss=dice_loss,     # artifacts.LossType.CrossEntropyLoss,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    # frozen_params=[],
    additional_output_names=output_names
    )