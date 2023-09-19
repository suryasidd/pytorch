import torch
import logging

from .registry import register_backend
from .common import fake_tensor_unsupported
from torch._inductor.compile_fx import compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
import os

log = logging.getLogger(__name__)

@register_backend
@fake_tensor_unsupported
def openvino(gm, example_inputs, **kwargs):
    opts = {}
    if "options" in kwargs.keys():
        opts = kwargs["options"]
    try:
        from openvino.frontend import FrontEndManager
        from openvino.runtime import Core, Type, PartialShape
        from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
        from openvino.frontend.pytorch.torchdynamo.execute import execute, execute_cached
        from openvino.frontend.pytorch.torchdynamo.compile import cached_model_name, cache_root_path, get_device, openvino_compile_cached_model
        from openvino.runtime import Core, Type, PartialShape
        executor_parameters = None
        inputs_reversed = False
        if "OPENVINO_TORCH_MODEL_CACHING" in opts.keys() and opts["OPENVINO_TORCH_MODEL_CACHING"]:
            # Create a hash to be used for caching
            model_hash_str = sha256(gm.code.encode('utf-8')).hexdigest()
            executor_parameters = {"model_hash_str": model_hash_str}
            # Check if the model was fully supported and already cached
            example_inputs.reverse()
            inputs_reversed = True
            maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", get_device(), example_inputs, cache_root_path())
            if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
                # Model is fully supported and already cached. Run the cached OV model directly.
                compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, *example_inputs)

                def _call(*args):
                    res = execute_cached(compiled_model, *args)
                    return res
                return _call
        if inputs_reversed:
            example_inputs.reverse()
        model = make_fx(gm)(*example_inputs)
        with torch.no_grad():
            model.eval()
        partitioner = Partitioner()
        compiled_model = partitioner.make_partitions(model)

        if executor_parameters is not None and 'model_hash_str' in executor_parameters:
            # Check if the model is fully supported.
            fully_supported = partitioner.check_fully_supported(compiled_model)
            if fully_supported:
                executor_parameters["model_hash_str"] += "_fs"

        def _call(*args):
            res = execute(compiled_model, *args, executor="openvino",
                          executor_parameters=executor_parameters)
            return res
        return _call
    except Exception as e:
        log.debug(f"Failed in OpenVINO execution: {e}")
        return compile_fx(gm, example_inputs)