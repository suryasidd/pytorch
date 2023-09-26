import logging

from .registry import register_backend
from .common import fake_tensor_unsupported
from torch._inductor.compile_fx import compile_fx


log = logging.getLogger(__name__)

@register_backend
@fake_tensor_unsupported
def openvino(gm, example_inputs, options=None):
    try:
        from openvino.frontend.pytorch.torchdynamo.backend import backend_init
        func = backend_init(gm, example_inputs, options)

        if callable(func):
            return func
        else:
            return compile_fx(gm, example_inputs)
    except Exception as e:
        log.debug("OpenVINO is not installed")