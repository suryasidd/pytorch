import logging

from .registry import register_backend
from .common import fake_tensor_unsupported

log = logging.getLogger(__name__)

@register_backend
@fake_tensor_unsupported
def openvino(gm, example_inputs, **kwargs):
    opts = {}
    if "options" in kwargs.keys():
        opts = kwargs["options"]
    try:
        from openvino.frontend.pytorch.torchdynamo.backend import backend_init
        func = backend_init(gm, example_inputs, opts)

        if callable(func):
            return func
        else:
            return compile_fx(gm, example_inputs)
    except Exception as e:
        log.debug("OpenVINO is not installed")