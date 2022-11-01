from .climatv1 import CLIMATv1
from .climatv2 import CLIMATv2


def create_model(cfg, device, pr_weights=None, pn_weights=None, y0_weights=None):
    if cfg.method_name == "climatv1":
        return CLIMATv1(cfg, device, pn_weights=pn_weights)
    elif cfg.method_name == "climatv2":
        return CLIMATv2(cfg, device, pn_weights=pn_weights)
    else:
        raise ValueError(f"Not support method name '{cfg.method_name}'.")
