# Reproducibility
from monai.utils import set_determinism
set_determinism(seed=100)

from .multin import Multi2Di


# Function that creates the model
def create_model(opt):
    """Create the specified model in options.

    Args:
        opt: parsed options.

    Returns:
        model (nn.Module)
    """
    model = Multi2Di(
        in_channels=opt.in_chans,
        num_classes=opt.num_classes,
        img_size=opt.image_size,
        use_auto_att=opt.use_auto_att,
        feat_comb_mode=opt.feat_comb_mode,
        backbone=opt.backbone,
        backbone_chkp=opt.checkpoint,
        use_mil_head=opt.use_mil_head
    )

    return model
