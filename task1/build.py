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
        cross_att_type=opt.cross_att_type,
        bicross_att_reduction=opt.cross_att_reduction,
        num_cross_att_layers=1,
        final_proj=opt.final_proj,
        backbone=opt.backbone,
        backbone_chkp=opt.checkpoint,
    )

    return model
