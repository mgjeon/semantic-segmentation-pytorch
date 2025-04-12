import torchvision
from torch.nn import Conv2d

def get_model(
        name="fcn_resnet50", 
        weights="COCO_WITH_VOC_LABELS_V1",
        weights_backbone=None,
        aux_loss=True,
        num_classes=2
    ):

    num_classes = 2  # Aurora dataset has 2 classes (background, aurora)

    if name == "fcn_resnet50":

        model = torchvision.models.get_model(
            name=name,
            weights=weights,
            weights_backbone=weights_backbone,
            aux_loss=aux_loss,
        )

        out_in_channels = model.classifier[4].in_channels
        model.classifier[4] = Conv2d(out_in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))

        aux_in_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = Conv2d(aux_in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))

        return model
    
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")