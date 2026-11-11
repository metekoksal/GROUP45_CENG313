import segmentation_models_pytorch as smp


def build_unet(encoder="resnet34", pretrained=True):
    """ImageNet ön-eğitimli encoder'lı U-Net, çıktı: (B, 1, H, W) logit."""
    return smp.Unet(encoder_name=encoder,
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)