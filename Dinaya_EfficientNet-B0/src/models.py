import timm
def create_model(name="efficientnet_b0", num_classes=2, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=3)
