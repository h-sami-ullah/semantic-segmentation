from torchvision import transforms


def train():
    trans = transforms.Compose([
        transforms.ToTensor()])
    return trans