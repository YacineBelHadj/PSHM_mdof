import torch
import torchvision.transforms as T


class ToTensor(T.ToTensor):
    """This class replace the torchvision ToTensor transform by allowing
    tensor to pass in, in which case it will be returned as is.

    """

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        else:
            return super().__call__(pic)
