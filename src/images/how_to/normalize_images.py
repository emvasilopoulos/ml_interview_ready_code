import cv2
import torch
import torchvision

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    opencv_image = cv2.imread("path/to/image.jpg")
    tensor_image = torch.from_numpy(opencv_image).to(device)
    tensor_image = tensor_image.float()  # Convolutionals take float
    tensor_image /= 255  # cv2 supplies images from 0 to 255
    tensor_image.unsqueeze(0)  # Add batch dimension

    """ TODO - BE CAREFUL IMAGE IS NOT READY FOR A CNN. IT NEEDS THE PROPER IMAGE SIZES """
