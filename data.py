import torch
import torchvision.transforms as transforms
from PIL import Image

class data_loader:
    def __init__(self):
        self.batch_size = 1
        self.x = torch.randn(self.batch_size, 1, 1440, 666, requires_grad=True)
        self.img = Image.open('/home/jysuh/Downloads/jihyeon.JPG')

    def return_data(self):
        return self.x

    def return_batch(self):
        return self.batch_size

    def process(self):
        resize = transforms.Resize([224, 224])
        img = resize(self.img)

        img_ycbcr = img.convert('YCbCr')
        self.img_y, self.img_cb, self.img_cr = img_ycbcr.split()

        to_tensor = transforms.ToTensor()
        self.img_y = to_tensor(self.img_y)
        self.img_y.unsqueeze_(0)

        return self.img_y

    def save(self, img_out_y):
        # PyTorch 버전의 후처리 과정 코드를 이용해 결과 이미지 만들기
        final_img = Image.merge(
            "YCbCr", [
                img_out_y,
                self.img_cb.resize(img_out_y.size, Image.BICUBIC),
                self.img_cr.resize(img_out_y.size, Image.BICUBIC),
            ]).convert("RGB")

        # 이미지를 저장하고 모바일 기기에서의 결과 이미지와 비교하기
        final_img.save("/home/jysuh/Downloads/jihyeoni.jpg")