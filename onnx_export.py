import torch

from data import data_loader
import torch.utils.model_zoo as model_zoo
import torch.onnx

from model import SuperResolutionNet

def forward(model, data):
    if torch.cuda.is_available():
        map_location = None
    else:
        map_location = lambda storage, loc: storage

    torch_model.load_state_dict(model_zoo.load_url(model._load_weights(), map_location=map_location))

    torch_model.eval()

    x = data.process()
    torch_out = torch_model(x)

    return x, torch_out

if __name__ == '__main__':
    data = data_loader()

    # 위에서 정의된 모델을 사용하여 초해상도 모델 생성
    torch_model = SuperResolutionNet(upscale_factor=3)

    batch_size = data.return_batch()

    x, torch_out = forward(model=torch_model, data=data)

    # 모델 변환
    torch.onnx.export(torch_model,  # 실행될 모델
                      x,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      "./onnx_out/super_resolution.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=10,  # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=['output'],  # 모델의 출력값을 가리키는 이름
                      dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                                    'output': {0: 'batch_size'}})