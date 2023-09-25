import torch
import onnx
import onnxruntime
import torch.utils.model_zoo as model_zoo
import numpy as np

from PIL import Image
from data import data_loader
from model import SuperResolutionNet

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

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
    torch_model = SuperResolutionNet(upscale_factor=3)

    x, torch_out = forward(model=torch_model, data=data)

    # Check the model
    onnx_model = onnx.load("./onnx_out/super_resolution.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("./onnx_out/super_resolution.onnx")

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data.process())}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    data.save(img_out_y)
    # # ONNX 런타임에서 계산된 결과값
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    #
    # # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")