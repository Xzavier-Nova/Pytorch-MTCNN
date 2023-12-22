import base64

import numpy as np
import requests
import cv2
global_AUTHORIZATION = (
    "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjUxNTkxLCJ1dWlkIjoiZDZlMjZlY2QtYWNjYS00ZDQ0L"
    "TlmODctM2Y3NDdiNDQxNGMyIiwiaXNfYWRtaW4iOmZhbHNlLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1"
    "lIjoiIiwidGVuYW50IjoiYXV0b2RsIn0.UorfBOe6JuShFvsjz7UcJB93ipq6TcP_Qvz-hc74i-rGbPdYeXS5D_CrF"
    "RuxDfExrsA-bZdbp9Ysm3ZtuW39SA"
)
url = "http://127.0.0.1:9000/matting/face_cut"

img_path = r"D:\AppsaboutCodes\Projects\Pytorch-MTCNN\dataset\test_img_1.jpg"

# 读取图像
img = cv2.imread(img_path)  # 这里是np数组了
# 将图像编码为 JPEG 格式
_, img_encoded = cv2.imencode('.jpg', img)
# 将编码后的图像转换为 base64 字符串
img_base64 = base64.b64encode(img_encoded).decode('utf-8')

response = requests.post(
    url=url,
    json={"image_file": img_base64},
    headers={"AUTHORIZATION": global_AUTHORIZATION}
)
img = response.json()["data"]["image_cropped"]  # 这一段获取数据（编码）
img_bytes = base64.b64decode(img)  # 解码
img_array = np.frombuffer(img_bytes, dtype=np.uint8)  # 转换为np数组
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 读取为cv可以处理的图像类型，同时也是一个np数组
cv2.imwrite("./test_1.jpg", image)  # 保存
