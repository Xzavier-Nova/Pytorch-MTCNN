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
url = "https://edith.nt.starmerx.com/matting/face_cut"

img_url = "https://starmerx.oss-cn-shanghai.aliyuncs.com/amz/76677be9b58b5ef8f1fbc8dc3e7609f7.jpg"

response = requests.post(
    url=url,
    json={"image_url": img_url},
    headers={"AUTHORIZATION": global_AUTHORIZATION}
)
# print(response.text)
img = response.json()["data"]["image_cropped"]  # 这一段获取数据（编码）
img_bytes = base64.b64decode(img)  # 解码
img_array = np.frombuffer(img_bytes, dtype=np.uint8)  # 转换为np数组
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 读取为cv可以处理的图像类型，同时也是一个np数组
cv2.imwrite("./test_nt.jpg", image)  # 保存

