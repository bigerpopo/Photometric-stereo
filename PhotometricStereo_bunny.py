import numpy as np
import cv2
from sklearn.preprocessing import normalize

# 輸出obj檔案函式
def output_depth_to_obj(depth_map, output_file):
    height, width = depth_map.shape

    with open(output_file, 'w') as f:
        # 寫入頂點
        for y in range(height):
            for x in range(width):
                f.write(f'v {x} {y} {depth_map[y, x]}\n')

        # 寫入面
        for y in range(height - 1):
            for x in range(width - 1):
                # 每個像素對應一個面，使用四個頂點來定義一個面
                # 面的定義順序為逆時針方向
                v1 = y * width + x + 1
                v2 = y * width + x + 2
                v3 = (y + 1) * width + x + 2
                v4 = (y + 1) * width + x + 1
                f.write(f'f {v1} {v2} {v3} {v4}\n')

def generate_height_map(N_mat):
    height, width = [120, 120]
    height_map = np.zeros((height, width))

    # 處理左列的像素
    for y in range(1, height):
        if N_mat[y][0][2] == 0:
            continue
        height_map[y][0] = height_map[y-1][0] + (N_mat[y][0][1] / N_mat[y][0][2])

    # 處理每一行的其餘像素
    for y in range(height):
        for x in range(1, width):
            if N_mat[y][x][2] == 0:
                continue
            height_map[y][x] = height_map[y][x-1] + (N_mat[y][x][0] / N_mat[y][x][2])

    return height_map


# 讀取圖檔
path = 'test_datasets/bunny/'
# path = 'test_datasets/teapot/'
# 存儲圖像的陣列
images = []

# 讀取圖像並存儲到陣列中
for i in range(1, 6):
    img_path = path + (f'pic{i}.bmp')  # 構建圖像檔案的完整路徑
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
    else:
        print(f"無法讀取圖像 {img_path}，請確認路徑是否正確。")
    print(img.shape)

light_list = []

# 讀取光源向量
with open(path + "light.txt", "r") as file :
    for line in file.readlines():
        # 將每一行的文字按冒號分割成名稱和數值部分
        name, values = line.strip().split(":")
        # 去除數值部分的括號，並按逗號分割成三個值
        values = values.strip()[1:-1].split(",")
        # 將數值部分轉換為浮點數
        values = [float(value) for value in values]

        # 正規化後存入light_list
        light_list.append(values / np.linalg.norm(values))

# 因為test_datasets/bunny少一張圖，刪除最後一筆資料
light_list = light_list[:-1]
light_list = np.array(light_list)

# 進行運算
# I = KdNL
# KdN = (LT * L)- LTI
# kd = |N|
Albedo_mat = np.zeros(images[0].shape)
N_mat = np.zeros(images[0].shape)
N_mat2 = np.zeros(images[0].shape)

for x in range(images[0].shape[0]) :
    for y in range(images[0].shape[1]) :
        I = np.array([images[i][x][y] for i in range(5)])
        
        # 依照公式計算KdN
        KdN = np.dot(np.dot(np.linalg.inv(np.dot(light_list.T, light_list)), light_list.T), I)

        # 計算Albedo: ro
        ro = np.linalg.norm(KdN, axis = 0)
        Albedo_mat[x][y] = ro

        # 計算N
        # 3*3 -> 1*3
        # 轉置方便運算
        N = KdN.T
        # 神奇的RGB TO GRAY轉換公式
        N_gray = N[0]*0.299+N[1]*0.587+N[2]*0.114
        N_gray2 = N[0]*0.299+N[1]*0.587+N[2]*0.114
        # 調換通道
        temp = N_gray[2]
        N_gray[2] = N_gray[0]
        N_gray[0] = temp
        Nnorm = np.linalg.norm(N_gray)
        Nnorm2 = np.linalg.norm(N_gray2)
        if Nnorm==0:
            continue
        N_mat[x][y] = N_gray/Nnorm
        N_mat2[x][y] = N_gray2/Nnorm2

depth_map = generate_height_map(N_mat2)
output_depth_to_obj(depth_map, "new.obj")


# 將數值縮放至0~255
Albedo_mat = ((Albedo_mat/np.max(Albedo_mat))*255).astype(np.uint8)
N_mat = ((N_mat*0.5 + 0.5)*255).astype(np.uint8)
# 顯示圖片
cv2.imshow('Albedo', Albedo_mat)
cv2.imshow('Normal', N_mat)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

# 寫入不同圖檔格式
cv2.imwrite(path + 'Albedo.png', Albedo_mat)
cv2.imwrite(path + 'Normal.png', N_mat)