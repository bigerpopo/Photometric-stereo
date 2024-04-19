import numpy as np
import cv2

# 讀取圖檔
path = 'test_datasets/bunny/'

# 存儲圖像的陣列
images = []

# 讀取圖像並存儲到陣列中
for i in range(1, 6):
    img_path = path + (f'pic{i}.bmp')  # 構建圖像檔案的完整路徑
    print(img_path)
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
    else:
        print(f"無法讀取圖像 {img_path}，請確認路徑是否正確。")

#-----------------------------------------------------

# 確認讀取內容  
# 顯示每一張圖像
# for i in range(0, 5):
#     cv2.imshow(f'Image{i}', images[i])

#     # 等待按鍵按下
#     cv2.waitKey(0)

#     # 關閉所有視窗
#     cv2.destroyAllWindows()

#-----------------------------------------------------

light_list = []

# 讀取光源向量
with open(path + "light.txt", "r") as file :
    for line in file.readlines():
        # 將每一行的文字按冒號分割成名稱和數值部分
        name, values = line.strip().split(":")
        # 去除數值部分的括號，並按逗號分割成三個值
        values = values.strip()[1:-1].split(",")
        # 將數值部分轉換為整數
        values = [float(value) for value in values]
        # 測試讀取結果
        # print(name.strip(), ":", values)
        # 存入 light_list
        light_list.append(values)

light_list = np.array(light_list)
# 因為少一張圖，刪除最後一筆資料
light_list = light_list[:-1]
# 確認light_list內容
# print(light_list)
# print(light_list.shape)

print(images[0].shape)  

# # 120*120 OR 114*196
# albedo_lst = np.zeros(images[0].shape)
# N_lst = np.zeros(images[0].shape)

# for x in range(images[0].shape[0]) :
#     for y in range(images[0].shape[1]) :
#         I = np.array([
#             images[0][x][y],
#             images[1][x][y],
#             images[2][x][y],
#             images[3][x][y],
#             images[4][x][y]
#         ])
#         # kn->N shadow_inverse*[In]
#         N = np.dot(np.dot(np.linalg.inv(np.dot(S_lst.T, S_lst)), S_lst.T), I)        
#         G = N.T # 因為先多轉一次去計算，後面要再轉回來
        
#         # 算Normal n = N/|N|
#         G_gray = G[0]*0.299+G[1]*0.587+G[2]*0.114
#         Gnorm = np.linalg.norm(G_gray)
#         # Normal是0，Aldedo就是0
#         if Gnorm==0:
#             continue
#         N_lst[x][y] = G_gray/Gnorm
        
#         # 算Albedo |N|
#         rho = np.linalg.norm(G, axis=1)
#         albedo_lst[x][y] = rho
# # 控制在0到255間               
# N_lst = ((N_lst*0.5 + 0.5)*255).astype(np.uint8)
# albedo_lst = (albedo_lst/np.max(albedo_lst)*255).astype(np.uint8)



# # 顯示圖片
# cv2.imshow('Albedo', albedo_lst)

# # 顯示圖片
# cv2.imshow('Normal', N_lst)

# # 按下任意鍵則關閉所有視窗
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 寫入不同圖檔格式
# cv2.imwrite(path + 'Albedo.png', albedo_lst)
# cv2.imwrite(path + 'Normal.png', N_lst)