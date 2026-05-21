import os
import shutil

# 设置 Tiny-ImageNet 的路径
DATA_DIR = './data/tiny-imagenet-200' 
VAL_DIR = os.path.join(DATA_DIR, 'val')
IMG_DIR = os.path.join(VAL_DIR, 'images')
ANNOTATION_FILE = os.path.join(VAL_DIR, 'val_annotations.txt')

def format_val():
    print(f"正在处理 Tiny-ImageNet 验证集: {VAL_DIR} ...")
    
    if not os.path.exists(ANNOTATION_FILE):
        print("错误：找不到 val_annotations.txt，请检查路径。")
        return

    # 1. 读取标注文件
    with open(ANNOTATION_FILE, 'r') as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        parts = line.strip().split('\t')
        filename = parts[0]
        class_id = parts[1] # 获取形如 n01443537 的类ID

        # 2. 创建对应的类文件夹 (如果不存在)
        class_dir = os.path.join(VAL_DIR, class_id)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # 3. 移动图片
        src_path = os.path.join(IMG_DIR, filename)
        dst_path = os.path.join(class_dir, filename)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            count += 1
    
    print(f"成功整理了 {count} 张验证集图片！")

    # 4. (可选) 删除空的 images 文件夹
    if os.path.exists(IMG_DIR) and not os.listdir(IMG_DIR):
        os.rmdir(IMG_DIR)
        print("已清理空的 images 文件夹。")
    elif os.path.exists(IMG_DIR):
        print(f"警告: images 文件夹不为空，未删除。请手动检查 {IMG_DIR}")

if __name__ == '__main__':
    if os.path.exists(VAL_DIR):
        format_val()
    else:
        print(f"错误：找不到目录 {VAL_DIR}，请先下载并解压数据集。")