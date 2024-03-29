import os
import shutil

def cleanup_except_latest(directory_path):
    # 读取 latest 文件的内容
    latest_file_path = os.path.join(directory_path, "latest")
    with open(latest_file_path, "r") as file:
        # 读取要保留的文件夹名称
        folders_to_keep = file.read().strip().split('\n')

    # 添加"latest"到保留列表
    folders_to_keep.append("latest")

    # 获取目录下的所有文件和文件夹
    all_items = os.listdir(directory_path)

    # 过滤出所有文件夹
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]

    # 删除不在保留列表中的文件夹
    for folder in folders:
        if folder not in folders_to_keep:
            # 构建完整路径
            folder_path = os.path.join(directory_path, folder)
            # 使用 os.removedirs 删除文件夹
            shutil.rmtree(folder_path)

    # 返回更新后的目录内容
    print(os.listdir(directory_path))


if __name__ == '__main__':
    cleanup_except_latest("/home/dcuuser/dxm/diffusers/train/FineTunedStableDiffusion-lora")