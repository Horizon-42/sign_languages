import torch

PTH_FILE_PATH = './data/thws-mai-idl-ss-25-sign-language/SignLanguage_kaggle/old_annotated.pth'

print(f"Attempting to inspect the content of {PTH_FILE_PATH}...")
try:
    # 加载文件，但暂时不进行结构判断，直接打印
    data_from_pth = torch.load(PTH_FILE_PATH)

    print(f"Type of loaded data: {type(data_from_pth)}")

    if isinstance(data_from_pth, torch.Tensor):
        print(f"It's a single Tensor. Shape: {data_from_pth.shape}")
        # 如果是单个Tensor，你需要知道这个Tensor是图像还是标签，或者包含两者
        # 你可能需要进一步的文档或线索来判断
    elif isinstance(data_from_pth, list):
        print(f"It's a list. Length: {len(data_from_pth)}")
        # 打印列表中前几个元素的类型和形状，以了解其结构
        for i, item in enumerate(data_from_pth[:5]): # 打印前5个元素
            if isinstance(item, torch.Tensor):
                print(f"  Item {i}: Tensor, Shape: {item.shape}")
            else:
                print(f"  Item {i}: Type: {type(item)}")
    elif isinstance(data_from_pth, dict):
        print(f"It's a dictionary. Keys: {data_from_pth.keys()}")
        # 打印字典中每个键对应的值的类型和形状
        for key, value in data_from_pth.items():
            if isinstance(value, torch.Tensor):
                print(f"  Key '{key}': Tensor, Shape: {value.shape}")
            else:
                print(f"  Key '{key}': Type: {type(value)}")
    else:
        print(f"It's an unknown type: {type(data_from_pth)}")

    print("\n--- Based on the above, please update your create_dataset.py script. ---")
    print("For example, if it's a list where the first element is images and second is labels:")
    print("all_images, all_labels = loaded_data[0], loaded_data[1]")
    print("Or if it's a dictionary with keys 'img_data' and 'labels_data':")
    print("all_images = loaded_data['img_data']")
    print("all_labels = loaded_data['labels_data']")

except Exception as e:
    print(f"Error loading or inspecting .pth file: {e}")
    print("Please ensure the .pth file is not corrupted and can be loaded by torch.load().")