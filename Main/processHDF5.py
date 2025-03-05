import h5py

def saveHDF5(flows, path):
    if len(flows) > 0:
        height, width = flows[0].shape[:2]
    with h5py.File(path, 'w') as file:
        for i, flow in enumerate(flows):
            file.create_dataset(f'optical_flow_{i}', data=flow)
        file.create_dataset(f'length', data=len(flows))
        file.create_dataset(f'height', data=height)
        file.create_dataset(f'width', data=width)

# Hàm để đọc dữ liệu từ file HDF5
def loadHDF5(file_path):
    with h5py.File(file_path, 'r') as file:
        # Đọc dữ liệu về chiều cao và chiều rộng
        height = file['height'][()]
        width = file['width'][()]

        # Đọc số lượng khung hình
        num_flows = file['length'][()]

        # Đọc tất cả các dataset optical flow
        flows = []
        for i in range(num_flows):
            flow = file[f'optical_flow_{i}'][:]
            flows.append(flow)

        return flows, height, width