import json
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import pylab as plt

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


def normalize(img):
    """Unit interval preprocessing"""
    img = (img - img.min()) / (img.max() - img.min())
    return img


def load_tfjs_model(json_path, bin_path):
    # Load JSON specification
    with open(json_path, "r") as f:
        model_spec = json.load(f)

    # Load binary weights
    with open(bin_path, "rb") as f:
        weights_data = np.frombuffer(f.read(), dtype=np.float32)

    return model_spec, weights_data


def create_activation(activation_name):
    activation_map = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
    }
    return activation_map.get(activation_name, nn.Identity())


def calculate_same_padding(kernel_size, dilation):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if isinstance(dilation, int):
        dilation = (dilation,) * 3

    padding = []
    for k, d in zip(kernel_size, dilation):
        padding.append((k - 1) * d // 2)
    return tuple(padding)


def create_pytorch_model(model_spec, weights_data):
    layers = []
    weight_index = 0
    in_channels = 1  # Start with 1 input channel

    for layer in model_spec["modelTopology"]["model_config"]["config"][
        "layers"
    ][
        1:
    ]:  # Skip input layer
        if layer["class_name"] == "Conv3D":
            config = layer["config"]
            padding = calculate_same_padding(
                config["kernel_size"], config["dilation_rate"]
            )
            conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=config["filters"],
                kernel_size=config["kernel_size"],
                stride=config["strides"],
                padding=padding,
                dilation=config["dilation_rate"],
            )

            # Load weights and biases
            weight_shape = conv.weight.shape
            # [config["filters"], in_channels] + config[
            #     "kernel_size"
            # ]
            # putting the shape into tfjs order
            weight_shape = [weight_shape[i] for i in (2, 3, 4, 1, 0)]
            bias_shape = conv.bias.shape  # [config["filters"]]

            weight_size = np.prod(weight_shape)
            bias_size = np.prod(bias_shape)

            weight = weights_data[
                weight_index : weight_index + weight_size
            ].reshape(weight_shape)
            # weight = np.transpose(weight, (2, 3, 4, 1, 0))
            # restoring pytorch order
            weight = np.transpose(weight, (4, 3, 0, 1, 2))
            weight_index += weight_size

            bias = weights_data[
                weight_index : weight_index + bias_size
            ].reshape(bias_shape)
            weight_index += bias_size

            conv.weight.data = torch.from_numpy(weight)
            conv.bias.data = torch.from_numpy(bias)

            layers.append(conv)

            # Update in_channels for the next layer
            in_channels = config["filters"]

        elif layer["class_name"] == "Activation":
            activation = create_activation(layer["config"]["activation"])
            layers.append(activation)

    return nn.Sequential(*layers)


def tfjs_to_pytorch(json_path, bin_path):
    model_spec, weights_data = load_tfjs_model(json_path, bin_path)
    pytorch_model = create_pytorch_model(model_spec, weights_data)
    return pytorch_model


def export_to_onnx(model, input_shape, onnx_path):
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


# Usage
json_path = "model.json"
bin_path = "model.bin"
onnx_path = "model.onnx"

pytorch_model = tfjs_to_pytorch(json_path, bin_path)

# Assuming input shape is [batch_size, channels, depth, height, width]
# input_shape = (1, 1, 256, 256, 256)  # Modify as needed


# export_to_onnx(pytorch_model, input_shape, onnx_path)
def crop_tensor(tensor, percentile=10):
    # Create a copy of the tensor to avoid modifying the original
    data_for_processing = tensor.copy()

    # Thresholding (assuming background has very low values compared to the head)
    threshold = np.percentile(data_for_processing, percentile)
    data_for_processing[data_for_processing < threshold] = 0

    # Find the bounding box around the head (non-zero region) in the filtered data
    indices = np.nonzero(data_for_processing)
    min_z, max_z = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    min_x, max_x = np.min(indices[2]), np.max(indices[2])

    # Crop the original tensor using the bounding box from the filtered data
    cropped_tensor = tensor[
        min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
    ]

    return cropped_tensor


def plot_tensor_slices(tensor, slice_dim=0, cmap="viridis", crop_percentile=10):
    # Crop the tensor
    cropped_tensor = crop_tensor(tensor, percentile=crop_percentile)

    # Determine the dimensions of the cropped tensor
    dim0, dim1, dim2 = cropped_tensor.shape

    # Determine the slicing dimensions based on the specified slice_dim
    if slice_dim == 0:
        num_slices = dim0
        slice_shape = (dim1, dim2)
    elif slice_dim == 1:
        num_slices = dim1
        slice_shape = (dim0, dim2)
    elif slice_dim == 2:
        num_slices = dim2
        slice_shape = (dim0, dim1)
    else:
        raise ValueError("Invalid slice_dim. Must be 0, 1, or 2.")

    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_slices)))

    # Create a larger matrix to hold the slices
    R = np.zeros((grid_size * slice_shape[0], grid_size * slice_shape[1]))

    # Iterate over the slices and place them in the larger matrix
    for i in range(grid_size):
        for j in range(grid_size):
            slice_index = i * grid_size + j
            if slice_index < num_slices:
                if slice_dim == 0:
                    slice_data = cropped_tensor[slice_index, :, :]
                elif slice_dim == 1:
                    slice_data = cropped_tensor[:, slice_index, :]
                else:  # slice_dim == 2
                    slice_data = cropped_tensor[:, :, slice_index]
                R[
                    i * slice_shape[0] : (i + 1) * slice_shape[0],
                    j * slice_shape[1] : (j + 1) * slice_shape[1],
                ] = slice_data

    # Plot the larger matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(R, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


img1 = nib.load("t1_crop.nii.gz")
img = np.asanyarray(img1.dataobj)
input1 = normalize(torch.from_numpy(img).float().unsqueeze(0))
# Convert the tensor to a NumPy array
input1_np = input1.squeeze().numpy()
nifti_img = nib.Nifti1Image(input1_np, np.eye(4)) 
nib.save(nifti_img, 'norm.nii')

with torch.no_grad():
    for ll, layer in enumerate(pytorch_model):
        input_ = layer(input1)
        del input1
        input1 = input_

del input_
# Convert the tensor to a NumPy array
input1_np = input1.squeeze().numpy()
input1_reordered = np.transpose(input1_np, (1, 2, 3, 0))  # Reorder to [x, y, z, v]
nifti_img = nib.Nifti1Image(input1_reordered, np.eye(4))
nib.save(nifti_img, 'result4D.nii')

result = np.squeeze(torch.argmax(input1, 0).cpu().numpy()).astype(np.uint8)
nifti_img = nib.Nifti1Image(result, np.eye(4))
nib.save(nifti_img, 'resultArgmax.nii')
plot_tensor_slices(result, 0)
