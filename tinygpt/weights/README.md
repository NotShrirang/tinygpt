# Weights Folder

This folder contains pre-trained model weights used by the TinyGPT project. These weights are essential for the model's inference and fine-tuning processes.

## Contents

- **Model Weights**: `.pt` files storing the parameters of the trained TinyGPT model.
- **Metadata**: Any additional files providing context or details about the weights, such as training configuration or dataset information.

## Usage

To use the weights in the TinyGPT project:

1. Ensure the weights are placed in this directory.
2. Load the weights in your script using the appropriate model loading function.

Example:

```python
from tinygpt.model import TinyGPT

model = TinyGPT()
model.load_weights('weights/your_weights_file.pt')
```

## Notes

- Ensure compatibility between the weights and the model version.
- For training or fine-tuning, refer to the main project documentation.

For more details, visit the [TinyGPT Documentation](../README.md).
