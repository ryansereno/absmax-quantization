import torch

X = torch.rand((5, 5))
print("Original Tensor:")
print(X)

#where X is an unquantized tensor
def absmax_quantize(X):

    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))

    # Quantize
    X_quant = (scale * X).round()

    # Dequantize
    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), X_dequant

X_quant, X_dequant = absmax_quantize(X)

print("\nQuantized Tensor:")
print(X_quant)

print("\nDequantized Tensor:")
print(X_dequant)
