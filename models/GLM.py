import torch
import torch.nn as nn


class GLM(nn.Module):
    """
    A Generalized Likehood Merger (GLM) that applies four element-wise operations
    (max, mean, sum, product) to input tensors, each weighted by learnable parameters.
    """

    def __init__(self):
        super(GLM, self).__init__()

        # Learnable weights for combining different operations
        self.max_weight = nn.Parameter(torch.randn(1))
        self.sum_weight = nn.Parameter(torch.randn(1))
        self.mul_weight = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """
        Computes a weighted combination of max, mean, sum, and product operations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, rows, cols).

        Returns:
            torch.Tensor: The weighted sum of the computed operations applied to x.
        """
        # Create a mask to identify columns with non-zero values
        mask = torch.any(x != 0, dim=1, keepdim=True)  # Shape (batch, 1, cols)

        max_x = self.compute_max(x)
        sum_x = self.compute_sum(x)
        mul_x = self.compute_mul(x, mask)

        # Combine results using learnable weights
        result = (self.max_weight * max_x +
                  self.sum_weight * sum_x + self.mul_weight * mul_x)
        return result

    def compute_max(self, x):
        """
        Computes the maximum value along the last dimension (column-wise max for each row).

        Args:
            x (torch.Tensor): Input tensor of shape (batch, rows, cols).

        Returns:
            torch.Tensor: Tensor of shape (batch, rows) containing max values per row.
        """
        return torch.max(x, dim=-1).values



    def compute_sum(self, x):
        """
        Computes the sum along the last dimension (column-wise sum for each row).

        Args:
            x (torch.Tensor): Input tensor of shape (batch, rows, cols).

        Returns:
            torch.Tensor: Tensor of shape (batch, rows) containing sum values per row.
        """
        return torch.sum(x, dim=-1)

    def compute_mul(self, x, mask):
        """
        Computes the product of non-masked elements along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, rows, cols).
            mask (torch.Tensor): Boolean mask of shape (batch, 1, cols) indicating valid columns.

        Returns:
            torch.Tensor: Tensor of shape (batch, rows) containing product values per row.
        """
        x_masked = x.clone()
        mask = mask.expand(-1, x_masked.shape[1], -1)
        x_masked[~mask] = 1  # Set masked elements to 1 (neutral element for multiplication)
        return torch.prod(x_masked, dim=-1)


if __name__ == "__main__":
    # Test cases
    torch.manual_seed(42)  # For reproducibility

    # Example input tensor of shape (batch, rows, cols)
    x = torch.tensor([
        [[1, 2, 0], [0, 0, 0], [4, 5, 0]],
        [[0, 2, 7], [0, 0, 0], [0, 5, 3]]  # All zeros
    ], dtype=torch.float32)

    model = GLM()
    mask = torch.any(x != 0, dim=1, keepdim=True)
    print("Input Tensor:")
    print(x)

    print("\nMax values:")
    print(model.compute_max(x))

    print("\nSum values:")
    print(model.compute_sum(x))

    print("\nProduct values:")
    print(model.compute_mul(x, mask))

    print("\nFinal Output:")
    print(model(x))
