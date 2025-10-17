import torch


class BaseModel(torch.nn.Module):
    """
    BaseModel
    ---------
    A lightweight base class that extends torch.nn.Module and provides a
    convenience method to load model weights from a checkpoint file.

    This helper handles both:
    - Plain state_dict checkpoints
    - Checkpoints saved as {"model": state_dict, "optimizer": ..., ...}
    """

    def load(self, path: str) -> None:
        """
        Load model parameters from a checkpoint file.

        Args:
            path (str): File path to a PyTorch checkpoint. This can be either:
                        - A raw state_dict (i.e., the direct output of model.state_dict()), or
                        - A dictionary that contains a "model" key (and possibly others like "optimizer").

        Behavior:
            - Loads the checkpoint onto CPU by default (map_location='cpu') to ensure portability.
            - If the loaded checkpoint contains an "optimizer" key, it assumes the "model" key
              holds the actual state_dict and uses that for loading.
            - Calls self.load_state_dict(...) to apply parameters to the current module.

        Raises:
            RuntimeError: If the state_dict keys do not match the model architecture.
            FileNotFoundError: If the given path does not exist.
        """
        # Load the checkpoint to CPU for maximum compatibility across devices
        parameters = torch.load(path, map_location=torch.device('cpu'))

        # Handle checkpoints that wrap the actual state_dict in a dictionary
        # with additional training artifacts (e.g., optimizer state, epoch).
        if isinstance(parameters, dict) and "optimizer" in parameters:
            parameters = parameters["model"]

        # Apply the loaded state_dict to this module
        self.load_state_dict(parameters)
