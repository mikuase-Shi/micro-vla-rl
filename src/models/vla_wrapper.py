import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoConfig

class FrozenVLAWrapper(nn.Module):
    """
    A lightweight, frozen vision encoder acting as our Micro-VLA backbone.
    Can use a pre-trained ResNet18 or a HuggingFace model.
    """
    def __init__(self, use_hf_model=False, hf_model_name="openvla/openvla-7b-prismatic"):
        super().__init__()
        self.use_hf_model = use_hf_model
        
        if not use_hf_model:
            # Load a pre-trained ResNet18
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            
            # Remove the final fully connected classification layer to extract features
            modules = list(resnet.children())[:-2] 
            self.backbone = nn.Sequential(*modules)
            
            # Add pooling and flattening to output a fixed flat vector [B, 512]
            self.pool_and_flatten = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.output_dim = 512
        else:
            # Load a real VLA / Vision model from HuggingFace
            # Using trust_remote_code=True for OpenVLA variants
            try:
                # Dummy implementation assuming HuggingFace VisionEncoder fallback
                # In actual OpenVLA, this will load the prismatic vision backbone
                self.backbone = AutoModel.from_pretrained(
                    hf_model_name, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16 # load in half precision to save memory
                )
                self.output_dim = self.backbone.config.hidden_size
            except Exception as e:
                print(f"Failed to load HF model {hf_model_name}, falling back to ResNet18. Error: {e}")
                self.use_hf_model = False
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                modules = list(resnet.children())[:-2] 
                self.backbone = nn.Sequential(*modules)
                self.pool_and_flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
                self.output_dim = 512
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Extracts visual features from an image.
        Args:
            rgb: Input image tensor of shape (Batch, C, H, W)
        Returns:
            features: A flat feature vector of shape (Batch, output_dim)
        """
        if not self.use_hf_model:
            features = self.backbone(rgb)
            flat_features = self.pool_and_flatten(features)
            return flat_features
        else:
            # HuggingFace VisionModel forward pass
            # Ensure proper casting depending on the loaded un-frozen model
            features = self.backbone(pixel_values=rgb.half())
            # Assuming pooled output is available, or take mean over sequence
            if hasattr(features, "pooler_output") and features.pooler_output is not None:
                return features.pooler_output
            else:
                # Mean pool the last hidden state
                return features.last_hidden_state.mean(dim=1)
