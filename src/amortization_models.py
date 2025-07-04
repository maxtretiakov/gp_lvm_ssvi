import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class AmortizedEncoder(nn.Module):
    """
    Neural network for amortized inference of latent variables.
    
    Takes inducing point samples, inducing point inputs, and observations
    and outputs variational parameters mu_x and log_s2x.
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu",
        dropout: float = 0.1,
        use_attention: bool = False,
        attention_heads: int = 4
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input feature computation
        # inducing_point_samples: (num_inducing, output_dim)
        # inducing_point_inputs: (num_u_samples, latent_dim, num_inducing)
        # observations: (batch_shape, output_dim)
        
        # We'll compute features for each component and concatenate
        self.inducing_samples_encoder = nn.Linear(output_dim, hidden_dims[0] // 4)
        self.inducing_inputs_encoder = nn.Linear(latent_dim, hidden_dims[0] // 4)
        self.observations_encoder = nn.Linear(output_dim, hidden_dims[0] // 2)
        
        # Optional attention mechanism for inducing points
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[0] // 4,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Main MLP layers
        layer_dims = [hidden_dims[0]] + hidden_dims[1:]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            self.layers.append(nn.Dropout(dropout))
        
        # Output heads for mu and log_s2x
        # Output shape: (batch_shape, num_inducing, latent_dim)
        final_dim = hidden_dims[-1]
        self.mu_head = nn.Linear(final_dim, latent_dim)
        self.log_s2x_head = nn.Linear(final_dim, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _encode_inducing_features(
        self,
        inducing_point_samples: torch.Tensor,
        inducing_point_inputs: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Encode inducing point features
        
        Args:
            inducing_point_samples: (num_inducing, output_dim)
            inducing_point_inputs: (num_u_samples, latent_dim, num_inducing)
            batch_size: size of batch dimension
            
        Returns:
            encoded_features: (batch_size, num_inducing, feature_dim)
        """
        num_inducing = inducing_point_samples.shape[0]
        num_u_samples = inducing_point_inputs.shape[0]
        
        # Encode inducing point samples
        # (num_inducing, output_dim) -> (num_inducing, hidden_dim//4)
        encoded_samples = self.inducing_samples_encoder(inducing_point_samples)
        
        # Encode inducing point inputs
        # (num_u_samples, latent_dim, num_inducing) -> (num_u_samples, num_inducing, hidden_dim//4)
        encoded_inputs = self.inducing_inputs_encoder(
            inducing_point_inputs.transpose(1, 2)
        )
        
        # Aggregate across u_samples (mean pooling)
        # (num_u_samples, num_inducing, hidden_dim//4) -> (num_inducing, hidden_dim//4)
        encoded_inputs = encoded_inputs.mean(dim=0)
        
        # Combine sample and input features
        # (num_inducing, hidden_dim//2)
        inducing_features = torch.cat([encoded_samples, encoded_inputs], dim=-1)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over inducing points
            # (1, num_inducing, hidden_dim//2) -> (1, num_inducing, hidden_dim//2)
            inducing_features = inducing_features.unsqueeze(0)
            attended_features, _ = self.attention(
                inducing_features, inducing_features, inducing_features
            )
            inducing_features = attended_features.squeeze(0)
        
        # Expand to batch size
        # (num_inducing, hidden_dim//2) -> (batch_size, num_inducing, hidden_dim//2)
        inducing_features = inducing_features.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        return inducing_features
    
    def forward(
        self,
        inducing_point_samples: torch.Tensor,
        inducing_point_inputs: torch.Tensor,
        observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of amortized encoder
        
        Args:
            inducing_point_samples: (num_inducing, output_dim)
            inducing_point_inputs: (num_u_samples, latent_dim, num_inducing)
            observations: (batch_shape, output_dim)
            
        Returns:
            mu_x: (batch_shape, num_inducing, latent_dim)
            log_s2x: (batch_shape, num_inducing, latent_dim)
        """
        batch_shape = observations.shape[:-1]
        batch_size = observations.shape[0] if len(batch_shape) == 1 else torch.prod(torch.tensor(batch_shape))
        num_inducing = inducing_point_samples.shape[0]
        
        # Flatten batch dimensions if needed
        if len(batch_shape) > 1:
            observations = observations.reshape(batch_size, -1)
        
        # Encode observations
        # (batch_size, output_dim) -> (batch_size, hidden_dim//2)
        encoded_obs = self.observations_encoder(observations)
        
        # Encode inducing point features
        # (batch_size, num_inducing, hidden_dim//2)
        inducing_features = self._encode_inducing_features(
            inducing_point_samples, inducing_point_inputs, batch_size
        )
        
        # Combine observation and inducing features
        # Broadcast observations to match inducing points
        # (batch_size, 1, hidden_dim//2) -> (batch_size, num_inducing, hidden_dim//2)
        encoded_obs = encoded_obs.unsqueeze(1).expand(-1, num_inducing, -1)
        
        # Concatenate features
        # (batch_size, num_inducing, hidden_dim)
        combined_features = torch.cat([encoded_obs, inducing_features], dim=-1)
        
        # Pass through MLP layers
        x = combined_features
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.activation(x)
            else:  # Dropout
                x = layer(x)
        
        # Generate outputs
        mu_x = self.mu_head(x)  # (batch_size, num_inducing, latent_dim)
        log_s2x = self.log_s2x_head(x)  # (batch_size, num_inducing, latent_dim)
        
        # Reshape back to original batch shape if needed
        if len(batch_shape) > 1:
            mu_x = mu_x.reshape(*batch_shape, num_inducing, self.latent_dim)
            log_s2x = log_s2x.reshape(*batch_shape, num_inducing, self.latent_dim)
        
        # Clamp log_s2x to prevent numerical issues
        log_s2x = torch.clamp(log_s2x, min=-10, max=10)
        
        return mu_x, log_s2x

# Alternative simpler version without attention
class SimpleAmortizedEncoder(nn.Module):
    """Simpler version of amortized encoder without attention mechanism"""
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        
        # Single encoder for concatenated input
        # Input: [inducing_samples_flat, inducing_inputs_flat, observations]
        self.input_dim = None  # Will be set dynamically
        
        # MLP layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.mu_head = None
        self.log_s2x_head = None
        self.hidden_dims = hidden_dims
        
    def _build_network(self, input_dim: int, num_inducing: int):
        """Build network dynamically based on input dimensions"""
        if self.input_dim is not None:
            return  # Already built
            
        self.input_dim = input_dim
        
        # Build MLP
        dims = [input_dim] + self.hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        # Output heads
        final_dim = self.hidden_dims[-1]
        self.mu_head = nn.Linear(final_dim, num_inducing * self.latent_dim)
        self.log_s2x_head = nn.Linear(final_dim, num_inducing * self.latent_dim)
        
        # Initialize
        for module in [self.mu_head, self.log_s2x_head] + list(self.layers):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        inducing_point_samples: torch.Tensor,
        inducing_point_inputs: torch.Tensor,
        observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - simpler concatenation-based approach
        
        Args:
            inducing_point_samples: (num_inducing, output_dim)
            inducing_point_inputs: (num_u_samples, latent_dim, num_inducing)
            observations: (batch_shape, output_dim)
            
        Returns:
            mu_x: (batch_shape, num_inducing, latent_dim)
            log_s2x: (batch_shape, num_inducing, latent_dim)
        """
        batch_shape = observations.shape[:-1]
        batch_size = observations.shape[0]
        num_inducing = inducing_point_samples.shape[0]
        
        # Flatten and concatenate all inputs
        inducing_samples_flat = inducing_point_samples.flatten()  # (num_inducing * output_dim,)
        inducing_inputs_flat = inducing_point_inputs.mean(dim=0).flatten()  # (latent_dim * num_inducing,)
        
        # Create input vector for each batch element
        fixed_input = torch.cat([inducing_samples_flat, inducing_inputs_flat])
        
        # Build network if not already built
        input_dim = fixed_input.shape[0] + self.output_dim
        self._build_network(input_dim, num_inducing)
        
        # Concatenate with observations
        # (batch_size, input_dim)
        batch_input = torch.cat([
            fixed_input.unsqueeze(0).expand(batch_size, -1),
            observations
        ], dim=-1)
        
        # Forward through MLP
        x = batch_input
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        
        # Generate outputs
        mu_flat = self.mu_head(x)  # (batch_size, num_inducing * latent_dim)
        log_s2x_flat = self.log_s2x_head(x)  # (batch_size, num_inducing * latent_dim)
        
        # Reshape to target format
        mu_x = mu_flat.reshape(batch_size, num_inducing, self.latent_dim)
        log_s2x = log_s2x_flat.reshape(batch_size, num_inducing, self.latent_dim)
        
        # Clamp log variance
        log_s2x = torch.clamp(log_s2x, min=-10, max=10)
        
        return mu_x, log_s2x

# Example usage and testing
if __name__ == "__main__":
    # Example dimensions
    latent_dim = 12
    output_dim = 3
    num_inducing = 64
    num_u_samples = 2
    batch_size = 128
    
    # Create model
    model = AmortizedEncoder(
        latent_dim=latent_dim,
        output_dim=output_dim,
        hidden_dims=[256, 128, 64],
        use_attention=True
    )
    
    # Create dummy data
    inducing_point_samples = torch.randn(num_inducing, output_dim)
    inducing_point_inputs = torch.randn(num_u_samples, latent_dim, num_inducing)
    observations = torch.randn(batch_size, output_dim)
    
    # Forward pass
    mu_x, log_s2x = model(inducing_point_samples, inducing_point_inputs, observations)
    
    print(f"Input shapes:")
    print(f"  inducing_point_samples: {inducing_point_samples.shape}")
    print(f"  inducing_point_inputs: {inducing_point_inputs.shape}")
    print(f"  observations: {observations.shape}")
    print(f"Output shapes:")
    print(f"  mu_x: {mu_x.shape}")
    print(f"  log_s2x: {log_s2x.shape}")
    
    # Test with different batch shapes
    observations_2d = torch.randn(32, 4, output_dim)
    mu_x_2d, log_s2x_2d = model(inducing_point_samples, inducing_point_inputs, observations_2d)
    print(f"2D batch output shapes:")
    print(f"  mu_x: {mu_x_2d.shape}")
    print(f"  log_s2x: {log_s2x_2d.shape}")
    
    print("\nModel created successfully!")