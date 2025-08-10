# In src/imm_model.py

import torch
import torch.nn as nn
from .modules import conv_block, deconv_block, linear_block
from .vgg16 import VGG16

class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_filters=64):
        super().__init__()
        # Input: (N, 3, 128, 128)
        self.block1 = conv_block(in_channels, n_filters, 4, 2, 1, activation='leaky_relu') # (N, 64, 64, 64)
        self.block2 = conv_block(n_filters, n_filters * 2, 4, 2, 1, activation='leaky_relu', use_batchnorm=True) # (N, 128, 32, 32)
        self.block3 = conv_block(n_filters * 2, n_filters * 4, 4, 2, 1, activation='leaky_relu', use_batchnorm=True) # (N, 256, 16, 16)
        self.block4 = conv_block(n_filters * 4, n_filters * 8, 4, 2, 1, activation='leaky_relu', use_batchnorm=True) # (N, 512, 8, 8)
        self.block5 = conv_block(n_filters * 8, n_filters * 8, 4, 2, 1, activation='leaky_relu', use_batchnorm=True) # (N, 512, 4, 4)
        self.block6 = conv_block(n_filters * 8, n_filters * 8, 4, 2, 1, activation='none', use_batchnorm=True) # (N, 512, 2, 2)

    def forward(self, x):
        out = self.block6(self.block5(self.block4(self.block3(self.block2(self.block1(x))))))
        return out

# In src/imm_model.py (continued)

class Generator(nn.Module):
    def __init__(self, out_channels=3, n_filters=64, n_landmarks=5):
        super().__init__()
        # The input to the generator is the feature vector from the encoder
        # plus a heatmap representation of the landmarks.
        # The original paper uses heatmaps, so we add n_landmarks to the input channels.
        in_channels = n_filters * 8 + n_landmarks

        # Input: (N, 512 + 5, 2, 2)
        self.block1 = deconv_block(in_channels, n_filters * 8, 4, 2, 1, use_batchnorm=True) # (N, 512, 4, 4)
        self.block2 = deconv_block(n_filters * 8, n_filters * 8, 4, 2, 1, use_batchnorm=True) # (N, 512, 8, 8)
        self.block3 = deconv_block(n_filters * 8, n_filters * 4, 4, 2, 1, use_batchnorm=True) # (N, 256, 16, 16)
        self.block4 = deconv_block(n_filters * 4, n_filters * 2, 4, 2, 1, use_batchnorm=True) # (N, 128, 32, 32)
        self.block5 = deconv_block(n_filters * 2, n_filters, 4, 2, 1, use_batchnorm=True) # (N, 64, 64, 64)
        self.block6 = deconv_block(n_filters, out_channels, 4, 2, 1, activation='tanh') # (N, 3, 128, 128)

    def forward(self, x, landmarks):
        # Create landmark heatmaps and concatenate them to the input
        heatmap = self.create_heatmap(landmarks, x.shape)
        x = torch.cat([x, heatmap], dim=1)
        
        out = self.block6(self.block5(self.block4(self.block3(self.block2(self.block1(x))))))
        return out
    
    def create_heatmap(self, landmarks, feature_shape):
        """ Creates a gaussian heatmap for each landmark. """
        N, _, H, W = feature_shape
        heatmaps = torch.zeros(N, landmarks.shape[1], H, W, device=landmarks.device)
        
        # Simple delta function for heatmap - can be replaced with a 2D Gaussian
        for i in range(N):
            for j in range(landmarks.shape[1]):
                # Normalize landmarks from image coordinates to feature map coordinates
                lm_x = (landmarks[i, j, 0] / 128 * W).long()
                lm_y = (landmarks[i, j, 1] / 128 * H).long()
                if 0 <= lm_y < H and 0 <= lm_x < W:
                    heatmaps[i, j, lm_y, lm_x] = 1
        return heatmaps

# In src/imm_model.py (continued)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, n_filters=64):
        super().__init__()
        # Input: (N, 3, 128, 128)
        self.block1 = conv_block(in_channels, n_filters, 4, 2, 1, activation='leaky_relu') # (N, 64, 64, 64)
        self.block2 = conv_block(n_filters, n_filters * 2, 4, 2, 1, activation='leaky_relu', use_batchnorm=True) # (N, 128, 32, 32)
        self.block3 = conv_block(n_filters * 2, n_filters * 4, 4, 2, 1, activation='leaky_relu', use_batchnorm=True) # (N, 256, 16, 16)
        self.block4 = conv_block(n_filters * 4, n_filters * 8, 4, 2, 1, activation='leaky_relu', use_batchnorm=True) # (N, 512, 8, 8)
        
        # Flatten and produce a single logit
        self.final_conv = conv_block(n_filters * 8, 1, 8, 1, 0, activation='none') # (N, 1, 1, 1)

    def forward(self, x):
        out = self.block4(self.block3(self.block2(self.block1(x))))
        out = self.final_conv(out)
        return out.view(out.size(0), -1) # Flatten to (N, 1)

# In src/imm_model.py (continued)

class IMM(nn.Module):
    def __init__(self, n_landmarks=5, lambda_l2=1.0, lambda_perceptual=0.1):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.lambda_l2 = lambda_l2
        self.lambda_perceptual = lambda_perceptual
        
        self.encoder = Encoder()
        self.generator = Generator(n_landmarks=n_landmarks)
        self.discriminator = Discriminator()
        self.vgg16 = VGG16(requires_grad=False)
        
        # Loss functions
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, source_image, target_landmarks):
        """
        Generates an image. This is the main inference path.
        """
        encoded_source = self.encoder(source_image)
        generated_image = self.generator(encoded_source, target_landmarks)
        return generated_image

    def calculate_generator_loss(self, source_image, target_image, target_landmarks):
        if generated_image is None:
            encoded_source = self.encoder(source_image)
            generated_image = self.generator(encoded_source, target_landmarks)
        # 1. Reconstruct the image to train the autoencoder part
        encoded_source = self.encoder(source_image)
        reconstructed_image = self.generator(encoded_source, target_landmarks)

        # 2. Adversarial Loss (fool the discriminator)
        fake_pred = self.discriminator(reconstructed_image)
        # We want the discriminator to think this is a real image (label=1)
        g_adv_loss = self.adv_loss(fake_pred, torch.ones_like(fake_pred))

        # 3. Perceptual Loss
        real_vgg_feats = self.vgg16(target_image)
        fake_vgg_feats = self.vgg16(reconstructed_image)
        perceptual_loss = self.l2_loss(fake_vgg_feats['conv4_3'], real_vgg_feats['conv4_3'])
        
        # 4. Landmark Loss (if applicable - for supervised pre-training)
        # In the unsupervised case, this loss might be applied differently or not at all
        # For now, let's assume we can calculate it for auto-encoding reconstruction
        # This part requires a separate landmark predictor, which we can add later.
        # For simplicity, we'll skip the landmark prediction loss for this step.

        total_g_loss = g_adv_loss + self.lambda_perceptual * perceptual_loss
        return total_g_loss, reconstructed_image

    def calculate_discriminator_loss(self, target_image, reconstructed_image):
        # Detach the reconstructed image so we don't backprop through the generator
        real_pred = self.discriminator(target_image)
        fake_pred = self.discriminator(reconstructed_image.detach())
        
        # We want the discriminator to identify real as 1 and fake as 0
        real_loss = self.adv_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.adv_loss(fake_pred, torch.zeros_like(fake_pred))
        
        total_d_loss = (real_loss + fake_loss) * 0.5
        return total_d_loss
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
    
    # In src/imm_model.py, add this method inside the IMM class

    def discover_landmarks(self, image, n_iterations=50):
        """
        Discovers landmarks in a new image via optimization.
        This is a simplified version of the paper's search method.
        """
        self.eval() # Set model to evaluation mode

        # Start with a random guess for the landmarks
        # Landmarks are in (x, y) format, scaled between 0 and 128
        initial_landmarks = torch.rand(image.size(0), self.n_landmarks, 2, device=image.device) * 128
        landmarks = nn.Parameter(initial_landmarks)
        
        # Use an optimizer to fine-tune the landmark positions
        optimizer = torch.optim.Adam([landmarks], lr=1.0)

        # Pre-calculate encoded features and VGG features once
        encoded_source = self.encoder(image)
        target_vgg_feats = self.vgg16(image)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            
            # Generate an image using the current landmark estimates
            generated_image = self.generator(encoded_source, landmarks)

            # Calculate reconstruction loss (L2 + Perceptual)
            l2_loss_val = self.l2_loss(generated_image, image)
            generated_vgg_feats = self.vgg16(generated_image)
            perceptual_loss_val = self.l2_loss(generated_vgg_feats['conv4_3'], target_vgg_feats['conv4_3'])
            
            # This is the loss we want to minimize by moving the landmarks
            reconstruction_loss = self.lambda_l2 * l2_loss_val + self.lambda_perceptual * perceptual_loss_val
            
            reconstruction_loss.backward()
            optimizer.step()
            
            # Keep landmarks within the image bounds (0, 128)
            landmarks.data.clamp_(0, 128)

        self.train() # Set model back to training mode
        return landmarks.detach()