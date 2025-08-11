import torch
import torch.nn as nn
from src.vgg import VGG16

class Encoder(nn.Module):
    def __init__(self, n_landmarks=10):
        super(Encoder, self).__init__()
        self.n_landmarks = n_landmarks
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1), nn.ReLU(True)
        )
        self.fc = nn.Linear(256 * 2 * 2, self.n_landmarks * 2)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Reshape to (batch_size, n_landmarks, 2)
        x = x.view(x.size(0), self.n_landmarks, 2)
        return x

class Generator(nn.Module):
    def __init__(self, n_landmarks=10):
        super(Generator, self).__init__()
        # The generator input is the image features + the landmark heatmap.
        # Original features are 512, heatmap is 1 channel. Total 513.
        self.main = nn.Sequential(
            nn.ConvTranspose2d(513, 512, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh() # Tanh for [-1, 1] output
        )
        self.feature_extractor = nn.Sequential(
             nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),
             nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
             nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),
             nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(True),
             nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(True),
        )

    def forward(self, image, landmarks_heatmap):
        features = self.feature_extractor(image)
        # Concatenate image features with landmark heatmaps
        x = torch.cat([features, landmarks_heatmap], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 8, 1, 0) # Output a single value (real/fake)
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(x.size(0), -1)

class IMM(nn.Module):
    def __init__(self, n_landmarks=10, lambda_perceptual=0.1):
        super(IMM, self).__init__()
        self.n_landmarks = n_landmarks
        self.lambda_perceptual = lambda_perceptual
        
        self.encoder = Encoder(n_landmarks=n_landmarks)
        self.generator = Generator(n_landmarks=n_landmarks)
        self.discriminator = Discriminator()
        self.vgg16 = VGG16(requires_grad=False)
        
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, source_image):
        """
        The main inference path: image -> landmarks -> heatmaps -> reconstructed_image
        """
        predicted_landmarks = self.encoder(source_image)
        # Scale landmarks from [-1, 1] (encoder output) to [0, H] image coords
        landmarks_scaled = (predicted_landmarks * 0.5 + 0.5) * source_image.size(2)
        landmarks_heatmap = create_heatmap_tensor(landmarks_scaled, source_image.shape[2:])
        reconstructed_image = self.generator(source_image, landmarks_heatmap)
        return reconstructed_image, predicted_landmarks

    def calculate_generator_loss(self, source_image, reconstructed_image):
        # Adversarial Loss (how well we fooled the discriminator)
        fake_pred = self.discriminator(reconstructed_image)
        g_adv_loss = self.adv_loss(fake_pred, torch.ones_like(fake_pred))

        # Perceptual Loss (how similar the reconstruction is to the original)
        real_vgg_feats = self.vgg16(source_image)
        fake_vgg_feats = self.vgg16(reconstructed_image)
        
        perceptual_loss = 0
        for real_feat, fake_feat in zip(real_vgg_feats.values(), fake_vgg_feats.values()):
            perceptual_loss += self.l2_loss(fake_feat, real_feat)
        
        total_g_loss = g_adv_loss + self.lambda_perceptual * perceptual_loss
        return total_g_loss