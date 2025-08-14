import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import imageio
from torchvision import transforms

LATENT_DIM = 100
LEARNING_RATE = 0.0002
NUM_EPOCHS = 50000

INPUT_FILENAME = "input_video.gif" 
OUTPUT_FILENAME = "generated_video.gif"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

def load_target_video(filename):

    print(f"loading from {filename}")
    reader = imageio.get_reader(filename)
    video_frames = []
    
    try:
        fps = reader.get_meta_data()['fps']
    except KeyError:
        print("Warning: Could not read FPS from GIF metadata. Defaulting to 10.")
        fps = 10

    for frame_data in reader: #type: ignore
        if frame_data.ndim == 2:
            frame_data = np.stack((frame_data,)*3, axis=-1)
        if frame_data.shape[2] == 4:
             frame_data = frame_data[:, :, :3]
        video_frames.append(frame_data)
    
    num_frames = len(video_frames)
    img_height, img_width, num_channels = video_frames[0].shape
    print(f"Video properties: {num_frames} frames, {img_width}x{img_height}, {fps} FPS")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    transformed_frames = [transform(frame) for frame in video_frames]
    video_tensor = torch.stack(transformed_frames)
    video_tensor = (video_tensor * 2.0) - 1.0
    
    return video_tensor.to(device), num_frames, img_height, img_width, num_channels, fps

class VideoGenerator(nn.Module):
    def __init__(self, num_frames, img_height, img_width, num_channels):
        super(VideoGenerator, self).__init__()
        self.num_frames = num_frames
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        
        layers = []
        
        layers.append(nn.ConvTranspose2d(LATENT_DIM, 512, kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(True))

        current_res = 4
        in_channels = 512

        while current_res < max(img_width, img_height) / 2:
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            
            in_channels = out_channels
            current_res *= 2
        
        layers.append(nn.ConvTranspose2d(in_channels, self.num_channels * self.num_frames, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())

        layers.append(nn.Upsample(size=(self.img_height, self.img_width), mode='bilinear', align_corners=False))
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, LATENT_DIM, 1, 1)
        output = self.main(x)
        
        return output.view(-1, self.num_frames, self.num_channels, self.img_height, self.img_width)


def train():
    target_video, num_frames, img_h, img_w, num_channels, fps = load_target_video(INPUT_FILENAME)
    
    fixed_noise = torch.randn(1, LATENT_DIM, device=device)

    generator = VideoGenerator(num_frames, img_h, img_w, num_channels).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        generator.train()
        
        generated_video = generator(fixed_noise)
        loss = criterion(generated_video.squeeze(0), target_video)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}")

    return generator, fixed_noise, fps

def save_video(generator, noise_vector, filename, fps):
    print(f"saving to {filename}...")
    
    generator.eval()
    
    with torch.no_grad():
        generated_video_tensor = generator(noise_vector).squeeze(0).cpu()
        generated_video_tensor = (generated_video_tensor + 1) / 2.0
        frames_np = (generated_video_tensor.permute(0, 2, 3, 1) * 255).byte().numpy()
        
        imageio.mimsave(filename, [frame for frame in frames_np], fps=fps)
        
    print("done")


if __name__ == "__main__":
    trained_generator, input_noise, fps = train()
    save_video(trained_generator, input_noise, OUTPUT_FILENAME, fps)
