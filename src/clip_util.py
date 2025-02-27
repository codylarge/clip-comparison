import clip
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from classes import get_candidate_captions 

class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)

    def encode_text(self, text):
        # Delegate to the base CLIP model's text encoder
        return self.model.encode_text(text)
    
    def encode_image(self, image):
        # Delegate to the base CLIP model's image encoder
        return self.model.encode_image(image)

def load_basic_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def load_custom_clip_model(model_path="clip_finetuned.pth", num_classes=13):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load original CLIP model
    base_model, _ = clip.load("ViT-B/32", device=device)

    # Wrap it in CLIPFineTuner
    model = CLIPFineTuner(base_model, num_classes).to(device)

    # Load the saved state_dict (instead of the full model)
    state_dict = torch.load(model_path, map_location=device)
    
    if 'state_dict' in state_dict:  # If saved inside a dict
        state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict)  # Load weights
    model.to(device)
    model.eval()

    # Define a preprocessing pipeline (must match training-time transforms)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Match model's expected input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])

    #print("Custom mode:", model)
    return model, preprocess, device

'''
def load_custom_clip_model(model_path="clip_finetuned.pth", num_classes=13):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model, _ = clip.load("ViT-B/32", device=device)

    # Get the actual CLIP model class from the loaded model
    CLIP = type(base_model)  # Extract the class dynamically

    # Allow CLIP to be loaded safely
    torch.serialization.add_safe_globals([CLIP])

    # Load the full model
    model = torch.load(model_path, map_location=device, weights_only=False)  
    model.to(device)
    model.eval()

    # Define preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])

    #print("Custom model:", model)
    return model, preprocess, device
'''


# Return vector embeddings for given text (caption)
def get_text_features(captions, device, model):
    text_tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():  
        text_features = model.encode_text(text_tokens)

    return text_features # -> dim: (num_captions, 512)

# Return vector embeddings for given image
def get_image_features(image, device, model, preprocess):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input) 

    return image_features # -> dim: (1, 512)

# Compute cosine similarity of image & text features
def compute_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).squeeze(0)
    
    return similarity

def classify_image(image_path, model, preprocess, device):
    # Load and preprocess the image
    image = Image.open(image_path)
    image_features = get_image_features(image, device, model, preprocess)

    # Get candidate captions and their features
    candidate_captions = get_candidate_captions()
    text_features = get_text_features(candidate_captions, device, model)

    # Compute similarity between image and text features
    similarity = compute_similarity(image_features, text_features)

    # Find the caption with the highest similarity score
    best_match_index = similarity.argmax().item()
    best_caption = candidate_captions[best_match_index]

    # Extract the class name from the best caption
    classified_class = best_caption.split(":")[0]

    return classified_class, best_caption, similarity[best_match_index].item()
