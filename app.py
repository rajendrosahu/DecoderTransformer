import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
from huggingface_hub import hf_hub_download
from transformers import GPT, GPTConfig  # Import your model class

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model from Hugging Face Hub
def load_model_from_huggingface():
    # Replace with your Hugging Face model ID (username/model-name)
    model_id = "EzhirkoArulmozhi/DecoderTransformerModel"
    checkpoint_path = hf_hub_download(repo_id=model_id, filename="gpt_checkpoint.pth")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    return model

model = load_model_from_huggingface()
# Force model to stay in eval mode
model.train(False)

def generate_text(prompt, max_length=25, num_samples=1):
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)    
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1)
    tokens = tokens.to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            if tokens.size(1) >= 1024:  # GPT context length
                break
                
            logits = model(tokens)[0]
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, ix)
            
            tokens = torch.cat((tokens, next_token), dim=1)
            
            # Remove special token check entirely
            # Just generate for the specified length or until context limit
    
    generated_texts = []
    for i in range(num_samples):
        text = enc.decode(tokens[i].tolist())
        generated_texts.append(text)
    
    return '\n\n---\n\n'.join(generated_texts)

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", value="How are you doing Raj? Have a wonderful day."),
        gr.Radio(choices=[25, 50, 75, 100, 125], value=100, label="Max Length", type="value"),
        gr.Radio(choices=[1, 2, 3, 4], value=1, label="Number of Samples", type="value"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Dialog Generator",
    description="Enter a prompt to generate a diaglog.",
    examples=[
        ["Lets fix the real issues in the world", 125, 1],
        ["Lets do something about it", 100, 2],
        ["I ahve not seen you today", 75, 3],
        ["I am a bad person", 50, 4],
    ]
)

if __name__ == "__main__":
    iface.launch() 