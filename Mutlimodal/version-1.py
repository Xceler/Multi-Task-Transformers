import os
import torch
import torch.nn as nn
import torch.optim as optim
from google.colab import drive
from groq import Groq
import numpy as np

class MultiTaskTransformer(nn.Module):
    def __init__(self, 
                 input_dim=512, 
                 hidden_dim=1024, 
                 num_heads=8, 
                 num_layers=6, 
                 num_classes_vision=10, 
                 num_classes_language=2):
        super(MultiTaskTransformer, self).__init__()
        
  
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads
        )
   
        self.transformer_layers = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads
            ),
            num_layers=num_layers
        )
        
        # Task-specific output heads
        self.vision_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_vision)
        )
        
        self.language_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_language)
        )
        

        self.groq_client = None
    
    def set_groq_client(self, api_key):
     
        self.groq_client = Groq(api_key=api_key)
    
    def forward(self, x, task='vision'):
   
        x = self.embedding(x)
        
     
        attn_output, _ = self.multihead_attn(x, x, x)
        
        
        transformed_x = self.transformer_layers(attn_output)
        
  
        if task == 'vision':
            return self.vision_head(transformed_x.mean(dim=1))
        elif task == 'language':
            return self.language_head(transformed_x.mean(dim=1))
    
    def generate_text(self, prompt, max_length=100):
        """
        Use Groq API for text generation
        """
        if not self.groq_client:
            raise ValueError("Groq API client not set. Use set_groq_client() first.")
        
        completion = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_length
        )
        return completion.choices[0].message.content
    
    def train_multi_task(self, vision_data, language_data, epochs=10):
     
        vision_criterion = nn.CrossEntropyLoss()
        language_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        losses = []
        for epoch in range(epochs):
            # Vision Task Training
            vision_inputs, vision_labels = vision_data
            vision_outputs = self(vision_inputs, task='vision')
            vision_loss = vision_criterion(vision_outputs, vision_labels)
            
            # Language Task Training
            language_inputs, language_labels = language_data
            language_outputs = self(language_inputs, task='language')
            language_loss = language_criterion(language_outputs, language_labels)
            
            # Combined loss
            total_loss = vision_loss + language_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
           
            losses.append(total_loss.item())
            
            print(f"Epoch {epoch+1}: Vision Loss = {vision_loss.item()}, Language Loss = {language_loss.item()}")
        
        return losses

def save_model_to_drive(model, filename='multi_task_transformer.pth'):
 
    drive.mount('/content/drive', force_remount=True)
    
 
    save_path = f'/content/drive/MyDrive/{filename}'
    

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to Google Drive: {save_path}")

def load_model_from_drive(filename='multi_task_transformer.pth'):
    """
    Load model from Google Drive
    """
  
    drive.mount('/content/drive', force_remount=True)
    
 
    load_path = f'/content/drive/MyDrive/{filename}'
    
    model = MultiTaskTransformer()
    
   
    model.load_state_dict(torch.load(load_path))
    model.eval() 
    
    print(f"Model loaded from Google Drive: {load_path}")
    return model

def interactive_colab_training():
    """
    Interactive model training for Google Colab
    """
  
    model = MultiTaskTransformer()

    while True:
        print("\n--- Multi-Task Transformer Colab Interface ---")
        print("1. Set Groq API Key")
        print("2. Train Model")
        print("3. Generate Text")
        print("4. Save Model to Drive")
        print("5. Load Model from Drive")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            # Manually input Groq API Key
            api_key = input("Enter your Groq API Key: ")
            try:
                model.set_groq_client(api_key)
                print("Groq API client successfully set!")
            except Exception as e:
                print(f"Error setting API client: {e}")

        elif choice == '2':
            
            vision_inputs = torch.randint(0, 512, (100, 50))  # 100 samples, sequence length 50
            vision_labels = torch.randint(0, 10, (100,))      # 10 vision classes
            
            language_inputs = torch.randint(0, 512, (100, 50))  # 100 samples, sequence length 50
            language_labels = torch.randint(0, 2, (100,))       # Binary classification
            
        
            epochs = int(input("Enter number of training epochs: "))
            
            # Train the model
            losses = model.train_multi_task(
                (vision_inputs, vision_labels), 
                (language_inputs, language_labels),
                epochs=epochs
            )
            
          
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,5))
            plt.plot(losses)
            plt.title('Training Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()

        elif choice == '3':
          
            if not model.groq_client:
                print("Please set Groq API Key first!")
                continue
            
            prompt = input("Enter a prompt for text generation: ")
            generated_text = model.generate_text(prompt)
            print("\nGenerated Text:\n", generated_text)

        elif choice == '4':
           
            filename = input("Enter filename to save model (default: multi_task_transformer.pth): ") or 'multi_task_transformer.pth'
            save_model_to_drive(model, filename)

        elif choice == '5':
         
            filename = input("Enter filename to load model (default: multi_task_transformer.pth): ") or 'multi_task_transformer.pth'
            try:
                model = load_model_from_drive(filename)
            except FileNotFoundError:
                print("Model file not found in Google Drive. Please check the filename.")

        elif choice == '6':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

def main():
    interactive_colab_training()

if __name__ == "__main__":
    main()