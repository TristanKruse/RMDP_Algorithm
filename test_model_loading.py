import torch
import os
from pathlib import Path

model_path = "data/models/rl_aca_phase1_final.pt"

print("=== MODEL FILE VALIDATION ===")
print(f"File path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Try to load the model
    try:
        print("\nAttempting to load model...")
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model loaded successfully!")
        
        print(f"\nModel type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print("Model keys:", list(model_data.keys()))
            
            # Check for common model components
            if 'model_state_dict' in model_data:
                print("✅ Found model_state_dict")
            if 'optimizer_state_dict' in model_data:
                print("✅ Found optimizer_state_dict")
            if 'exploration_rate' in model_data:
                print(f"✅ Exploration rate: {model_data['exploration_rate']}")
            if 'total_training_steps' in model_data:
                print(f"✅ Training steps: {model_data['total_training_steps']}")
                
        else:
            print("⚠️ Model is not a dictionary - unusual format")
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print(f"Error type: {type(e)}")
        
else:
    print("❌ Model file does not exist!")