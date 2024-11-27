# Multi-Task Transformer

## Overview
This project features a **Multi-Task Transformer** model that performs:
1. **Vision Tasks** (e.g., image classification)
2. **Language Tasks** (e.g., sentiment analysis or text classification)

It also integrates with **Groq API** to generate text based on user prompts. The model is designed for learning, prototyping, and exploring AI in real-world applications.

---

## Features
- Shared Transformer architecture for both vision and language tasks.
- Task-specific outputs for better performance.
- Supports text generation using Groq API.
- Easy training, saving, and loading in **Google Colab**.

---

## Requirements
- Python 3.8+
- Libraries: `torch`, `groq`, `numpy`, `matplotlib`

Install required libraries with:
```bash
pip install torch groq numpy matplotlib
Setup in Google Colab
Mount Google Drive:
python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Copy the script into a Colab notebook.
Install required libraries using the command above.
How to Use
Run the script to access the menu:

Set Groq API Key: Input your API key for text generation.
Train the Model: Use vision and language data to train.
Generate Text: Provide a prompt for text generation.
Save/Load Model: Save or load models to/from Google Drive.
Exit: Quit the program.

Applications
Education: AI tutors for images and text.
E-commerce: Analyze reviews and classify products.
Customer Support: Handle both image and text data.


Future Improvements
Add support for multi-modal tasks (e.g., image captioning).
Train on larger datasets for better accuracy.
Deploy as a web app for broader access.
License
Free to use and modify under the MIT License.