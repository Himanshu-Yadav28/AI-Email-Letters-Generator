"""
Flask Web UI for AI Document Generator
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class DocumentGenerator:
    def __init__(self):
        print("🚀 Loading AI model...")
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        print(f"✅ Model loaded on {self.device.upper()}!")
    
    def create_prompt(self, user_request):
        today = datetime.now().strftime('%B %d, %Y')
        
        prompt = f"""You are a professional document writer. Write a complete, well-formatted document based on this request.

Request: {user_request}

Instructions:
- Write ONLY the document, no explanations
- Use proper professional formatting
- Include today's date: {today}
- Use placeholders like [Your Name], [Company Name] for customization
- Make it complete and ready to use

Document:

"""
        return prompt
    
    def generate(self, user_prompt):
        try:
            # Detect document type
            doc_type = self.detect_document_type(user_prompt)
            print(f"\n{'='*60}")
            print(f"📋 Document Type Detected: {doc_type}")
            print(f"📝 User Prompt: {user_prompt}")
            print(f"{'='*60}\n")
            
            full_prompt = self.create_prompt(user_prompt)
            
            output = self.generator(
                full_prompt,
                max_new_tokens=250,  # Reduced from 500
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # Faster generation
            )
            
            generated_text = output[0]['generated_text']
            document = generated_text.replace(full_prompt, '').strip()
            
            # Clean output
            if document and not document[-1] in '.!?':
                last_sentence_end = max(
                    document.rfind('.'),
                    document.rfind('!'),
                    document.rfind('?')
                )
                if last_sentence_end > len(document) * 0.7:
                    document = document[:last_sentence_end + 1]
            
            print(f"✅ Document generated successfully! ({len(document)} characters)\n")
            print(type(document))

            return document
            
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
            return f"Error: {str(e)}"
    
    def detect_document_type(self, prompt):
        """Detect the type of document from the prompt"""
        lower = prompt.lower()
        
        if 'email' in lower:
            if 'formal' in lower:
                return "Formal Email"
            elif 'informal' in lower or 'casual' in lower:
                return "Informal Email"
            return "Email"
        elif 'leave' in lower or 'absence' in lower:
            if 'sick' in lower:
                return "Sick Leave Letter"
            elif 'casual' in lower:
                return "Casual Leave Letter"
            elif 'emergency' in lower:
                return "Emergency Leave Letter"
            return "Leave Application"
        elif 'resignation' in lower or 'resign' in lower:
            return "Resignation Letter"
        elif 'job' in lower and ('application' in lower or 'apply' in lower):
            return "Job Application"
        elif 'complaint' in lower:
            return "Complaint Letter"
        elif 'blog' in lower or 'article' in lower:
            return "Blog Post"
        elif 'report' in lower:
            return "Business Report"
        elif 'invitation' in lower or 'invite' in lower:
            return "Invitation Letter"
        elif 'thank' in lower:
            return "Thank You Letter"
        elif 'cover letter' in lower:
            return "Cover Letter"
        elif 'recommendation' in lower or 'reference' in lower:
            return "Recommendation Letter"
        elif 'proposal' in lower:
            return "Business Proposal"
        else:
            return "General Document"

# Initialize generator (loads model at startup)
print("Initializing Document Generator...")
doc_gen = DocumentGenerator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Please enter a prompt'}), 400
    
    document = doc_gen.generate(prompt)
    return jsonify({'document': document})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': doc_gen.model_name,
        'device': doc_gen.device
    })

@app.route('/api/info', methods=['GET'])
def info():
    """API information endpoint"""
    return jsonify({
        'version': '1.0',
        'model': 'TinyLlama-1.1B-Chat',
        'endpoints': {
            'generate': '/generate (POST)',
            'health': '/health (GET)',
            'info': '/api/info (GET)'
        }
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    # bind to 0.0.0.0 so Railway can reach it
    app.run(host="0.0.0.0", port=port, debug=False)



# """
# Flask Web UI for AI Document Generator
# Run: python app.py
# Then open: http://localhost:5000
# """

# from flask import Flask, render_template, request, jsonify
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# app = Flask(__name__)

# class DocumentGenerator:
#     def __init__(self):
#         print("🚀 Loading AI model...")
#         self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Load tokenizer and model
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name,
#             trust_remote_code=True
#         )
        
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             device_map="auto" if self.device == "cuda" else None,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True
#         )
        
#         if self.device == "cpu":
#             self.model = self.model.to(self.device)
        
#         self.generator = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device=0 if self.device == "cuda" else -1
#         )
        
#         print(f"✅ Model loaded on {self.device.upper()}!")
    
#     def create_prompt(self, user_request):
#         today = datetime.now().strftime('%B %d, %Y')
        
#         prompt = f"""You are a professional document writer. Write a complete, well-formatted document based on this request.

# Request: {user_request}

# Instructions:
# - Write ONLY the document, no explanations
# - Use proper professional formatting
# - Include today's date: {today}
# - Use placeholders like [Your Name], [Company Name] for customization
# - Make it complete and ready to use

# Document:

# """
#         return prompt
    
#     def generate(self, user_prompt):
#         try:
#             full_prompt = self.create_prompt(user_prompt)
            
#             output = self.generator(
#                 full_prompt,
#                 max_new_tokens=500,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True,
#                 num_return_sequences=1,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )
            
#             generated_text = output[0]['generated_text']
#             document = generated_text.replace(full_prompt, '').strip()
            
#             # Clean output
#             if document and not document[-1] in '.!?':
#                 last_sentence_end = max(
#                     document.rfind('.'),
#                     document.rfind('!'),
#                     document.rfind('?')
#                 )
#                 if last_sentence_end > len(document) * 0.7:
#                     document = document[:last_sentence_end + 1]
            
#             return document
            
#         except Exception as e:
#             return f"Error: {str(e)}"

# # Initialize generator (loads model at startup)
# print("Initializing Document Generator...")
# doc_gen = DocumentGenerator()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.json
#     prompt = data.get('prompt', '')
    
#     if not prompt:
#         return jsonify({'error': 'Please enter a prompt'}), 400
    
#     document = doc_gen.generate(prompt)
#     return jsonify({'document': document})

# if __name__ == '__main__':
#     print("\n" + "="*70)
#     print("🌐 Server starting at http://localhost:5000")
#     print("="*70 + "\n")
#     app.run(debug=True, host='0.0.0.0', port=5000)