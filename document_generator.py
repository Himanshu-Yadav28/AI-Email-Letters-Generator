"""
AI Document Generator using Local LLM
Runs completely offline using Hugging Face transformers

Installation:
pip install transformers torch sentencepiece
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DocumentGeneratorChatbot:
    def __init__(self):
        """Initialize with TinyLlama model (1.1B parameters)"""
        print("=" * 70)
        print("🚀 AI DOCUMENT GENERATOR")
        print("=" * 70)
        print("\n📦 Loading AI model (TinyLlama)...")
        print("⏳ First time setup: Downloading model (~2GB)")
        print("   This happens only once. Please wait...\n")
        
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print(f"✅ Model loaded successfully on {self.device.upper()}!")
            print("🎉 Ready to generate documents!\n")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Make sure you have installed: pip install transformers torch sentencepiece")
            raise
    
    def create_prompt(self, user_request):
        """Create a structured prompt for document generation"""
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
    
    def generate_document(self, user_prompt):
        """Generate document using local LLM"""
        try:
            print("\n🤖 AI is writing your document...\n")
            
            full_prompt = self.create_prompt(user_prompt)
            
            # Generate text
            output = self.generator(
                full_prompt,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Extract generated document
            generated_text = output[0]['generated_text']
            document = generated_text.replace(full_prompt, '').strip()
            
            # Clean up output
            document = self.clean_output(document)
            
            return document
            
        except Exception as e:
            return f"❌ Error generating document: {str(e)}"
    
    def clean_output(self, text):
        """Clean and format the generated output"""
        text = text.strip()
        
        # Remove incomplete sentences at the end
        if text and not text[-1] in '.!?':
            last_sentence_end = max(
                text.rfind('.'),
                text.rfind('!'),
                text.rfind('?')
            )
            if last_sentence_end > len(text) * 0.7:
                text = text[:last_sentence_end + 1]
        
        return text
    
    def chat(self):
        """Main chat interface"""
        print("=" * 70)
        print("💡 EXAMPLES - Try these prompts:")
        print("=" * 70)
        print("  • Write a leave letter for 2 days sick leave")
        print("  • Create a formal email about project delay")
        print("  • Generate a blog post about artificial intelligence")
        print("  • Write a job application for software engineer")
        print("  • Create a resignation letter with 2 weeks notice")
        print("  • Write a complaint letter about poor service")
        print("  • Generate a business report on sales performance")
        print("  • Create an invitation for a birthday party")
        print("\n" + "=" * 70)
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                # Get user input
                user_input = input("📝 Your prompt: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thank you for using AI Document Generator!")
                    break
                
                # Generate document
                document = self.generate_document(user_input)
                
                # Display result
                print("\n" + "=" * 70)
                print("📄 GENERATED DOCUMENT:")
                print("=" * 70)
                print(document)
                print(type(document))   
                print("=" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

def main():
    """Main function to run the chatbot"""
    try:
        # Initialize chatbot (downloads model on first run)
        bot = DocumentGeneratorChatbot()
        
        # Start chat interface
        bot.chat()
        
    except Exception as e:
        print(f"\n❌ Failed to start: {e}")
        print("\n💡 Installation command: pip install transformers torch sentencepiece")

if __name__ == "__main__":
    main()