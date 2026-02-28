import onnxruntime_genai as oga
import os

class AMDHybridLLM:
    def __init__(self, model_path="../models/local_llm"):
        """
        Initializes the ONNX Runtime GenAI (OGA) model.
        The genai_config.json in the model folder routes workloads 
        to the AMD XDNA NPU and RDNA iGPU.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please download the AMD Ryzen AI ONNX model.")
            
        print("Loading model into AMD Ryzen AI Subsystem...")
        self.model = oga.Model(model_path)
        self.tokenizer = oga.Tokenizer(self.model)
        print("Model loaded successfully.")

    def generate_response(self, prompt, max_length=512):
        """Generates text locally using AMD hardware acceleration."""
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        input_tokens = self.tokenizer.encode(formatted_prompt)
        
        params = oga.GeneratorParams(self.model)
        params.set_search_options(max_length=max_length, past_present_share_buffer=True)
        params.input_ids = input_tokens

        generator = oga.Generator(self.model, params)
        
        response = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()
            response += self.tokenizer.decode([new_token])
            
        return response
