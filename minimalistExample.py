"""This is just an simple example, not a real world application"""
import uuid
import json

class SimpleMCP:
    def __init__(self):
        self.models = {"mistral": self.mistral_mock, "gemini": self.gemini_mock}

    def process_request(self, model, text):
        request = self.create_request(model, text)
        response = self.models.get(model, self.default_response)(request)
        return self.format_response(model, response)

    def create_request(self, model, text):
        return {"id": str(uuid.uuid4()), "model": model, "query": text}

    def mistral_mock(self, request):
        return f"Mistral Response: '{request['query']}'"

    def gemini_mock(self, request):
        return f"Gemini Response: '{request['query']}'"

    def default_response(self, request):
        return "Model not available"

    def format_response(self, model, response):
        return json.dumps({"model": model, "response": response}, indent=2)

# Example Usage
mcp = SimpleMCP()
print(mcp.process_request("mistral", "How does AI impact business?"))
print(mcp.process_request("gemini", "Explain quantum computing."))
