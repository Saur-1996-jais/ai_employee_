# Configuration of anthropics
from anthropic import Anthropic
from decouple import config

client = Anthropic(
    api_key=config('CLAUDE_API_KEY'),
)

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Give 5 names of animals",
        }
    ],
    model=config("ANTHROPIC_MODEL")
)

print(message.content)