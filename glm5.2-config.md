
Quick Start
Drop-in code to call this model with OpenRouter's OpenAI-compatible API.

1
Get your API key
Create an API key from your OpenRouter dashboard and set it as an environment variable:

Create API Key

Copy
export OPENROUTER_API_KEY=sk-or-v1-...
2
Make your first request
Use z-ai/glm-5.2:free with the OpenRouter API:

OpenRouter supports reasoning-enabled models that can show their step-by-step thinking process. Use the reasoning parameter in your request to enable reasoning, and access the reasoning_details array in the response to see the model's internal reasoning before the final answer. When continuing a conversation, preserve the complete reasoning_details when passing messages back to the model so it can continue reasoning from where it left off. Learn more about reasoning tokens.

In the examples below, the OpenRouter-specific headers are optional. Setting them allows your app to appear on the OpenRouter leaderboards.

TypeScript SDK
Python
TypeScript (fetch)
cURL
Python (OpenAI)
TypeScript (OpenAI)

Copy
import requests
import json

# First API call with reasoning
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "z-ai/glm-5.2:free",
    "messages": [
        {
          "role": "user",
          "content": "How many r's are in the word 'strawberry'?"
        }
      ],
    "reasoning": {"enabled": True}
  })
)

# Extract the assistant message with reasoning_details
response = response.json()
response = response['choices'][0]['message']

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.get('content'),
    "reasoning_details": response.get('reasoning_details')  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  data=json.dumps({
    "model": "z-ai/glm-5.2:free",
    "messages": messages,  # Includes preserved reasoning_details
    "reasoning": {"enabled": True}
  })
)
Using third-party SDKs
For information about using third-party SDKs and frameworks with OpenRouter, please see our frameworks documentation.

3
Enable streaming
Add "stream": true to your request body to receive responses as server-sent events:


Copy
curl -N https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -d '{
  "model": "z-ai/glm-5.2:free",
  "stream": true,
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}'
Endpoint
Sends a request for a model response for the given chat conversation. Supports both streaming and non-streaming modes.

POST
https://openrouter.ai/api/v1/chat/completions
Authorization
Bearer $OPENROUTER_API_KEY
Content-Type
application/json
HTTP-Referer
optional — your site URL, for rankings
X-Title
optional — your site name, for rankings
Model
z-ai/glm-5.2:free
Creates a streaming or non-streaming response using the OpenAI Responses API format.

Docs
POST
https://openrouter.ai/api/v1/responses
Authorization
Bearer $OPENROUTER_API_KEY
Content-Type
application/json
HTTP-Referer
optional — your site URL, for rankings
X-Title
optional — your site name, for rankings
Model
z-ai/glm-5.2:free
Creates a message using the Anthropic Messages API format. Supports text, images, PDFs, tools, and extended thinking.

Docs
POST
https://openrouter.ai/api/v1/messages
Authorization
Bearer $OPENROUTER_API_KEY
Content-Type
application/json
HTTP-Referer
optional — your site URL, for rankings
X-Title
optional — your site name, for rankings
Model
z-ai/glm-5.2:free
Parameters
Name	Type	Default	Description
reasoning	map	—	Controls reasoning behavior for models that support thinking tokens, including whether reasoning is enabled, the reasoning effort, maximum reasoning tokens, and whether reasoning is excluded from the response.
temperature	float	1	This setting influences the variety in the model's responses.
top_p	float	0.95	This setting limits the model's choices to a percentage of likely tokens: only the top tokens whose probabilities add up to P.
top_k	integer	0	This limits the model's choice of tokens at each step, making it choose from a smaller set.
min_p	float	0	Represents the minimum probability for a token to be considered, relative to the probability of the most likely token.
frequency_penalty	float	0	This setting aims to control the repetition of tokens based on how often they appear in the input.
presence_penalty	float	0	Adjusts how often the model repeats specific tokens already used in the input.
repetition_penalty	float	1	Helps to reduce the repetition of tokens from the input.
stop	array	—	Stop generation immediately if the model encounter any token specified in the stop array.
seed	integer	—	If specified, the inferencing will sample deterministically, such that repeated requests with the same seed and parameters should return the same result.
max_tokens	integer	—	This sets the upper limit for the number of tokens the model can generate in response.
response_format	map	—	Forces the model to produce specific output format.
tools	array	—	Tool calling parameter, following OpenAI's tool calling request shape.
tool_choice	string or object	—	Controls which (if any) tool is called by the model.