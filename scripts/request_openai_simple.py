import base64
from pathlib import Path
from openai import OpenAI


client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="-")
image_path = 'example_files/image_text2.jpg'
image_base64 = base64.b64encode(Path(image_path).read_bytes()).decode()
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            },
            {
                "type": "text",
                "text": "OCR:"
            },
        ],
    }
]
completion = client.chat.completions.create(
    model="-",
    messages=messages,
    temperature=0.0,
    max_tokens=1024,
)
print(completion.choices[0].message.content)