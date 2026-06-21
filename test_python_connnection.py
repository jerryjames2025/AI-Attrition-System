from ollama import chat

response = chat(
    model="phi3",
    messages=[
        {
            "role":"user",
            "content":"What is employee attrition?"
        }
    ]
)

print(response["message"]["content"])