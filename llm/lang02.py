from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

completion = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages=[{
        'role':'user',
        'temperature':0.9,  # 얼마나 창작할거야?
        'content':'버락 오바마에 대해 알려줘'
    }]
)

print(completion)
print("="*20)
print(completion.choices[0].message.content)