from openai import OpenAI

OPENAI_API_KEY=0000

client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages=[{
        'role':'user',
        'temperature':0.9,  # 얼마나 창작할거야?
        'content':''
    }]
)

print(completion)
print("="*20)
print(completion)