from openai import OpenAI
from dotenv import load_dotenv

# load_dotenv()
""" 

시스템 환경 변수 편집 -> 계정의 환경 변수 편집
사용자 변수에 '새로 만들기'
변수이름 : OPENAI_API_KEY
변수값: sk- ~~~
재부팅 반드시

"""
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