from langchain.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"]="YOUR_OPENAI_API_KEY"

examples = [
    {"input": "India", "output": "aidnI"},
    {"input": "Canada", "output": "adanaC"},
    {"input": "Australia", "output": "ailartsuA"},
]


example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.format())

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a linguistic specialist."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

prompt=prompt_template.format_messages(input="Brazil")

model = ChatOpenAI()
response=model.invoke(prompt)

response.content