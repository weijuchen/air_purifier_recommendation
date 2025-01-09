from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI

agent_executor = create_csv_agent(

    llm=ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0,),
    # llm=ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0,),
    # llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0,),
    # llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0,),
    path="table1_eng.csv",
    # encoding="unicode_escape",
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
    allow_dangerous_code=True
)

agent_executor.invoke({"input": "ISO8 4坪所對應的室內CADR值為何？請用中文回答"})

# agent_executor.invoke({"input": "数据集有多少行？用中文回复"})