from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI



tools = [PythonREPLTool()]
agent_executor = create_python_agent(
    llm=ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0),

    # llm=ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0),
    # llm=ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0),
    # llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0,),
    # llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0,),
    # llm=ChatOpenAI(model="", temperature=0),
    # llm=ChatOpenAI(model="", temperature=0),


    tool=PythonREPLTool(),
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)
# agent_executor
# print(agent_executor)
# agent_executor.invoke({"input": "7的2.3次方是多少？"})

agent_executor.invoke({"input": """Please implement a dynamic programming algorithm to select air purifiers of different levels to achieve a total CADR value greater than or equal to the target value.
I will start by selecting the highest CADR level air purifier and incrementing the quantity until the total CADR value exceeds the target.  
Then, I will move on to the next lower level air purifier and repeat the process.                                  
I will keep track of the selected air purifiers and their quantities.                                         
Finally, I will print out the selected air purifiers and their quantities. 

                    
target_CADR = 3037
CADR_levels = [1600, 1200, 800, 600, 400, 200]               
                                         
"""})



# agent_executor.invoke({"input": "第12个斐波那契数列的数字是多少？"})