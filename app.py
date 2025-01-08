
# deal with two csv file and algorithm
# english version
# user give a detailed guide 
from typing import List, Dict
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, Tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

model = ChatOpenAI(model='gpt-4o-2024-08-06')
# model = ChatOpenAI(model='gpt-3.5-turbo')

# csv_files = ["table1_eng.csv", "table2_eng.csv"]
csv_files = ["table1_eng.csv","table2_eng.csv"]
# 模擬台灣各地區pm2.5的資料

pm25_data_location: List[Dict] =  [
    {"sitename":"土城","pm2.5_avg":3.2,"max/avg ratio":2.5},
    {"sitename":"板橋","pm2.5_avg":11.2,"max/avg ratio":9.3},
    {"sitename":"新莊","pm2.5_avg":12.6,"max/avg ratio":5.2},
    {"sitename":"淡水","pm2.5_avg":19.3,"max/avg ratio":8.1},
    {"sitename":"林口","pm2.5_avg":23.1,"max/avg ratio":11},
    {"sitename":"松山","pm2.5_avg":2.5,"max/avg ratio":5.7},
    {"sitename":"古亭","pm2.5_avg":9.4,"max/avg ratio":12.3},
    {"sitename":"萬華","pm2.5_avg":14.1,"max/avg ratio":3.5},
    {"sitename":"中山","pm2.5_avg":15.6,"max/avg ratio":2.1},
    {"sitename":"士林","pm2.5_avg":21.4,"max/avg ratio":4.7},
]


class getPM25Data(BaseTool):
    name: str  = "取得戶外PM2.5 資料"
    description: str  = "當使用者提出安裝地點或是使用清淨機地區，使用此工具，以取得該地點之pm2.5_avg及max/avg ratio"
    def _run(self, location):
        pm25_data={}
        for site in pm25_data_location:
            if location==site["sitename"]:
                pm25_data["pm2.5_avg"]=site["pm2.5_avg"]
                pm25_data["max/avg ratio"]=site["max/avg ratio"]
                print(pm25_data)
                return pm25_data



class EvaluateMathExpression(BaseTool):
    name: str = "Math Evaluation"  # Add type annotation for 'name'
    description: str = 'Use this tool to evaluate a math expression.'  # Add type annotation for 'description'

    def _run(self, expr: str) -> float:  # Annotate the input and output types
        return eval(expr)

    def _arun(self, query: str):
        raise NotImplementedError("Async operation not supported yet")



table_agent_executor = create_csv_agent(
    llm=ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0,),
    path=csv_files,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

algorithm_agent_executor = create_python_agent(
    llm=ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0),
    tool=PythonREPLTool(),
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)



tools=[

    Tool( 
        name="最佳化工具",
        description="""Use this tool when you need to take help of Python interpreter.
        Give the request to the tool in natural language and it will generate the Python code and return the result of the code execution.""",
        func=algorithm_agent_executor.invoke
    ),

    Tool(
        name="table analysis tools",
        description="""Use this tool when you need to answer questions about the table1_eng.csv and table2_eng.csv files. 
        It accepts all question as input and returns the answer after calculating it using the Pandas library.
        """,
        func=table_agent_executor.invoke
    ),
    EvaluateMathExpression(),
    getPM25Data(),
]

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

prompt = hub.pull("hwchase17/structured-chat-agent")

agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    memory=memory, 
    verbose=True, 
    handle_parsing_errors=True
)



response=agent_executor.invoke({"input": """
 I would like to request an ISO level of 8 for an indoor space of 9-Ping, 
 with the installation location in "板橋" and a floor height of 2.9 meters. 
What level of CADR air purifier and how many units are recommended? 
                                
Please follow the steps below:
                            
1.1 Following user location and using getPM25Data tool to get PM2.5 Average Value and PM2.5 Max/Avg Ratio
1.2 In table2_eng.csv file, looking up  ISO value which user asked ,PM2.5 Average Value, and PM2.5 Max/Avg Ratio to  determinte m3 CADR values.                               
1.3. Using Math Evaluation tool to calculate m3 CADR values * indoor space * floor height * 3.3058, the final value is called CADR values_table2

                                
2. get the CADR values corresponding to ISO8, 9 ping in table1_eng.csv
3. compare the value with CADR values_table2. Pick out the larger value. 
4. Treating this larger value as target_CADR. 
5.1 Air purifiers CADR levels are as following,
                       1600 CADR, 1200 CADR, 800 CADR, 600 CADR, 400 CADR, 200 CADR,
                       please  select air purifiers of different levels to achieve a total CADR value greater than or equal to the target_CADR.
                       and implement a dynamic programming algorithm to achieve it.
5.2 I will start by selecting the highest CADR level air purifier and incrementing the quantity until the total CADR value exceeds the target_CADR.  
Then, I will move on to the next lower level air purifier and repeat the process.                                  
5.3 I will keep track of the selected air purifiers and their quantities. 
 
5.4 Finally, I will print out the selected air purifiers and their quantities.
 Before printing out the data, recalculate the CADR provided by the air purifiers, if it is less than the target_CADR, add another 200 CADR air purifier."""})
print("Here is the response: ",response["output"])