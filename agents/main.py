from fastapi import FastAPI
from orchestrator.agent_manager import AgentManager

app = FastAPI()

manager = AgentManager()

@app.get("/")
def read_root():
    return {"Hello": "World"}   

@app.get("/run-agent")
def run_agent(company: str, website: str):
    result = manager.run(company, website)
    return result

