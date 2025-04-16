# Imports
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Any, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls
from langchain_core.tools import tool
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

@tool
def search_web(query: str) -> str:
    """Search the web and extract plain text from results."""
    safe_query = quote_plus(query)
    url = f"https://www.bing.com/search?q={safe_query}&mkt=en-us"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find("main").text
        print(results[:200])
        
        return results if results else "No results found."
        
    except Exception as e:
        return f"Search failed: {str(e)}"
    
def build_system():
    # Register tool
    tools = {"search_web": search_web}
    LargeLanguageModel.register_available_tools(tools)
    
    # State definition
    class AgentState(TypedDict):
        messages: List[Any]
        claim: str
        search_results: str
        prediction: str
    
    graph = StateGraph(AgentState)
    
    # Search node
    def search_node(state):
        search = tools["search_web"].invoke(state["claim"])
        
        new_state = state.copy()
        new_state["search_results"] = search
        return new_state
    
    # Evaluation node
    def agent_node(state):
        llm = LargeLanguageModel(temperature=0.2)
        
        system_prompt = """
            You will evaluate factual claims.
            
            For your analysis, please classify the given claim into one of these categories:
            - SUPPORTS: The claim is supported by factual evidence
            - REFUTES: The claim contradicts factual evidence
            - NOT ENOUGH INFO: There is insufficient evidence to determine if the claim is supported or refuted
            
            You get information from a web search. However, this web search alone may not be sufficient to properly evaluate the claim.
            
            Write your final prediction in the last line using exactly one of these three labels: SUPPORTS, REFUTES, or NOT ENOUGH INFO.    
        """

        search_content = "\nWebsearch results: " + state['search_results']

        full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["claim"] + search_content)]
        response = llm.invoke(full_messages)
        response_text = response.content
        print(response_text[-50:])
        
        # Try to extract the prediction from the response
        labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        final_answer = "NOT ENOUGH INFO" 
        
        last_line = response_text[-20:]
        
        for label in labels:
            if label in last_line:
                final_answer = label
                break
        
        # If not found in the last line, search the entire response
        if final_answer == "NOT ENOUGH INFO" and "NOT ENOUGH INFO" not in last_line:
            final_answer = "NO ANSWER"
        
        new_state = state.copy()
        new_state["prediction"] = final_answer
        return new_state
    
    # Add nodes
    graph.add_node("SearchNode", search_node)
    graph.add_node("FEVERAgent", agent_node)
    
    # Connect
    graph.add_edge("SearchNode", "FEVERAgent")
    
    # Entry/exit
    graph.set_entry_point("SearchNode")
    graph.set_finish_point("FEVERAgent")
    
    return graph.compile(), tools