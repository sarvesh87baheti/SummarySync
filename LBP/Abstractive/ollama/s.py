from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3")
response = llm.invoke("Summarize this: Hello world!")
print(response)
