# def handle_tech_question(question: str, llm) -> str:
#     prompt = f"You are a tech assistant. Answer the following technical question:\n{question}"
#     return llm.invoke(prompt)
# Agents/tech_agent.py

from vectorstore import retrieve_context

def handle_tech_question(question: str, llm) -> str:
    context = retrieve_context(question)

    prompt = f"""
    You are a technical career assistant.
    Answer the question ONLY using the context below.
    If the answer is not present, say you don't have enough information.

    Context:
    {context}

    Question:
    {question}
    """

    return llm.invoke(prompt)
