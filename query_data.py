import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

# Prompt template for information retrieval without hiding sensitive information.

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context. If the question cannot be properly answered based on the context, say you dont know.: {question}
"""

# Prompt template for information retrieval hiding sensitive information.

REDACT_TEMPLATE = """
Change the following text to hide sensitive information:
{text}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    # Generate the prompt for RAG.
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt).content

    # Hide sensitive information.
    redact_template = ChatPromptTemplate.from_template(REDACT_TEMPLATE)
    redact_prompt = redact_template.format(text=response_text)
    hidden_response_text = model.invoke(redact_prompt).content

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {hidden_response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
