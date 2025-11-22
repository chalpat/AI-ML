import re
import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Load the CSV for direct numeric filtering
df = pd.read_csv("products.csv")

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about electronic products.

You are given some relevant products from a database.
Each product has fields like Name, Category, Price, etc.

Use ONLY the given products to answer the question.
If the user asks for something outside these products, say you don't have that information.

Here are some relevant products:
{products}

Here is the question to answer:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def is_price_query(question: str):
    """
    Detects simple queries like:
    - 'give me all products less than 200'
    - 'show items under 500'
    - 'products greater than 300'
    Returns a tuple (mode, value) where mode is 'lt' or 'gt', or (None, None).
    """
    q = question.lower()

    # Try to detect "less than / under / below"
    m_lt = re.search(r"(less than|under|below)\s+(\d+)", q)
    if m_lt:
        return "lt", int(m_lt.group(2))

    # Try to detect "greater than / above / over"
    m_gt = re.search(r"(greater than|above|over|more than)\s+(\d+)", q)
    if m_gt:
        return "gt", int(m_gt.group(2))

    return None, None


def handle_price_query(mode: str, value: int):
    """
    Filter the dataframe based on mode ('lt' or 'gt') and value.
    Prints the results in a clean format.
    """
    if mode == "lt":
        filtered = df[df["Price"] < value]
        condition_text = f"Price < {value}"
    else:
        filtered = df[df["Price"] > value]
        condition_text = f"Price > {value}"

    if filtered.empty:
        print(f"No products found with {condition_text}.")
        return

    print(f"Products with {condition_text}:\n")
    for _, row in filtered.iterrows():
        print(
            f"- {row['Name']} | Category: {row['Category']} | "
            f"Price: {row['Price']} {row['Currency']} | Stock: {row['Stock']}"
        )


while True:
    print("\n\n-----------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.strip().lower() == "q":
        break

    # 1️⃣ Check if it's a price filter question
    mode, value = is_price_query(question)
    if mode is not None:
        # Use direct numeric filtering – no LLM, no embeddings
        handle_price_query(mode, value)
        continue

    # 2️⃣ Otherwise, use retriever + LLM
    docs = retriever.invoke(question)

    # Convert documents into a readable string for the prompt
    products_text = "\n\n".join(
        [
            f"Content: {d.page_content}\nMetadata: {d.metadata}"
            for d in docs
        ]
    )

    result = chain.invoke({"products": products_text, "question": question})
    print(result)