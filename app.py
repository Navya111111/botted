import streamlit as st
import duckdb
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

st.set_page_config(page_title="SQL Chatbot", layout="wide")
st.title("🤖 CSV SQL Chatbot (DuckDB + Groq)")

# --------------------------------------------------
# Initialize LLM
# --------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview of Dataset")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Create DuckDB table
    # --------------------------------------------------
    con = duckdb.connect()
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM df")

    columns = list(df.columns)
    st.success(f"✅ Table created with columns: {columns}")

    # --------------------------------------------------
    # SQL GENERATION PROMPT
    # --------------------------------------------------
    sql_prompt = PromptTemplate(
        input_variables=["question", "columns"],
        template="""
You are a senior SQL expert.

Table: data
Columns: {columns}

Rules:
- Output ONLY SQL
- Use correct aggregate functions
- Use GROUP BY properly
- DuckDB compatible SQL only
- No explanations

Question:
{question}
"""
    )

    # --------------------------------------------------
    # SQL FIX PROMPT
    # --------------------------------------------------
    fix_prompt = PromptTemplate(
        input_variables=["sql", "error", "columns"],
        template="""
The following SQL caused an error:

SQL:
{sql}

Error:
{error}

Table columns:
{columns}

Fix the SQL so that it executes correctly.
Return ONLY corrected SQL.
"""
    )

    # --------------------------------------------------
    # Generate SQL
    # --------------------------------------------------
    def generate_sql(question):
        return llm.invoke(
            sql_prompt.format(
                question=question,
                columns=", ".join(columns)
            )
        ).content.strip().replace("", "")

    # --------------------------------------------------
    # Execute SQL with auto-fix
    # --------------------------------------------------
    def execute_with_retry(sql, retries=2):
        for attempt in range(retries + 1):
            try:
                return con.execute(sql).df(), sql
            except Exception as e:
                if attempt == retries:
                    raise e

                sql = llm.invoke(
                    fix_prompt.format(
                        sql=sql,
                        error=str(e),
                        columns=", ".join(columns)
                    )
                ).content.strip().replace("", "")

    # --------------------------------------------------
    # User Query Input
    # --------------------------------------------------
    st.subheader("💬 Ask a question about your data")
    user_question = st.text_input(
        "Example: total sales by region",
        placeholder="Type your question here..."
    )

    if st.button("🚀 Run Query") and user_question:
        try:
            sql = generate_sql(user_question)
            result, final_sql = execute_with_retry(sql)

            st.subheader("🧠 Generated SQL")
            st.code(final_sql, language="sql")

            st.subheader("📈 Query Result")
            st.dataframe(result)

        except Exception as e:
            st.error(f"❌ Failed to execute query: {e}")

else:
    st.info("👆 Upload a CSV file to start chatting with your data")