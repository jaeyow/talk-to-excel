import os
import chainlit as cl
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
import logging
import sys
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logging.getLogger("httpx").setLevel(logging.ERROR)

def load_language_models():
    """
    Load the language models for the Retriever and Generator.
    """
    from llama_index.embeddings.bedrock import BedrockEmbedding
    from llama_index.llms.bedrock import Bedrock
    import os
    from dotenv import load_dotenv

    print(f"Loading env: {load_dotenv(verbose=True, dotenv_path=".env")}")
    
    embedding_model = "amazon.titan-embed-text-v2:0"
    print(f"Setting up remote Retriever model (embedding: {embedding_model})...")
    Settings.embed_model = BedrockEmbedding(
        model_name=embedding_model,
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20
            
    # Setup Generator model
    llm_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    print(f"Setting up remote Generator model (main LLM: {llm_model})...")
    Settings.llm = Bedrock(
        model=llm_model,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        context_window=8192,
        request_timeout=120,
    )
    
load_language_models()

@cl.on_chat_start
async def on_chat_start():
    """
    This function is called when the user starts a new chat session.
    """
    await cl.Message(
        content=
        """
        Welcome to Titanic Data Explorer! Ask any question about the Titanic passenger data. For example, you can ask:
        - What is the average age of passengers?
        - What is the average fare paid by the passengers?
        - What was the average fare paid by first-class passengers compared to third-class passengers?
        - Did paying a higher fare increase the chance of survival?
        - Plot the distribution of passengers with respect to sex.
        - Plot how many passengers were in each class.
        
        """
    ).send()
    
    try:
        file_path = os.path.join(os.path.dirname(__file__), "data", "titanic_train.xlsx")
        df = pd.read_excel(file_path)
        
        query_engine = PandasQueryEngine(
            df=df,
            verbose=True,
            synthesize_response=True,
            description="This dataframe contains passenger information from the Titanic, \
                including columns like 'PassengerId', 'Survived' (0=No, 1=Yes), 'Pclass' (ticket class 1-3), \
                'Name', 'Sex', 'age', 'SibSp' (siblings/spouses), 'Parch' (parents/children), 'Ticket' number, \
                'Fare', 'Cabin', and 'Embarked' (port of embarkation: C=Cherbourg, Q=Queenstown, S=Southampton).",
        )
        
        cl.user_session.set("query_engine", query_engine)
        
        message = cl.Message(content="Loaded Titanic dataset successfully. Here's a preview:")
        await message.send()
        
        element = cl.Dataframe(data=df.head(20), name="Titanic Data Preview", size="large")
        await cl.Message(content="", elements=[element]).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Error loading dataset: {str(e)}"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    This function is called whenever a new message is received from the user.
    """
    import matplotlib.pyplot as plt
    # Get the query engine from user session
    query_engine = cl.user_session.get("query_engine")
    
    if not query_engine:
        await cl.Message(content="The query engine has not been properly initialized.").send()
        return
    
    thinking_msg = cl.Message(content="Analyzing the data...")
    await thinking_msg.send()
    
    try:
        fig, _ = plt.subplots()
        
        # Use the language model to check if the message is asking for a plot or visualization
        prediction = Settings.llm.predict(
            prompt=PromptTemplate(f"Does the following message explicitly ask for a graph, plot or visualization? \
                {message.content}. Answer Yes or No only."),
            max_tokens=10,
        )
        print(f"Plot or graph needed: {prediction}")
        
        query_engine._synthesize_response = True
        response = query_engine.query(message.content)
        
        if "yes" in prediction.lower():
            print("Generating plot...")
            elements = [cl.Pyplot(
                name="plot",
                figure=fig,
                display="inline",
                size="large")]

            thinking_msg.content = f"{response}"
            thinking_msg.elements = elements
            await thinking_msg.update()
        else:
            print("No plot needed")
            
            thinking_msg.content = f"{response}"
            await thinking_msg.update()
        
    except Exception as e:
        thinking_msg.content = f"Error processing your question: {str(e)}"
        await thinking_msg.update()