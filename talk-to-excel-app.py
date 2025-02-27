import os
import chainlit as cl
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
import logging
import sys

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.ERROR)

def load_language_models():
    from llama_index.embeddings.bedrock import BedrockEmbedding
    from llama_index.llms.bedrock import Bedrock
    from llama_index.core import Settings
    import os
    
    from dotenv import load_dotenv

    load_dotenv(verbose=True, dotenv_path=".env")
    
    embedding_model = "amazon.titan-embed-text-v2:0"
    print(f"Setting up remote Retriever model (embedding: {embedding_model})...")
    Settings.embed_model = BedrockEmbedding(
        model_name=embedding_model,
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20
            
    # Setup Generator model
    # llm_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    llm_model = "anthropic.claude-3-haiku-20240307-v1:0"
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

# Load the Titanic dataset
@cl.on_chat_start
async def on_chat_start():
    # Send an initial message
    await cl.Message(
        content="Welcome to Titanic Data Explorer! Ask any question about the Titanic passenger data."
    ).send()
    
    # Load the dataset
    try:
        file_path = os.path.join(os.path.dirname(__file__), "data", "titanic_train.xlsx")
        df = pd.read_excel(file_path)
        
        # Initialize the query engine
        query_engine = PandasQueryEngine(
            df=df,
            verbose=True,
            description="This dataframe contains passenger information from the Titanic, including columns like 'PassengerId', 'Survived' (0=No, 1=Yes), 'Pclass' (ticket class 1-3), 'Name', 'Sex', 'Age', 'SibSp' (siblings/spouses), 'Parch' (parents/children), 'Ticket' number, 'Fare', 'Cabin', and 'Embarked' (port of embarkation: C=Cherbourg, Q=Queenstown, S=Southampton).",
            instruction_str="Analyze the dataframe to answer the question. Always respond in complete sentences with explanations when appropriate. Round numerical answers to 2 decimal places when needed."
        )
        
        # Store the query engine in user session
        cl.user_session.set("query_engine", query_engine)
        
        # Display dataframe preview
        message = cl.Message(content="Loaded Titanic dataset successfully. Here's a preview:")
        await message.send()
        
        # Create and send dataframe as element
        element = cl.Dataframe(data=df.head(5), name="Titanic Data Preview")
        await element.send()
        
    except Exception as e:
        await cl.Message(
            content=f"Error loading dataset: {str(e)}"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Get the query engine from user session
    query_engine = cl.user_session.get("query_engine")
    
    if not query_engine:
        await cl.Message(content="The query engine has not been properly initialized.").send()
        return
    
    # Show thinking/loading message
    thinking_msg = cl.Message(content="Analyzing the data...", disable_feedback=True)
    await thinking_msg.send()
    
    try:
        # Process the query
        response = query_engine.query(message.content)
        
        # Update the message with the response
        await thinking_msg.update(content=f"{response}")
        
        # For some questions, we can enhance with visualizations
        if any(keyword in message.content.lower() for keyword in 
              ['distribution', 'compare', 'correlation', 'relation', 'survival', 'chart', 'graph', 'plot']):
            
            # Get dataframe from session
            df = query_engine._df
            
            # Simple visualization based on query content
            if any(word in message.content.lower() for word in ['age', 'distribution']):
                import matplotlib.pyplot as plt
                import io
                import base64
                
                plt.figure(figsize=(10, 6))
                plt.hist(df['Age'].dropna(), bins=20, alpha=0.7)
                plt.title('Age Distribution of Titanic Passengers')
                plt.xlabel('Age')
                plt.ylabel('Count')
                plt.grid(alpha=0.3)
                
                # Convert plot to base64 image
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                # Send the plot as an image element
                await cl.Image(
                    name="Age Distribution",
                    display="inline",
                    content=buffer.getvalue()
                ).send()
                
    except Exception as e:
        await thinking_msg.update(content=f"Error processing your question: {str(e)}")