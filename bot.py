import os
import discord
from discord.ext import commands
import openai
import dotenv
import boto3  # AWS SDK for interacting with S3
import pickle
import io
import json


# Load environment variables (for local testing, not needed in Lambda)
dotenv.load_dotenv()

# --- Constants and Configurations ---
MAX_PROMPT_SIZE = 4096
RETURN_SIZE = 500
INDEX_FILE_S3_KEY = "yuseiito-private.pickle"  # Key in your S3 bucket
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")  # Must be set in Lambda config!
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Must be set in Lambda config!
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")  # Must be set in Lambda config!


# --- Helper Functions ---


def load_index_from_s3(bucket_name, key):
    """Loads the pickled index from S3."""
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        index_data = response["Body"].read()
        index = pickle.loads(index_data)
        return index
    except Exception as e:
        print(f"Error loading index from S3: {e}")
        return None



def get_size(text):
    return len(text.encode("utf-8"))

class VectorStore: #Simplified VectorStore. Replace with your actual implementation if needed.
    def __init__(self, index_data):
        self.index_data = index_data

    def get_sorted(self, input_str):
        #Replace with actual similarity search based on your index structure
        #This example just returns some data from the loaded index

        #Simulate sorted results based on a simple string match
        results = []
        for title, body in self.index_data.items():
            if input_str.lower() in title.lower() or input_str.lower() in body.lower():
                results.append((1.0, body, title)) #Simulated similarity score of 1.0
            else:
                results.append((0.5, body, title)) #Simulated lower similarity score


        return results

def ask(input_str, index, openai_client):
    """Asks the OpenAI model a question, using the index for context."""

    # TODO: Prompt engineering for better results
    PROMPT = """
    あなたは、yuseiitoのAIアシスタントで、クオリアといいます。
    あなたは、以下にyuseiitoのWikiに書かれた内容の例示と入力を受け取ります。
    入力に対して、Wikiの中から関連する情報を取得したり、それらと矛盾しないように回答を生成してください。

    ## Wiki
    {text}
    ## Input
    {input}
    """.strip()


    PROMPT_SIZE = get_size(PROMPT)
    rest = MAX_PROMPT_SIZE - RETURN_SIZE - PROMPT_SIZE
    input_size = get_size(input_str)
    if rest < input_size:
        return "Input is too long!"
    rest -= input_size

    vs = VectorStore(index)
    samples = vs.get_sorted(input_str)

    to_use = []
    used_title = []
    for _sim, body, title in samples:
        if title in used_title:
            continue
        size = get_size(body)
        if rest < size:
            break
        to_use.append(body)
        used_title.append(title)
        rest -= size

    text = "\n\n".join(to_use)
    prompt = PROMPT.format(input=input_str, text=text)

    print("\nTHINKING...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=RETURN_SIZE,
            temperature=0.0,
        )

        content = response.choices[0].message.content
        print("\nANSWER:")
        print(f">>>> {input_str}")
        print(">", content)
        return content

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return "An error occurred with the AI assistant."


# --- Discord Bot ---

intents = discord.Intents.default()
intents.message_content = True  # Enable reading message content

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} ({bot.user.id})")

@bot.command(name="ask")
async def ask_command(ctx, *, question): #The * makes the bot take the rest of the message as the 'question' arg
    """Asks the AI assistant a question."""
    await ctx.send("Thinking...")  # Send an initial response

    # Load index within the command to ensure it is fresh
    index_data = load_index_from_s3(S3_BUCKET_NAME, INDEX_FILE_S3_KEY)
    if not index_data:
        await ctx.send("Index data could not be loaded from S3.")
        return

    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI() #Create an OpenAI Client to use for this command

    try:
        answer = ask(question, index_data, client)
        await ctx.send(answer)
    except Exception as e:
        print(f"Error processing question: {e}")
        await ctx.send("An error occurred while processing the question.")


if DISCORD_BOT_TOKEN:
    bot.run(DISCORD_BOT_TOKEN)
else:
    print("Discord bot token not found in environment variables.")
