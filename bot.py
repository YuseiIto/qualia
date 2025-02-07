"""
This code originally based on https://github.com/nishio/scrapbox_chatgpt_connector by @nishio released under MIT license.

Copyright (c) 2023 NISHIO

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import discord
from discord.ext import commands
import openai
import dotenv
import boto3  # AWS SDK for interacting with S3
import logging
from make_index import VectorStore, get_size

# Load environment variables (for local testing, not needed in Lambda)
dotenv.load_dotenv()

# --- Constants and Configurations ---
MAX_PROMPT_SIZE = 4096
RETURN_SIZE = 500
INDEX_FILE_S3_KEY = "yuseiito-private.pickle"  # Key in your S3 bucket

MINIO_ENDPOINT_URL = os.environ.get("MINIO_ENDPOINT_URL")
MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")


logger = logging.getLogger(__name__)

# --- Helper Functions ---


def acquire_index_from_s3(bucket_name, key):
    """Loads the pickled index from S3."""
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )
    try:
        logger.info(f"aquiring index from S3: {key}")
        response = s3.get_object(Bucket=bucket_name, Key=key)
        body = response["Body"].read()
        with open(key, "wb") as f:
            f.write(body)
        logger.info(f"Index loaded from S3: {key}")
    except Exception as e:
        logger.error(f"Error loading index from S3: {e}")
        return None


def ask(input_str, openai_client):
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
        raise RuntimeError("too large input!")
    rest -= input_size

    vs = VectorStore(INDEX_FILE_S3_KEY)
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

    logger.debug("\nTHINKING...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=RETURN_SIZE,
            temperature=0.0,
        )

        content = response.choices[0].message.content
        logger.debug("\nANSWER:")
        logger.debug(f">>>> {input_str}")
        logger.debug(">", content)
        return content

    except Exception as e:
        logger.error(f"Error during OpenAI API call: {e}")
        return "An error occurred with the AI assistant."


# --- Discord Bot ---

intents = discord.Intents.default()
intents.message_content = True  # Enable reading message content

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")


@bot.command(name="ask")
async def ask_command(
    ctx, *, question
):  # The * makes the bot take the rest of the message as the 'question' arg
    """Asks the AI assistant a question."""
    await ctx.send("Thinking...")  # Send an initial response

    client = openai.OpenAI()  # Create an OpenAI Client to use for this command

    try:
        answer = ask(question, client)
        await ctx.send(answer)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        await ctx.send("An error occurred while processing the question.")


if not MINIO_ENDPOINT_URL:
    logger.error("`MINIO_ENDPOINT_URL` not set.")

if not MINIO_BUCKET_NAME:
    logger.error("`MINIO_BUCKET_NAME` not set.")

if not MINIO_ACCESS_KEY:
    logger.error("`MINIO_ACCESS_KEY` not set.")

if not MINIO_SECRET_KEY:
    logger.error("`MINIO_SECRET_KEY` not set.")

if not OPENAI_API_KEY:
    logger.error("`OPENAI_API_KEY` not set.")

try:
    with open(INDEX_FILE_S3_KEY,"rb") as f:
        logger.info(f"Index file already exists at {INDEX_FILE_S3_KEY}")
except FileNotFoundError:
    logger.warning(f"Index file {INDEX_FILE_S3_KEY} not found.")
    acquire_index_from_s3(MINIO_BUCKET_NAME, INDEX_FILE_S3_KEY)
except Exception as e:
    logger.error(f"Error loading index file: {e}")

if DISCORD_BOT_TOKEN:
    bot.run(DISCORD_BOT_TOKEN)
else:
    logger.error("Discord bot token not found in environment variables.")
