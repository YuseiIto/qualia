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
from time import sleep

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


def ask(input_str,history, openai_client):
    """Asks the OpenAI model a question, using the index for context."""

    # TODO: Prompt engineering for better results
    SYSTEM_PROMPT = """
    あなたは、yuseiitoのAIアシスタントで、クオリアといいます。
    あなたは、以下にyuseiitoのWikiに書かれた内容を受け取ります。
    入力に対して、Wikiの中から関連する情報を取得したり、それらと矛盾しないように回答を生成してください。

    ## Wiki
    {text}
    """.strip()

    PROMPT_SIZE = get_size(SYSTEM_PROMPT)
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
    prompt = SYSTEM_PROMPT.format(text=text)

    logger.debug("\nTHINKING...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system","content": prompt},*history,{"role": "user", "content": input_str}],
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


async def fetch_reply_history(ctx,message):
    ref = message.reference
    history = []
    while ref is not None:
        parent = await ctx.fetch_message(ref.message_id)
        history.append(parent)
        ref = parent.reference
    history.reverse()
    return [{"role": "assistant" if m.author.id==bot.user.id else "user", "content": m.content} for m in history]

@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")

@bot.event
async def on_message(message):
    # Check if I'm mentioned in the message
    if bot.user.mentioned_in(message):
        content = message.content
        history = await fetch_reply_history(message.channel,message)
        try:
            async with message.channel.typing():
                client = openai.OpenAI()  # Create an OpenAI Client to use for this command
                answer = ask(content,history,client)
                await message.reply(answer)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            await message.channel.send("An error occurred while processing the question.")


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
