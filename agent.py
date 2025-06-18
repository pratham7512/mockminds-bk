import logging
from dotenv import load_dotenv
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.plugins import elevenlabs, openai, silero, deepgram, google, noise_cancellation
from livekit.plugins.elevenlabs import Voice
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pymongo import MongoClient
from datetime import datetime

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def save_conversation(ctx: JobContext):
    async def on_shutdown():
        try:
            client = MongoClient("mongodb+srv://admin:desai1969@cluster0.mqucw.mongodb.net/db1?retryWrites=true&w=majority")
            db = client.db1
            
            conversation_data = {
                "session_id": ctx.job.id,
                "room_name": ctx.room.name,
                "messages": ctx.chat_ctx.messages,
                "timestamp": datetime.now(),
                "interview_type": "cybersecurity"
            }
            
            db.conversations.insert_one(conversation_data)
            client.close()
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
    
    ctx.add_shutdown_callback(on_shutdown)

class InterviewAgent(Agent):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions=(
                """start with introduction and inform user about interview, You are an AI conducting an interview of javascript Developer. Your role is to manage the interview effectively by:
      - Understanding the candidate's intent, especially when using voice recognition which may introduce errors.
      - Asking follow-up questions to clarify any doubts without leading the candidate.
      - Focusing on collecting and questioning about the javascript core concepts.
      - Ensuring the interview flows smoothly, avoiding repetitions or direct hints, and steering clear of unproductive tangents.

      - Your visible messages will be read out loud to the candidate.
      - Use mostly plain text, avoid markdown and complex formatting, unless necessary avoid code and formulas in the visible messages.
      - Use '\n\n' to split your message in short logical parts, so it will be easier to read for the candidate.
      - Be very concise in your responses. Allow the candidate to lead the discussion, ensuring they speak more than you do.
      - Never repeat, rephrase, or summarize candidate responses. Never provide feedback during the interview.
      - Never repeat your questions or ask the same question in a different way if the candidate already answered it.
      - Never give away the solution or any part of it. Never give direct hints or part of the correct answer.
      - If the candidate asks appropriate questions about data not mentioned in the problem statement (e.g., scale of the service, time/latency requirements, nature of the problem, etc.), you can make reasonable assumptions and provide this information.
      - Actively listen and adapt your questions based on the candidate's responses. Avoid repeating or summarizing the candidate's responses."""
            ),
            chat_ctx=chat_ctx
        )

async def entrypoint(ctx: JobContext):
    await save_conversation(ctx)
    await ctx.connect()

    # You can use your custom voice if needed, or just use the default TTS
    voice = Voice(
        id="vO7hjeAjmsdlGgUdvPpe",  # Replace with your voice ID
        name="Amrut Deshmukh - Booklet Guy",
        category='premade'
    )

    session = AgentSession(
        stt=openai.STT.with_groq(model="whisper-large-v3-turbo"),
        llm=openai.LLM.with_groq(model="llama-3.3-70b-versatile"),
        tts=elevenlabs.TTS(voice=voice, model="eleven_flash_v2_5"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(instructions="Hey, can we start with the interview")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
