from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    elevenlabs,
    silero,
    groq,
    noise_cancellation
)

load_dotenv(dotenv_path=".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""start with introduction and inform user about intrview, You are an AI conducting an interview of javascript Developer. Your role is to manage the interview effectively by:
      - Understanding the candidate’s intent, especially when using voice recognition which may introduce errors.
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
      - Actively listen and adapt your questions based on the candidate's responses. Avoid repeating or summarizing the candidate's responses.""")

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=groq.STT(model="whisper-large-v3-turbo"),
        llm=groq.LLM(model="meta-llama/llama-4-scout-17b-16e-instruct"),
        tts=elevenlabs.TTS(voice_id="vO7hjeAjmsdlGgUdvPpe",model="eleven_flash_v2_5"),
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
