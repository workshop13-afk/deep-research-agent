import json
import os
import queue
import sys
import threading

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from agent import DeepResearchAgent
from prompts import DEFAULT_PROMPT, PROMPT_DESCRIPTIONS, SYSTEM_PROMPTS, VULNERABLE_PROMPTS

app = FastAPI(title="Deep Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DONE = object()


class ResearchRequest(BaseModel):
    query: str
    mode: str = DEFAULT_PROMPT


@app.get("/api/modes")
def get_modes():
    return {
        "modes": [
            {
                "name": name,
                "description": PROMPT_DESCRIPTIONS.get(name, ""),
                "vulnerable": name in VULNERABLE_PROMPTS,
            }
            for name in SYSTEM_PROMPTS
        ],
        "default": DEFAULT_PROMPT,
    }


@app.post("/api/research")
def research(req: ResearchRequest):

    q: queue.Queue = queue.Queue()

    def on_thinking(token: str) -> None:
        q.put(("thinking", {"token": token}))

    def on_action(kind: str, value: str) -> None:
        q.put(("action", {"kind": kind, "value": value}))

    def run() -> None:
        try:
            agent = DeepResearchAgent(system_prompt_name=req.mode)
            result = agent.research(req.query, on_action=on_action, on_thinking=on_thinking)
            q.put(("complete", dict(result)))
        except Exception as exc:
            q.put(("error", {"message": str(exc)}))
        finally:
            q.put(_DONE)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            if item is _DONE:
                break
            event, data = item
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
