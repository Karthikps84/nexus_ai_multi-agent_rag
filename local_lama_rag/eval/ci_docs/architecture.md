# Nexus AI architecture

Nexus AI answers questions about a user's documents with a team of agents orchestrated by LangGraph.
The LangGraph pipeline has four nodes that run in order: analyze, plan, execute, and finalize.

The analyze node uses the Router agent to pick a route such as question answering, summarizing,
comparing documents, extracting tables, code help, or image tasks. The plan node uses the Planner
agent to split a compound question into steps. The execute node runs the specialist agents, and the
finalize node verifies the answer and stores the conversation turn.

The Verifier agent checks every answer against the retrieved sources and produces a confidence score.
A Reflection agent re-checks answers whose confidence is low. The Memory agent keeps the
conversation history so that follow-up questions like "explain more about it" work.

For images, a real YOLO model detects objects and draws boxes, while a vision-language model
describes or classifies the picture. All language, vision and embedding models run on NVIDIA NIM.
