"""
Pipeline agents for each stage of video reel generation.

Each agent subclasses BaseReelAgent and provides:
- System prompt and user prompt template (from src/prompts/templates.py)
- LLM call via the router (local-first, with cloud fallback)
- Structured JSON output parsing
- Confidence scoring
- Feedback injection for iterative improvement

Import agents directly from their modules:
    from src.agents.screenplay_agent import ScreenplayAgent
"""
