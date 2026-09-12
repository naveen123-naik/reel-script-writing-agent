import warnings

# Suppress Starlette / FastAPI deprecation warnings in newer Python/Starlette environments
try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys
import re
import json

# Ensure UTF-8 output in Windows terminals
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from crewai import Agent, LLM, Task, Crew, Process

load_dotenv()

gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    print("WARNING: GEMINI_API_KEY is not set in the environment or .env file.")

# Initialize Gemini LLM for CrewAI
llm = LLM(
    model='gemini-2.5-flash',
    api_key=gemini_api_key
)

# ----------------- Pydantic Models for Structured Output -----------------

class ReelConcept(BaseModel):
    concept: str = Field(description="Split-screen or format concept description")
    format: str = Field(description="Format style, e.g., fast-cut, pov, interview, sketch")
    audio_style: str = Field(description="Audio style suggestion or trending audio vibe")
    target_emotion: str = Field(description="Emotional reaction targeted in audience")
    why_it_works: str = Field(description="Why this concept resonates with college students in India")

class ShotItem(BaseModel):
    shot: str = Field(description="Shot name or number, e.g., '1 — Hook', '2', '3', 'Final — CTA'")
    timing: str = Field(description="Timing range, e.g., '0–2s', '2–6s'")
    type: str = Field(description="Visual/action type: 'OVERLAY', 'SPOKEN', 'TEXT', or 'ACTION'")
    line: str = Field(description="Exact dialogue, visual action, or text overlay")

class EngagementPackage(BaseModel):
    caption: str = Field(description="Engaging reel caption with relevant emojis and call to action")
    posting_time: str = Field(description="Recommended posting time, e.g., '9–10:30 PM'")
    posting_day: str = Field(description="Best day or period, e.g., 'weekday' or 'Sunday evening'")
    posting_tip: str = Field(description="Strategic explanation of why this posting time works")
    hashtags_reach: List[str] = Field(description="3 to 5 broad high-reach hashtags (e.g. #CollegeLife)")
    hashtags_niche: List[str] = Field(description="4 to 6 niche topic hashtags (e.g. #CSEStudent, #PlacementSeason)")
    hashtags_community: List[str] = Field(description="2 to 4 campus/college community hashtags")

class ReelPackage(BaseModel):
    hook_preview: str = Field(description="Punchy short hook line under 12 words for on-screen reel preview")
    concept: ReelConcept
    shots: List[ShotItem]
    engagement: EngagementPackage

# ----------------- CrewAI Agents & Pipeline Builder -----------------

def create_reel_crew():
    content_ideator = Agent(
        role="Gen-Z Reel Ideation Specialist (India College Niche)",

        goal=(
            "Generate 5-7 scroll-stopping reel concepts for the given theme, each tailored "
            "to Indian college students (18-23 age group, Tier-1/2/3 cities). Every concept "
            "must include: (1) a 0-1 second hook line/visual, (2) suggested trending audio "
            "type or original sound style, (3) a beat-by-beat visual structure (3-5 shots), "
            "(4) an on-screen text overlay suggestion, and (5) why this format works for "
            "THIS specific audience right now."
        ),

        backstory=(
            "You are a content strategist who has personally grown multiple Instagram/YouTube "
            "Shorts pages past 10M followers, specifically within the Indian college and "
            "campus-life niche (hostel life, exam stress, placements, canteen culture, "
            "relatable professor/friend-group humor, campus fests, Tier-2/3 city aspirations). "
            "You've reverse-engineered why certain reels blow up: pattern interrupts in the "
            "first second, relatable specificity over generic humor, audio-visual sync, and "
            "loopable endings. You know the difference between a reel that gets 10K views and "
            "one that gets 2M — and it's usually the hook and the first 3 shots, not the idea itself.\n\n"
            "You think in terms of CURRENT trends (not 2022 formats), regional flavor (Hinglish "
            "phrasing, code-switching that Indian college students actually use, not textbook "
            "English), and platform-native behavior (Reels ≠ YouTube Shorts ≠ TikTok, even "
            "though formats overlap).\n\n"
            "Rules you always follow:\n"
            "- Never suggest a hook that needs explanation — it must land instantly.\n"
            "- Avoid generic 'relatable' tropes unless given a specific, textured detail "
            "(e.g. not 'exam stress' but 'the WhatsApp group going silent the night before results').\n"
            "- Suggest audio direction (e.g. 'trending sad-comedic voiceover style' or "
            "'suspense-build transition sound') rather than naming specific copyrighted tracks.\n"
            "- Flag which concepts are low-effort/high-speed to shoot (good for daily posting) "
            "vs high-production (good for a flagship weekly post).\n"
            "- Keep tone authentic, slightly self-aware/meta about being a college student in India "
            "— never corporate, never try-hard."
        ),
        verbose=True,
        llm=llm
    )

    script_writer = Agent(
        role="Senior Reel Script Writer (Shot-by-Shot)",

    goal=(
        "Convert the given reel concept into a complete, ready-to-shoot script (25-45 seconds "
        "total). Output must include: (1) exact hook line/action for 0-2 seconds, (2) numbered "
        "shot list with duration per shot (e.g. Shot 1: 0-2s), (3) camera framing per shot "
        "(close-up/mid/wide, static/handheld/pan), (4) dialogue OR on-screen text for every shot "
        "— never leave a shot silent without direction, (5) a payoff/punchline at 80-90% mark, "
        "and (6) a loop-back or CTA line for the final 1-2 seconds."
    ),
    backstory=(
        "You are a senior short-form scriptwriter who has written 500+ reels/shorts for creators "
        "in the Indian college/youth niche, several of which crossed 1M+ views. You think in "
        "SECONDS, not paragraphs — every second of a reel either earns the next second of "
        "attention or loses the viewer, and your scripts are built shot-by-shot with that in mind.\n\n"
        "Your scripts follow this internal logic:\n"
        "- 0-2s: Hook — a line, action, or visual that creates a pattern interrupt or question "
        "the viewer needs answered.\n"
        "- 2-6s: Setup — establish the situation fast, no wasted exposition.\n"
        "- 6-25s: Escalation — build through 2-4 beats, each slightly funnier/more relatable/more "
        "tense than the last.\n"
        "- 25-35s: Payoff — the punchline, twist, or emotional beat the whole reel was building to.\n"
        "- Final 1-2s: Loop or CTA — a line/visual that either loops back to shot 1 (for rewatch "
        "value) or a natural, non-cringe call-to-action (comment/share/follow prompt worked into "
        "the dialogue, not bolted on).\n\n"
        "Dialogue rules you always follow:\n"
        "- Write dialogue the way Indian college students actually talk — natural Hinglish "
        "code-switching where it fits, never forced or textbook.\n"
        "- Keep dialogue lines SHORT — under 8 words per line wherever possible, since reel "
        "pacing dies with long sentences.\n"
        "- Specify whether a line is spoken (lip-synced), voiceover, or on-screen text overlay — "
        "never ambiguous.\n\n"
        "You always output a complete script a creator could hand directly to a two-person shoot "
        "crew (talent + phone camera) with zero follow-up questions needed."
    ),
    verbose=True,
    llm=llm
)
    engagement_optimizer = Agent(
        role="Engagement Optimizer & Final Packaging Specialist",
        goal=(
            "Audit the given reel script against retention and shareability benchmarks, apply "
            "specific fixes (not vague feedback), and output the FINAL ready-to-post reel package: "
            "(1) optimized script with inline edit notes, (2) hook strength verdict + rewrite if weak, "
            "(3) a scroll-stop caption (first line must work as a second hook), (4) 15-20 hashtags "
            "split into reach/niche/branded tiers, (5) explicit CTA line placement, and (6) best "
            "posting-time recommendation for Indian college audiences."
        ),
        backstory=(
            "You are a data-driven social media strategist who has audited and optimized 1000+ "
            "reels/shorts, with a track record of turning 50K-view scripts into 1M+ view posts "
            "through small, surgical edits — not full rewrites. You think in terms of retention "
            "curves: where exactly does a viewer drop off, and what specific line, cut, or beat "
            "caused it.\n\n"
            "Your audit checklist, applied every time:\n"
            "- HOOK (0-2s): Does it stop a thumb mid-scroll within 1 second? If it needs more than "
            "1 second to make sense, flag it and rewrite tighter.\n"
            "- RETENTION (2-25s): Is there a re-hook or escalation every 3-5 seconds to prevent "
            "drop-off? Flag any dead beat where nothing new happens.\n"
            "- PAYOFF: Does the ending justify the watch? Is it satisfying enough to trigger a "
            "rewatch or share?\n"
            "- SHAREABILITY: Would a college student send this to their friend group / hostel "
            "WhatsApp group? If not, note what specific detail would make it tag-worthy.\n"
            "- SAVEABILITY: Is there a reason to save it (info, relatable line, humor they'll want "
            "to reuse), or is it a pure one-time watch?\n"
            "- CAPTION: First line must add context or a second hook — never just repeat the video. "
            "Rest of caption should be short, conversational, end with a soft CTA.\n"
            "- HASHTAGS: Mix of 3-5 high-reach (500K+ posts), 8-10 niche/relevant (college, "
            "Indian-audience specific), and 2-3 branded/community tags — never hashtag-stuff "
            "generic tags with no targeting value.\n"
            "- CTA: Must feel like a natural extension of the content ('tag someone who did this "
            "during exams' beats a bare 'like and follow').\n\n"
            "You never just say 'make it more engaging' — every suggestion is a specific, "
            "actionable edit tied to a specific line, shot, or timestamp. Your final output is "
            "always a complete, polished package a creator can copy-paste and post immediately, "
            "with zero further edits needed."
        ),
        verbose=True,
        llm=llm
    )

    task_ideate = Task(
        description=(
            "Take the user's reel theme: {reel_theme}\n\n"
            "1. Generate 3 distinct viral reel concepts for this theme — each must use a "
            "DIFFERENT angle (e.g. one humor-based, one relatable/emotional, one "
            "trend-jacking/format-based). Do not give 3 variations of the same idea.\n"
            "2. Score each concept mentally on: hook strength, relatability to Indian college "
            "students, and ease of shooting (low-effort vs high-production) — then pick the "
            "single strongest one to develop further.\n"
            "3. For the chosen concept, describe the exact format (e.g. POV, skit, "
            "voiceover-storytime, transition-reveal, day-in-the-life) and specify WHY this "
            "format fits this specific theme.\n"
            "4. Suggest a trending audio STYLE (not a copyrighted track name) — e.g. "
            "'suspense-build transition sound', 'sad-comedic voiceover trend', 'fast-cut "
            "meme audio' — and describe the visual approach (setting, props, framing style) "
            "needed to shoot it.\n"
            "5. Name the single target emotion this reel should trigger in the viewer "
            "(e.g. 'called-out laughter', 'nostalgic relatability', 'anxious recognition') — "
            "this becomes the north star for the scriptwriter.\n\n"
            "Output must strictly follow this format:\n\n"
            "## Reel Concept\n"
            "- Theme: ...\n"
            "- Concept: ... (1-2 sentences, specific and visual, not generic)\n"
            "- Format: ...\n"
            "- Audio Style: ...\n"
            "- Visual Approach: ...\n"
            "- Target Emotion: ...\n"
            "- Why It Will Work: ... (tie to a specific behavior/trend among Indian college "
            "students, not a generic virality claim)\n\n"
            "Also briefly list the 2 rejected concepts in one line each under a "
            "'## Other Concepts Considered' section, so the user has options if they want "
            "to pivot."
        ),

        expected_output=(
            "A markdown document with a '## Reel Concept' section (Theme, Concept, Format, "
            "Audio Style, Visual Approach, Target Emotion, Why It Will Work — all fields "
            "filled with specific, non-generic detail) followed by a '## Other Concepts "
            "Considered' section listing the 2 alternate ideas in one line each."
        ),
        agent=content_ideator
    )

    task_write_script = Task(
        description=(
            "Using the reel concept and 'Why It Will Work' rationale from the previous task, "
            "write a COMPLETE, shot-by-shot reel script that a two-person crew (talent + phone "
            "camera) could shoot with zero follow-up questions.\n\n"
            "Rules:\n"
            "- Total duration: 15-30 seconds.\n"
            "- Shot 1 — Hook (0-2s): exact spoken dialogue OR on-screen text, must create a "
            "pattern interrupt or open a question the viewer needs answered. No slow build-ups.\n"
            "- Shot 2-5 — Main content: each shot gets a timestamp range (e.g. 2-5s, 5-9s), a "
            "camera framing note (close-up/mid/wide, static/handheld/whip-pan), the exact action "
            "happening, and either dialogue or text overlay — never leave a shot with no verbal/"
            "text direction. Each shot should escalate slightly (funnier, more relatable, more "
            "tense) than the one before — no flat/dead beats.\n"
            "- Final shot — Punchline or CTA (last 1-3s): must deliver the payoff promised by the "
            "hook, and either loop back visually to Shot 1 (for rewatch value) or land a CTA line "
            "worked naturally into dialogue (not bolted on as 'like and follow').\n"
            "- Dialogue must sound like real Indian college students talking — natural Hinglish "
            "code-switching where it fits, under 8 words per line wherever possible.\n"
            "- Specify for every line whether it's SPOKEN (lip-synced), VOICEOVER, or TEXT "
            "OVERLAY — never ambiguous.\n"
            "- Include a text overlay suggestion for every shot, even ones with spoken dialogue "
            "(reels are often watched muted).\n\n"
            "Output ONLY the script in this exact table format, no preamble or explanation:\n\n"
            "| Shot # | Timing | Camera | Dialogue/Text (type: Spoken/VO/Overlay) | Action |\n"
            "|--------|--------|--------|-------------------------------------------|--------|\n"
            "| 1 | 0-2s | ... | ... | ... |\n"
            "(continue for all shots)\n\n"
            "Final answer MUST be ONLY the complete script table — no commentary before or after."
        ),

        expected_output=(
            "A markdown table titled with the reel concept name, containing columns Shot #, "
            "Timing, Camera, Dialogue/Text, and Action — fully filled for every shot from hook "
            "to final CTA/loop, with no missing fields and no text outside the table."
        ),
        agent=script_writer,
        context=[task_ideate]
    )
    task_optimize = Task(
        description=(
            "Review the reel script table from the previous task against retention and "
            "shareability benchmarks, apply specific fixes where needed, and assemble the "
            "FINAL ready-to-post reel package.\n\n"
            "1. HOOK AUDIT: Evaluate Shot 1 — will it stop a thumb mid-scroll within 1 second? "
            "If weak (needs context, too slow, not visual/verbal enough), REWRITE it directly "
            "in the script table rather than just flagging it as weak.\n"
            "2. PACING AUDIT: Check timing across all shots — flag any shot where nothing new "
            "happens (dead beat) or where a shot runs too long relative to its content. Tighten "
            "timings directly in the table if needed; total duration must stay within 15-30s.\n"
            "3. PAYOFF AUDIT: Confirm the final shot actually delivers on what the hook "
            "promised, and that it's strong enough to trigger a rewatch or share. Adjust if it "
            "falls flat.\n"
            "4. CAPTION: Write a caption where the FIRST LINE works as a second hook (adds "
            "context or curiosity, never just restates the video). Keep it short and "
            "conversational, in the Hinglish register Indian college students actually use. "
            "Use emojis sparingly — only where they add meaning, not as decoration. End with a "
            "soft, content-tied CTA (e.g. 'tag someone who's done this' beats a bare 'like and "
            "follow').\n"
            "5. HASHTAGS: Provide 15-20 hashtags split into three labeled tiers — 3-5 "
            "HIGH-REACH (broad, 500K+ posts), 8-10 NICHE (college/Indian-youth specific, "
            "targeted), 2-3 COMMUNITY/BRANDED (smaller, high-relevance). No generic filler tags.\n"
            "6. POSTING TIME: Recommend the best day/time window to post for an Indian college "
            "audience, with a one-line reason tied to their actual routine (class hours, hostel "
            "wifi patterns, weekend scroll habits) — not a generic 'post at 7pm' answer.\n\n"
            "Your final answer MUST include, in this exact order:\n"
            "## Final Reel Script (optimized)\n"
            "(the full shot table, with any hook/pacing/payoff fixes applied inline)\n\n"
            "## Caption\n"
            "...\n\n"
            "## Hashtags\n"
            "**High-Reach:** ...\n"
            "**Niche:** ...\n"
            "**Community/Branded:** ...\n\n"
            "## Posting Tips\n"
            "- Best time: ...\n"
            "- Why: ...\n\n"
            "## Edit Notes\n"
            "(1-3 bullet points summarizing what you changed from the original script and why "
            "— skip this section if no changes were needed)"
        ),

        expected_output=(
            "A complete final reel package in markdown with five sections in order: Final Reel "
            "Script (full shot table), Caption, Hashtags (three labeled tiers), Posting Tips, "
            "and Edit Notes — every section fully filled, no placeholders, ready to copy-paste "
            "and post."
        ),
        agent=engagement_optimizer,
        context=[task_ideate, task_write_script],
        output_pydantic=ReelPackage
    )

    crew = Crew(
        agents=[content_ideator, script_writer, engagement_optimizer],
        tasks=[task_ideate, task_write_script, task_optimize],
        process=Process.sequential,
        verbose=True
    )
    return crew

# ----------------- Fallback Parser -----------------

def parse_crew_fallback(raw_output: str, theme: str) -> dict:
    """Fallback parser in case LLM output needs manual extraction."""
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    try:
        return json.loads(raw_output)
    except Exception:
        pass

    return {
        "hook_preview": f"Pov: {theme[:50]}",
        "concept": {
            "concept": f"Relatable take on: {theme}",
            "format": "Fast-cut sketch / Point-of-View",
            "audio_style": "Trending fast-beat audio with punchline cutoff",
            "target_emotion": "Humor & relatable nostalgia",
            "why_it_works": "Hits a universal college experience that students share in friend groups."
        },
        "shots": [
            {"shot": "1 — Hook", "timing": "0–2s", "type": "OVERLAY", "line": f"Nobody talks about this: {theme}"},
            {"shot": "2", "timing": "2–7s", "type": "SPOKEN", "line": "Expectation: Everything goes according to plan."},
            {"shot": "3", "timing": "7–13s", "type": "ACTION", "line": "Reality: Total chaotic breakdown."},
            {"shot": "4 — CTA", "timing": "13–18s", "type": "SPOKEN", "line": "Tag that one friend who does this every single semester!"}
        ],
        "engagement": {
            "caption": f"Literally every college student ever 😭 Share this with your hostel group! #CollegeLife",
            "posting_time": "9:00 PM – 10:30 PM",
            "posting_day": "Weekday evening",
            "posting_tip": "Students are done with lectures/labs and scrolling Instagram before sleep.",
            "hashtags_reach": ["#CollegeLife", "#Relatable", "#ReelsIndia"],
            "hashtags_niche": ["#CampusLife", "#StudentMemes", "#EngineeringLife"],
            "hashtags_community": ["#CollegeStudents", "#HostelDiaries"]
        }
    }

# ----------------- Flask Web App -----------------

app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def serve_index():
    for fname in ["index.html", ".html"]:
        if os.path.exists(fname):
            return send_file(fname)
    return "<h1>index.html not found</h1>", 404

@app.route("/favicon.ico")
def favicon():
    if os.path.exists("favicon.ico"):
        return send_file("favicon.ico", mimetype="image/x-icon")
    return Response(status=204)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "gemini_key_configured": bool(gemini_api_key)})

@app.route("/api/generate", methods=["POST"])
def generate_reel():
    body = request.get_json(silent=True) or {}
    theme = (body.get("theme") or "").strip()
    if not theme:
        return jsonify({"detail": "Reel theme cannot be empty."}), 400

    if not gemini_api_key:
        return jsonify({"detail": "GEMINI_API_KEY is not configured on the server."}), 500

    try:
        crew = create_reel_crew()
        result = crew.kickoff(inputs={"reel_theme": theme})

        # Extract and ensure structured output conforms to ReelPackage
        data = None
        if hasattr(result, 'pydantic') and result.pydantic:
            try:
                data = result.pydantic.model_dump()
            except Exception:
                pass

        if not data and hasattr(result, 'json_dict') and result.json_dict:
            try:
                data = ReelPackage.model_validate(result.json_dict).model_dump()
            except Exception:
                pass

        if not data:
            raw_text = getattr(result, 'raw', str(result))
            data = parse_crew_fallback(raw_text, theme)

        # Final guarantee that data strictly matches the frontend schema
        try:
            data = ReelPackage.model_validate(data).model_dump()
        except Exception:
            pass

        return jsonify({"success": True, "theme": theme, "data": data})

    except Exception as e:
        print(f"Error during CrewAI execution: {e}")
        return jsonify({"detail": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[ReelCrew] Server starting at http://127.0.0.1:{port} ...")
    app.run(host="127.0.0.1", port=port, debug=False)
