# ReelCrew 🎬

> **Automated Campus Reel Pipeline powered by CrewAI and Google Gemini.**

ReelCrew transforms a single reel theme into a complete, ready-to-shoot short-form video package tailored for Indian college audiences.

---

## 🚀 Features

- **Multi-Agent Architecture**:
  1. **Ideation Specialist**: Brainstorms 3 distinct viral angles and picks the highest-impact concept.
  2. **Shot-by-Shot Scriptwriter**: Crafts a 15–30s script with exact camera angles, Hinglish dialogue, and visual actions.
  3. **Engagement Optimizer**: Generates hook-optimized captions, 3-tier hashtags (Reach / Niche / Community), and peak posting times.
- **Modern Interactive UI**: Real-time pipeline progression, live mobile reel preview, and 1-click clipboard copy.
- **Structured JSON Engine**: Powered by Pydantic models for reliable data delivery.

---

## 🛠️ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Script writter"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=5000
```

### 4. Run the application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
# reel-script-writing-agent
