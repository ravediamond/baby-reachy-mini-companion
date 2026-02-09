# /// script
# dependencies = ["gradio>=5.0"]
# ///
import gradio as gr

custom_css = """
/* ---- Reset & Base ---- */
* { box-sizing: border-box; }
.gradio-container { background: transparent !important; }

.page { max-width: 960px; margin: 0 auto; padding: 3rem 1.5rem 2rem; font-family: 'Inter', system-ui, -apple-system, sans-serif; }

/* ---- Hero ---- */
.hero { text-align: center; margin-bottom: 3rem; }
.hero-badge {
    display: inline-block; padding: 0.35rem 1rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase;
    background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%); color: #fff;
    margin-bottom: 1.25rem;
}
.hero h1 {
    font-size: 2.75rem; font-weight: 800; line-height: 1.15; margin: 0 0 0.75rem;
    color: var(--body-text-color);
}
.hero p {
    font-size: 1.15rem; line-height: 1.6; color: var(--body-text-color-subdued);
    max-width: 640px; margin: 0 auto;
}

/* ---- Sections ---- */
.section-label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: #6366f1; margin-bottom: 0.5rem;
}
.section-title { font-size: 1.5rem; font-weight: 700; margin: 0 0 1.25rem; color: var(--body-text-color); }

/* ---- Cards ---- */
.card {
    background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
    border-radius: 0.875rem; padding: 1.75rem; transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover { border-color: #6366f1; box-shadow: 0 4px 24px rgba(99,102,241,0.08); }

.card-icon {
    width: 2.75rem; height: 2.75rem; border-radius: 0.625rem; display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; margin-bottom: 1rem; flex-shrink: 0;
}
.card-icon-purple { background: rgba(99,102,241,0.12); }
.card-icon-pink   { background: rgba(236,72,153,0.12); }
.card-icon-blue   { background: rgba(59,130,246,0.12); }
.card-icon-green  { background: rgba(16,185,129,0.12); }
.card-icon-amber  { background: rgba(245,158,11,0.12); }
.card-icon-red    { background: rgba(239,68,68,0.12); }

.card h3 { font-size: 1.05rem; font-weight: 600; margin: 0 0 0.4rem; color: var(--body-text-color); }
.card p  { font-size: 0.9rem; line-height: 1.55; color: var(--body-text-color-subdued); margin: 0; }

/* ---- Feature Grid ---- */
.feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2.5rem; }
@media (max-width: 700px) { .feature-grid { grid-template-columns: 1fr; } }

/* ---- Install Panel ---- */
.install-panel {
    background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
    border-radius: 0.875rem; padding: 2rem; margin-bottom: 2.5rem;
}
.install-panel h2 { font-size: 1.3rem; font-weight: 700; margin: 0 0 1rem; color: var(--body-text-color); }

.install-steps { list-style: none; padding: 0; margin: 0; counter-reset: step; }
.install-steps li {
    counter-increment: step; position: relative; padding-left: 2.75rem; margin-bottom: 1rem;
    font-size: 0.95rem; line-height: 1.5; color: var(--body-text-color-subdued);
}
.install-steps li::before {
    content: counter(step); position: absolute; left: 0; top: 0;
    width: 2rem; height: 2rem; border-radius: 0.5rem; font-size: 0.85rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    background: rgba(99,102,241,0.1); color: #6366f1;
}

.callout {
    margin-top: 1.25rem; padding: 1rem 1.25rem; border-radius: 0.5rem; font-size: 0.875rem;
    line-height: 1.5;
    background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.15);
    color: var(--body-text-color);
}

/* ---- Deployment Tags ---- */
.deploy-tags { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 2.5rem; justify-content: center; }
.deploy-tag {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.8rem; font-weight: 500;
    background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
    color: var(--body-text-color-subdued);
}
.deploy-tag-dot { width: 6px; height: 6px; border-radius: 50%; }
.dot-green  { background: #10b981; }
.dot-blue   { background: #3b82f6; }
.dot-purple { background: #6366f1; }

/* ---- Footer ---- */
.footer {
    text-align: center; padding-top: 2rem; margin-top: 1rem;
    border-top: 1px solid var(--border-color-primary);
    color: var(--body-text-color-subdued); font-size: 0.8rem; line-height: 1.6;
}
.footer a { color: #6366f1; text-decoration: none; }
.footer a:hover { text-decoration: underline; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=custom_css,
    title="Baby Reachy-Mini Companion",
) as demo:

    gr.HTML("""
    <div class="page">

        <!-- Hero -->
        <div class="hero">
            <div class="hero-badge">Reachy Mini App</div>
            <h1>Baby Reachy-Mini Companion</h1>
            <p>A fully local AI companion for babies and kids. Voice interaction, baby monitoring,
            vision, and expressive motion &mdash; all running privately on your own hardware.</p>
        </div>

        <!-- Deploy tags -->
        <div class="deploy-tags">
            <span class="deploy-tag"><span class="deploy-tag-dot dot-green"></span> Mac (Ollama)</span>
            <span class="deploy-tag"><span class="deploy-tag-dot dot-blue"></span> Mac + Jetson vLLM</span>
            <span class="deploy-tag"><span class="deploy-tag-dot dot-purple"></span> Jetson Orin (fully local)</span>
        </div>

        <!-- Features -->
        <div class="section-label">Capabilities</div>
        <div class="section-title">What it can do</div>

        <div class="feature-grid">
            <div class="card">
                <div class="card-icon card-icon-pink">&#x1F476;</div>
                <h3>Baby Monitor</h3>
                <p>Detects crying via on-device YAMNet audio classification, soothes the baby automatically, and sends you a Signal alert.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-purple">&#x1F3A4;</div>
                <h3>Voice Conversation</h3>
                <p>Natural speech interaction using local STT (Faster-Whisper), LLM (Qwen via Ollama or vLLM), and TTS (Kokoro).</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-blue">&#x1F441;</div>
                <h3>Vision</h3>
                <p>Sees and describes the world through the camera using a local multimodal LLM. Play "I Spy" or ask "What do you see?"</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-green">&#x1F4F1;</div>
                <h3>Remote Alerts</h3>
                <p>Sends instant notifications and photos to your phone via Signal when the baby needs attention.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-amber">&#x1F57A;</div>
                <h3>Expressive Motion</h3>
                <p>Dances, emotional antenna expressions, face tracking, and speech-reactive head movement.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-red">&#x1F50A;</div>
                <h3>Sound Awareness</h3>
                <p>Reacts to environmental sounds &mdash; laughter, coughing, alarms &mdash; for context-aware autonomous responses.</p>
            </div>
        </div>

        <!-- Install -->
        <div class="install-panel">
            <h2>Getting Started</h2>
            <ol class="install-steps">
                <li>Install the <strong>Reachy Mini SDK</strong> on your machine.</li>
                <li>Clone the repository and install with <code>uv sync --extra local</code>.</li>
                <li>Start your LLM server (Ollama, vLLM, or any OpenAI-compatible endpoint).</li>
                <li>Configure <code>.env</code> &mdash; or use the built-in Settings UI in headless mode.</li>
                <li>Run <code>uv run reachy-mini-conversation-app</code> and start talking.</li>
            </ol>
            <div class="callout">
                This Space hosts the application code and documentation.
                The app runs locally on your robot &mdash; no cloud required.
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            Built for <a href="https://www.pollen-robotics.com/" target="_blank">Reachy Mini</a>
            &nbsp;&middot;&nbsp; Powered by local AI
            &nbsp;&middot;&nbsp; <a href="https://github.com/ravediamond/baby-reachy-mini-companion" target="_blank">Source on GitHub</a>
        </div>

    </div>
    """)

if __name__ == "__main__":
    demo.launch()
