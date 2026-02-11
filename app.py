# /// script
# dependencies = ["gradio>=6.0"]
# ///
import gradio as gr


custom_css = """
/* ---- Reset & Base ---- */
* { box-sizing: border-box; }
.gradio-container { background: transparent !important; max-width: 100% !important; }

.page { max-width: 1200px; margin: 0 auto; padding: 3rem 1.5rem 2rem; font-family: 'Inter', system-ui, -apple-system, sans-serif; }

/* ---- Hero ---- */
.hero { text-align: center; margin-bottom: 2.5rem; }
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
    max-width: 720px; margin: 0 auto;
}

/* ---- Hero Image ---- */
.hero-image {
    max-width: 560px !important; margin: 0 auto 2.5rem !important;
}
.hero-image img {
    border-radius: 1rem !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
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

/* ---- Benefit Grid ---- */
.benefit-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 2.5rem; }
@media (max-width: 700px) { .benefit-grid { grid-template-columns: 1fr; } }

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

/* ---- Mission ---- */
.mission {
    text-align: center; margin-bottom: 2.5rem; padding: 1.75rem 2rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(236,72,153,0.06) 100%);
    border: 1px solid rgba(99,102,241,0.15); border-radius: 0.875rem;
}
.mission p {
    font-size: 1.05rem; line-height: 1.7; color: var(--body-text-color);
    max-width: 720px; margin: 0 auto;
}
.mission .signature {
    font-size: 0.9rem; color: var(--body-text-color-subdued); margin-top: 0.75rem;
    font-style: italic;
}

/* ---- Differentiators ---- */
.diff-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; margin-bottom: 2.5rem; }
@media (max-width: 700px) { .diff-grid { grid-template-columns: 1fr; } }
.diff-item {
    display: flex; align-items: baseline; gap: 0.5rem; padding: 0.75rem 1rem;
    background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
    border-radius: 0.625rem; font-size: 0.9rem; color: var(--body-text-color-subdued);
}
.diff-item strong { color: var(--body-text-color); }

/* ---- Footer ---- */
.footer {
    text-align: center; padding-top: 2rem; margin-top: 1rem;
    border-top: 1px solid var(--border-color-primary);
    color: var(--body-text-color-subdued); font-size: 0.8rem; line-height: 1.6;
}
.footer a { color: #6366f1; text-decoration: none; }
.footer a:hover { text-decoration: underline; }
"""

with gr.Blocks(title="Baby Reachy-Mini Companion") as demo:

    gr.HTML("""
    <div class="page">

        <!-- Hero -->
        <div class="hero">
            <div class="hero-badge">Reachy Mini App</div>
            <h1>Baby Reachy-Mini Companion</h1>
            <p>A fully local AI companion for babies and kids. Voice interaction, baby monitoring,
            vision, and expressive motion &mdash; all running privately on your own hardware.</p>
        </div>
    </div>
    """)

    gr.Image(
        value="docs/assets/baby-reachy-mini.jpg",
        show_label=False,
        interactive=False,
        container=False,
        elem_classes=["hero-image"],
    )

    gr.Image(
        value="docs/assets/reachy.gif",
        show_label=False,
        interactive=False,
        container=False,
        elem_classes=["hero-image"],
    )

    gr.HTML("""
    <div class="page">

        <!-- At a glance -->
        <div style="text-align:center; margin-bottom: 2rem; padding: 1rem 1.5rem;
                    background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(236,72,153,0.10) 100%);
                    border: 1px solid rgba(99,102,241,0.25); border-radius: 0.75rem;">
            <p style="font-size: 1.05rem; font-weight: 600; margin: 0; color: var(--body-text-color);">
                The only fully local Reachy Mini AI stack &mdash; 7 AI models running concurrently,
                autonomous baby safety monitoring, tested on NVIDIA Jetson Orin NX.
                No cloud. No data leaves your home.
            </p>
        </div>

        <!-- Deploy tags -->
        <div class="deploy-tags">
            <span class="deploy-tag"><span class="deploy-tag-dot dot-green"></span> Mac (Ollama)</span>
            <span class="deploy-tag"><span class="deploy-tag-dot dot-blue"></span> Mac + Jetson vLLM</span>
        </div>

        <!-- Mission -->
        <div class="mission">
            <p>I'm building a nursery companion that actually respects our family's privacy.
            No cloud, no data leaks &mdash; what happens at home stays at home.
            Proving that high-end robotics can run on consumer hardware instead of massive servers.</p>
            <p class="signature">&mdash; A dad building cool tech for his son</p>
        </div>

        <!-- Principles -->
        <div class="section-label">My belief</div>
        <div class="section-title">Design principles</div>

        <div class="diff-grid">
            <div class="diff-item"><strong>Privacy first</strong> &mdash; Something running in your home, around your child, should never send data to a third party</div>
            <div class="diff-item"><strong>Consumer hardware</strong> &mdash; Runs on a Mac with a $700 Jetson Orin NX for GPU inference &mdash; not a data center. That's how robotics reaches homes</div>
            <div class="diff-item"><strong>Physically safe</strong> &mdash; Reachy Mini has no hands or manipulators &mdash; it can express and communicate, not grab or push. Its antennas are only used for emotional expression</div>
            <div class="diff-item"><strong>Empathy matters</strong> &mdash; A robot that ignores human distress has failed. Detecting emotions and responding with care is the goal</div>
        </div>

        <!-- What makes this different -->
        <div class="section-label">Why it matters</div>
        <div class="section-title">What makes this different</div>

        <div class="diff-grid">
            <div class="diff-item"><strong>100% Local</strong> &mdash; No cloud APIs, no internet required</div>
            <div class="diff-item"><strong>7+ AI Models</strong> &mdash; VAD, STT, LLM, TTS, VLM, YOLO, YAMNet on-device</div>
            <div class="diff-item"><strong>Safety Monitor</strong> &mdash; YOLO detects hazards, VLM analyzes, Signal alerts you</div>
            <div class="diff-item"><strong>Jetson vLLM</strong> &mdash; Offload LLM inference to a Jetson Orin via NVIDIA AI containers with quantized models</div>
            <div class="diff-item"><strong>25+ TPS</strong> &mdash; KV cache warmup, streaming TTS, and 3B&ndash;4B models tuned for real-time conversation</div>
            <div class="diff-item"><strong>Concurrent Pipeline</strong> &mdash; 100Hz motion, 30Hz camera, speech detection, and safety scanning run in parallel</div>
        </div>

        <!-- All-in-one companion -->
        <div class="section-label">All-in-one solution</div>
        <div class="section-title">A complete companion for your child</div>

        <div class="benefit-grid">
            <div class="card">
                <div class="card-icon card-icon-pink">&#x1F3AD;</div>
                <h3>Entertain</h3>
                <p>Play, chat, and interact with your child &mdash; all screen-free. Voice conversations, dances, and expressive movements keep them engaged naturally.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-blue">&#x1F319;</div>
                <h3>Soothe &amp; Sleep</h3>
                <p>Sing lullabies and nursery rhymes to calm or put your baby to sleep. Gentle rocking motions and soft speech create a soothing bedtime routine.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-amber">&#x2728;</div>
                <h3>Spark Imagination</h3>
                <p>Tell stories tailored to your child's world &mdash; pick characters, animals, or themes and let the companion weave them into an adventure.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-green">&#x1F393;</div>
                <h3>Learn &amp; Discover</h3>
                <p>Recite the alphabet, explore animals, discover the solar system &mdash; interactive lessons adapted to your child's curiosity and pace.</p>
            </div>
        </div>

        <!-- Features -->
        <div class="section-label">Capabilities</div>
        <div class="section-title">What it can do</div>

        <div class="feature-grid">
            <div class="card">
                <div class="card-icon card-icon-pink">&#x1F476;</div>
                <h3>Baby Monitor</h3>
                <p>Detects crying via YAMNet audio classification and scans for dangerous objects via YOLO. Soothes the baby, triggers VLM analysis, and sends Signal alerts.</p>
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
    demo.launch(
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=custom_css,
        allowed_paths=["docs/assets"],
    )
