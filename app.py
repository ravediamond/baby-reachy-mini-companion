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

/* ---- Architecture Diagram ---- */
.arch-image {
    max-width: 900px !important; margin: 0 auto 2.5rem !important;
}
.arch-image img {
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
            <div class="diff-item"><strong>7+ AI Models</strong> &mdash; VAD, STT, TTS, YOLO, YAMNet, and a single vision-language model for both conversation and sight</div>
            <div class="diff-item"><strong>Autonomous Intelligence</strong> &mdash; The robot reasons about what to do: hears crying and decides to soothe, spots danger and alerts you, answers questions by looking around. Not scripted &mdash; it thinks</div>
            <div class="diff-item"><strong>One VLM Does It All</strong> &mdash; A single 3B&ndash;4B vision-language model handles text conversation, visual understanding, and tool-calling decisions &mdash; no separate models needed</div>
            <div class="diff-item"><strong>Jetson vLLM</strong> &mdash; Offload inference to a Jetson Orin via NVIDIA AI containers with quantized models at 25+ tokens/s</div>
            <div class="diff-item"><strong>Concurrent Pipeline</strong> &mdash; 100Hz motion, 30Hz camera, speech detection, and safety scanning run in parallel on consumer hardware</div>
        </div>

        <!-- Features -->
        <div class="section-label">Features</div>
        <div class="section-title">What it does for your family</div>

        <div class="feature-grid">
            <div class="card">
                <div class="card-icon card-icon-pink">&#x1F476;</div>
                <h3>Baby Safety Monitor</h3>
                <p>Listens for crying and scans for dangerous objects near the baby. Automatically soothes with gentle rocking and calming words, and sends you a photo alert via Signal from another room.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-purple">&#x1F3A4;</div>
                <h3>Interactive Learning</h3>
                <p>Teaches your child through natural conversation &mdash; counting, colors, animals, and language practice. The robot listens, responds, and adapts. Screen-free learning through voice alone.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-blue">&#x1F319;</div>
                <h3>Soothe &amp; Comfort</h3>
                <p>Speaks gentle, calming words with slow rocking motions to comfort a crying baby. Triggered automatically when crying is detected, or on demand.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-amber">&#x2728;</div>
                <h3>Story Time</h3>
                <p>Reads classic children's stories (Three Little Pigs, Goldilocks) with expressive narration and emotional prosody. A screen-free storytime experience.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-green">&#x1F441;</div>
                <h3>Contextual Awareness</h3>
                <p>Combines what it hears and sees to understand the situation. Detects crying through audio, spots dangerous objects through its camera, and can describe the world around it when asked.</p>
            </div>
            <div class="card">
                <div class="card-icon card-icon-red">&#x1F57A;</div>
                <h3>Expressive &amp; Alive</h3>
                <p>Dances, wiggles its antennas to show emotions, tracks faces, and moves its head while speaking. A companion that feels present and responsive, not a static speaker.</p>
            </div>
        </div>

        <!-- Disclaimer -->
        <div class="callout" style="margin-bottom: 2.5rem;">
            <strong>Note:</strong> This is a personal project and technology demonstration &mdash; not a finished product.
            It is not intended to replace parental supervision or serve as a certified childcare device.
            Always supervise your child around any robotic device.
        </div>

        <!-- Architecture -->
        <div class="section-label">Under the hood</div>
        <div class="section-title">Architecture</div>

    </div>
    """)

    gr.Image(
        value="docs/assets/architecture.svg",
        show_label=False,
        interactive=False,
        container=False,
        elem_classes=["arch-image"],
    )

    gr.HTML("""
    <div class="page">

        <!-- Install -->
        <div class="install-panel">
            <h2>Getting Started</h2>
            <ol class="install-steps">
                <li>Download <strong>Baby Reachy-Mini Companion</strong> from the Reachy Mini app store.</li>
                <li>Install <a href="https://ollama.com" target="_blank">Ollama</a> and pull a model: <code>ollama pull ministral-3:3b</code></li>
                <li>Launch the app &mdash; a settings dashboard opens in your browser.</li>
                <li>Fill in the LLM server URL, model name, select your microphone, and toggle the features you want.</li>
                <li>Click <strong>Start</strong> and your baby's companion is ready!</li>
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
