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
    max-width: 100% !important; margin: 0 auto 2.5rem !important;
}
.arch-image img {
    width: 100% !important;
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

/* ---- Feature Tags ---- */
.tag-row { display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center; margin-bottom: 0.75rem; }
.tag-row:last-child { margin-bottom: 2.5rem; }
.feat-tag {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.45rem 0.95rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
    background: rgba(99,102,241,0.10); color: var(--body-text-color);
}
.tech-tag {
    display: inline-flex; align-items: center;
    padding: 0.35rem 0.85rem; border-radius: 0.4rem; font-size: 0.78rem; font-weight: 500;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    background: transparent; border: 1px solid rgba(99,102,241,0.3); color: #6366f1;
}

/* ---- Mission / Testimonial ---- */
.mission {
    position: relative; margin-bottom: 2.5rem; padding: 2.25rem 2.5rem 2rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(236,72,153,0.06) 100%);
    border: 1px solid rgba(99,102,241,0.15); border-left: 4px solid #6366f1; border-radius: 0.875rem;
}
.mission .quote-mark {
    position: absolute; top: 0.75rem; left: 1.25rem;
    font-size: 4rem; line-height: 1; color: rgba(99,102,241,0.15); font-family: Georgia, serif;
    pointer-events: none;
}
.mission blockquote {
    margin: 0; padding: 0; font-size: 1.08rem; line-height: 1.75; color: var(--body-text-color);
    font-style: italic; max-width: 720px; margin: 0 auto; text-align: justify;
}
.mission blockquote strong { font-style: normal; color: #6366f1; }
.mission .signature {
    font-size: 0.9rem; color: var(--body-text-color-subdued); margin-top: 1rem;
    font-style: normal; font-weight: 600; text-align: right;
}

/* ---- Demo GIFs ---- */
.demo-section { margin-bottom: 2.5rem; }
.demo-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.25rem; }
@media (max-width: 700px) { .demo-grid { grid-template-columns: 1fr; } }
.demo-item {
    background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
    border-radius: 0.875rem; overflow: hidden; transition: border-color 0.2s, box-shadow 0.2s;
}
.demo-item:hover { border-color: #6366f1; box-shadow: 0 4px 24px rgba(99,102,241,0.08); }
.demo-item img {
    width: 100%; max-height: 300px; object-fit: contain; display: block; border-bottom: 1px solid var(--border-color-primary);
    background: #000;
}
.demo-caption {
    padding: 0.875rem 1rem;
}
.demo-caption h3 { font-size: 0.95rem; font-weight: 600; margin: 0 0 0.25rem; color: var(--body-text-color); }
.demo-caption p { font-size: 0.8rem; line-height: 1.45; color: var(--body-text-color-subdued); margin: 0; }

/* ---- Differentiators ---- */
.diff-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 2.5rem; }
@media (max-width: 700px) { .diff-grid { grid-template-columns: 1fr; } }
.diff-item {
    display: flex; flex-direction: column; gap: 0.5rem; padding: 1.35rem 1.5rem;
    background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
    border-left: 3px solid #6366f1;
    border-radius: 0.75rem; font-size: 0.9rem; line-height: 1.6; color: var(--body-text-color-subdued);
    text-align: justify;
}
.diff-item strong { color: var(--body-text-color); font-size: 1rem; font-weight: 700; display: block; letter-spacing: -0.01em; }

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
            <p>A Reachy Mini that <strong>thinks, decides, and acts on its own</strong>. Voice interaction, baby safety monitoring,
            vision, and expressive motion &mdash; autonomous, fully local, running on consumer hardware you already own.</p>
        </div>
    </div>
    """)

    gr.Image(
        value="docs/assets/reachy-local.jpg",
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
                Autonomous. Local. Affordable. &mdash; 7 AI models orchestrated on-device.
                The robot reasons through a vision-language model with tool calling, not pre-written scripts.
                Runs on consumer hardware. No cloud APIs, no subscriptions, no data leaves your home.
            </p>
        </div>

        <!-- Deploy tags -->
        <div class="deploy-tags">
            <span class="deploy-tag"><span class="deploy-tag-dot dot-green"></span> Mac (Ollama)</span>
            <span class="deploy-tag"><span class="deploy-tag-dot dot-blue"></span> Mac + Jetson vLLM</span>
            <a href="https://github.com/ravediamond/baby-reachy-mini-companion" target="_blank" style="text-decoration:none;">
                <span class="deploy-tag"><span class="deploy-tag-dot dot-purple"></span> GitHub &rarr; Full Documentation</span>
            </a>
        </div>

        <!-- Feature & tech tags -->
        <div class="tag-row">
            <span class="feat-tag">&#x1F9E0; 7 AI Models</span>
            <span class="feat-tag">&#x1F3A4; Voice Conversation</span>
            <span class="feat-tag">&#x1F476; Cry Detection</span>
            <span class="feat-tag">&#x1F52D; YOLO Vision</span>
            <span class="feat-tag">&#x1F6E1; Danger Alerts</span>
            <span class="feat-tag">&#x1F529; 16+ Tools</span>
        </div>
        <div class="tag-row" style="margin-bottom: 2.5rem;">
            <span class="tech-tag">Ollama</span>
            <span class="tech-tag">Faster-Whisper</span>
            <span class="tech-tag">Kokoro TTS</span>
            <span class="tech-tag">Silero VAD</span>
            <span class="tech-tag">YAMNet</span>
            <span class="tech-tag">YOLO</span>
            <span class="tech-tag">Jetson Orin</span>
        </div>

        <!-- Demo GIFs -->
        <div class="demo-section">
            <div class="section-label">See it in action</div>
            <div class="section-title">What it actually does</div>
            <div class="demo-grid">
                <div class="demo-item">
                    <img src="https://huggingface.co/spaces/ravediamond/baby-reachy-mini-companion/resolve/main/docs/assets/demo-cry-soothe.gif" alt="Baby cry detection and soothing demo" />
                    <div class="demo-caption">
                        <h3>Baby Cry Detection &rarr; Soothe &rarr; Alert</h3>
                        <p>YAMNet detects crying, robot soothes with rocking and calming words, photo alert sent to parent via Signal.</p>
                    </div>
                </div>
                <div class="demo-item">

                    <img src="https://huggingface.co/spaces/ravediamond/baby-reachy-mini-companion/resolve/main/docs/assets/demo-danger.gif" alt="Danger detection demo" />
                    <div class="demo-caption">
                        <h3>Danger Detection &rarr; Instant Alert</h3>
                        <p>YOLO spots scissors or knives, robot speaks a warning, photo sent directly to parent.</p>
                    </div>
                </div>
                <div class="demo-item">

                    <img src="https://huggingface.co/spaces/ravediamond/baby-reachy-mini-companion/resolve/main/docs/assets/demo-teaching.gif" alt="Interactive teaching demo" />
                    <div class="demo-caption">
                        <h3>Interactive Teaching</h3>
                        <p>Ask anything &mdash; the robot explains concepts at a child's level through natural conversation.</p>
                    </div>
                </div>
                <div class="demo-item">

                    <img src="https://huggingface.co/spaces/ravediamond/baby-reachy-mini-companion/resolve/main/docs/assets/demo-story.gif" alt="Story time demo" />
                    <div class="demo-caption">
                        <h3>Story Time</h3>
                        <p>Classic children's stories with expressive narration, head movement, and emotional prosody.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Mission -->
        <div class="mission">
            <span class="quote-mark">&ldquo;</span>
            <blockquote>
                When my son was born, I asked myself a simple question: <strong>what kind of technology do I actually want around him?</strong>
                Not something that sends his voice to a server I don't control. Not something that costs a monthly subscription to keep running.
                Something I built myself, that I understand, that stays in our home.
                <br><br>
                This project is my answer. A robot that <strong>listens when he cries and comforts him</strong>.
                That <strong>watches over him and alerts me if something is wrong</strong>.
                That will <strong>teach him, tell him stories, and grow alongside him</strong>.
                All running locally, on hardware any family could afford.
                <br><br>
                I don't know if it's perfect. But I know it's built with love, and every line of code
                exists because I want the best for my child.
            </blockquote>
            <p class="signature">&mdash; Ravin, new dad &amp; builder</p>
        </div>

        <!-- What makes this different -->
        <div class="section-label">Why it matters</div>
        <div class="section-title">What makes this different</div>

        <div class="diff-grid">
            <div class="diff-item"><strong>Truly autonomous</strong><span>A VLM with tool calling reasons about what to do: hears crying &rarr; soothes; spots a knife &rarr; warns and alerts. The robot decides and acts on its own.</span></div>
            <div class="diff-item"><strong>Fully local</strong><span>7 AI models on your hardware. No internet required. Audio and video never leave your home.</span></div>
            <div class="diff-item"><strong>Consumer hardware</strong><span>Mac + $700 Jetson Orin NX. One-time hardware cost, no subscriptions. That's how robotics reaches families.</span></div>
            <div class="diff-item"><strong>Safe by design</strong><span>Cry and danger alerts sent directly in code &mdash; guaranteed delivery, never gated on the LLM. Empathy first.</span></div>
        </div>

        <!-- Architecture -->
        <div class="section-label">Under the hood</div>
        <div class="section-title">Architecture &mdash; 7 models, one pipeline</div>

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
