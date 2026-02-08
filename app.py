import gradio as gr

# Custom CSS to mimic the project's style
custom_css = """
.container { max-width: 900px; margin: auto; padding-top: 2rem; }
.hero-header { text-align: center; margin-bottom: 2rem; }
.hero-title { font-size: 3rem; font-weight: 800; background: -webkit-linear-gradient(45deg, #ff5c70, #45c4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
.hero-subtitle { font-size: 1.2rem; color: var(--body-text-color-subdued); font-weight: 400; }
.install-card { background: var(--background-fill-secondary); border-radius: 1rem; padding: 2rem; border: 1px solid var(--border-color-primary); margin-bottom: 2rem; box-shadow: var(--shadow-drop); }
.feature-card { background: var(--background-fill-secondary); border-radius: 0.75rem; padding: 1.5rem; border: 1px solid var(--border-color-primary); height: 100%; transition: all 0.2s; }
.feature-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-drop); border-color: var(--color-accent); }
.feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.feature-title { font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--body-text-color); }
.feature-desc { color: var(--body-text-color-subdued); line-height: 1.5; }
.footer-note { text-align: center; color: var(--body-text-color-subdued); font-size: 0.875rem; margin-top: 3rem; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css, title="Baby Reachy-Mini Companion") as demo:
    with gr.Column(elem_classes="container"):
        
        # Hero Section
        with gr.Column(elem_classes="hero-header"):
            gr.HTML('<div class="hero-title">🤖🍼 Baby Reachy-Mini Companion</div>')
            gr.HTML('<div class="hero-subtitle">A fully local AI companion for babies and kids, designed for the Reachy Mini robot.</div>')

        # Main Content
        with gr.Row(equal_height=True):
            # Left Column: Installation
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="install-card">
                    <h2 style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: var(--body-text-color);">🚀 How to Install</h2>
                    <ol style="padding-left: 1.5rem; color: var(--body-text-color-subdued); line-height: 2;">
                        <li>Open your <strong>Reachy Mini Dashboard</strong>.</li>
                        <li>Navigate to the <strong>App Store</strong>.</li>
                        <li>Search for <strong>"Baby Reachy-Mini Companion"</strong>.</li>
                        <li>Click <strong>Install</strong> to start the download.</li>
                    </ol>
                    <div style="margin-top: 1.5rem; padding: 1rem; background: var(--block-background-fill); border-radius: 0.5rem; color: var(--body-text-color); font-size: 0.9rem; border: 1px solid var(--border-color-primary);">
                        <strong>Note:</strong> This Space hosts the application code. The actual magic happens locally on your robot!
                    </div>
                </div>
                """)

            # Right Column: Visual
            with gr.Column(scale=1):
                # Using the relative path to the image in the repo
                gr.Image("docs/assets/baby-reachy-mini.jpg", show_label=False, container=False, elem_id="hero-image")

        # Features Grid
        gr.HTML('<h2 style="font-size: 1.8rem; font-weight: 700; margin: 2rem 0 1.5rem; text-align: center; color: var(--body-text-color);">✨ Key Features</h2>')
        
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="feature-card">
                    <div class="feature-icon">👶</div>
                    <div class="feature-title">Smart Baby Monitor</div>
                    <div class="feature-desc">Automatically detects crying sounds and uses gentle motion and soothing sounds to comfort the baby.</div>
                </div>
                """)
            with gr.Column():
                gr.HTML("""
                <div class="feature-card">
                    <div class="feature-icon">👂</div>
                    <div class="feature-title">Sound Awareness</div>
                    <div class="feature-desc">Detects and reacts to environmental sounds like laughter, coughing, or alarms for context-aware interaction.</div>
                </div>
                """)
            with gr.Column():
                gr.HTML("""
                <div class="feature-card">
                    <div class="feature-icon">🗣️</div>
                    <div class="feature-title">Friendly Chat</div>
                    <div class="feature-desc">Engages in safe, age-appropriate conversations using a locally running Large Language Model.</div>
                </div>
                """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="feature-card">
                    <div class="feature-icon">💃</div>
                    <div class="feature-title">Expressive Motion</div>
                    <div class="feature-desc">Performs fun dances and head gestures to entertain and interact with kids.</div>
                </div>
                """)
                gr.HTML("""
                <div class="feature-card">
                    <div class="feature-icon">👀</div>
                    <div class="feature-title">Interactive Vision</div>
                    <div class="feature-desc">"Sees" toys and people to play interactive games like "I Spy" or describe the room.</div>
                </div>
                """)
            with gr.Column():
                gr.HTML("""
                <div class="feature-card">
                    <div class="feature-icon">📱</div>
                    <div class="feature-title">Parent Alerts</div>
                    <div class="feature-desc">Sends instant notifications to your phone via Signal if the baby needs your attention.</div>
                </div>
                """)

        # Footer
        gr.HTML('<div class="footer-note">Built with ❤️ for Reachy Mini • Powered by Local AI</div>')

if __name__ == "__main__":
    demo.launch()