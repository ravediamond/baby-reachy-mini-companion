import gradio as gr

def app_description():
    return """
    # 🤖 Reachy Mini Conversation App
    
    This is a local conversation app for the Reachy Mini robot.
    
    ## 🚀 How to Install
    
    1. Open your Reachy Mini Dashboard.
    2. Go to the **App Store**.
    3. Find **Reachy Mini Companion** and click **Install**.
    
    ## ✨ Features
    
    - **🗣️ Voice Interaction:** Talk to Reachy naturally.
    - **👀 Vision:** Reachy can see and describe what it sees.
    - **👶 Baby Monitor:** Detects crying and soothes the baby.
    - **📱 Signal Integration:** Chat with Reachy remotely.
    
    *Note: This Space hosts the application code. The app runs locally on your Reachy Mini robot.*
    """

with gr.Blocks() as demo:
    gr.Markdown(app_description())
    
if __name__ == "__main__":
    demo.launch()
