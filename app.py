import gradio as gr

def app_description():
    return """
    # 🤖🍼 Baby Reachy-Mini Companion
    
    **A fully local AI companion for babies and kids, designed for the Reachy Mini robot.**
    
    This app turns your Reachy Mini into a smart, interactive friend that can entertain, soothe, and watch over your little ones.
    
    ## 🚀 How to Install on Your Robot
    
    1. Open your **Reachy Mini Dashboard**.
    2. Navigate to the **App Store**.
    3. Search for **"Baby Reachy-Mini Companion"**.
    4. Click **Install**.
    
    ## ✨ Key Features
    
    - **👶 Smart Baby Monitor:** Automatically detects crying and uses the robot's motion and voice to soothe the baby.
    - **🗣️ Friendly Voice Interaction:** Chats naturally with kids using a safe, local language model.
    - **👀 Vision Capabilities:** Can "see" toys, people, and objects to play interactive games.
    - **💃 Dance & Move:** Performs expressive dances and head gestures to entertain.
    - **📱 Parent Alerts:** Sends notifications to your phone (via Signal) if the baby needs attention.
    
    ---
    
    *Note: This Space hosts the application code for the Reachy Mini ecosystem. To use these features, you must install the app on a physical Reachy Mini robot.*
    """

with gr.Blocks(title="Baby Reachy-Mini Companion") as demo:
    gr.Markdown(app_description())
    
if __name__ == "__main__":
    demo.launch()
