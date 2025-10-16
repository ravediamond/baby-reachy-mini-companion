---
config:
  layout: fixed
  flowchart:
    htmlLabels: true
---
flowchart TB
    User["User"] <-- voice --> UI@{ label: "<b>UI Layer</b><br><span style=\"font-size:11px;color:#01579b\">Gradio / Console</span>" }
    UI <-- audio --> OpenAI@{ label: "<b>OpenAI Realtime Speech</b><br><span style=\"font-size:11px;color:#4a148c\">Tool Calls</span>" }
    OpenAI -- tool calls --> Tools@{ label: "<b>Tool System</b><br><span style=\"font-size:10px;color:#01579b\">Dispatcher + Handlers</span>" }
    OpenAI -- audio deltas --> Motion@{ label: "<b>Motion Control</b><br><span style=\"font-size:11px;color:#01579b\">Audio Offsets + Tracking + Commands</span>" }
    Tools -- commands --> Motion
    Tools <-- vision requests --> Vision@{ label: "<b>Vision</b><br><span style=\"font-size:11px;color:#4a148c\">Local VLM / OpenAI</span>" }
    Tools -- frame requests --> Camera["<b>Camera Worker +<br>Face Tracking</b>"]
    Tools -- tool results --> OpenAI
    Robot["<b>reachy_mini repo</b>"] -- frames --> Camera
    Camera -- frames --> Vision
    Camera -- tracking offsets --> Motion
    Motion -- commands --> Robot
    UI@{ shape: rect}
    OpenAI@{ shape: rect}
    Tools@{ shape: rect}
    Motion@{ shape: rect}
    Vision@{ shape: rect}
     User:::userStyle
     UI:::uiStyle
     OpenAI:::aiStyle
     Tools:::coreStyle
     Motion:::coreStyle
     Vision:::aiStyle
     Camera:::coreStyle
     Robot:::hardwareStyle
    classDef userStyle fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef uiStyle fill:#b3e5fc,stroke:#0277bd,stroke-width:2px
    classDef aiStyle fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    classDef coreStyle fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef hardwareStyle fill:#ef9a9a,stroke:#c62828,stroke-width:3px
