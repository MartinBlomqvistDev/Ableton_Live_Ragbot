import os

import streamlit as st
from dotenv import load_dotenv
from numpy import dot
from numpy.linalg import norm

load_dotenv()

# Push API key from Streamlit secrets into env so core modules can call os.getenv("API_KEY").
try:
    if "API_KEY" in st.secrets:
        os.environ.setdefault("API_KEY", st.secrets["API_KEY"])
except Exception:
    pass

from config import (
    CHATBOT_TOP_K,
    EMBEDDINGS_PATH,
    EVALUATION_TOP_K,
    NO_ANSWER_EN,
    NO_ANSWER_SV,
)
from core.embeddings import create_embeddings
from core.llm import generate_response
from core.vector_store import VectorStore

st.set_page_config(
    page_title="The Ableton Live 12 RAG-Bot",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🎹",
)

st.markdown("""
<style>
:root {
    --primaryColor: #FF6F61;
    --backgroundColor: #004D4D;
    --secondaryBackgroundColor: #006666;
    --font-family: "Arial, sans-serif";
    --textColor: white;
}
body, main {
    margin: 0; padding: 0; min-height: 100vh;
    background-color: var(--backgroundColor) !important;
    color: var(--textColor) !important;
    font-family: var(--font-family) !important;
}
section.main, div.block-container {
    max-width: 800px !important;
    margin: 80px auto 40px auto !important;
    background-color: var(--secondaryBackgroundColor) !important;
    padding: 30px 40px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    color: var(--textColor) !important;
}
h1, h2, h3 {
    color: var(--primaryColor) !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
}
.stTextInput > div > div > input {
    width: 100% !important;
    background-color: white !important;
    color: black !important;
    border: 2px solid var(--primaryColor) !important;
    border-radius: 6px !important;
    padding: 8px !important;
    font-family: var(--font-family) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #888 !important;
}
div.stButton > button {
    background-color: var(--primaryColor) !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease !important;
    font-family: var(--font-family) !important;
}
div.stButton > button:hover {
    background-color: #e65b50 !important;
}
[data-testid="stSidebar"] {
    background-color: var(--secondaryBackgroundColor) !important;
}
[data-testid="stSidebar"] * {
    color: var(--textColor) !important;
}
div[role="combobox"] > div > div > select {
    background-color: white !important;
    color: black !important;
    border: 2px solid var(--primaryColor) !important;
    border-radius: 6px !important;
    padding: 6px 8px !important;
    font-family: var(--font-family) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_vector_store() -> VectorStore:
    store = VectorStore()
    if store.load(str(EMBEDDINGS_PATH)):
        return store
    st.error(
        f"Embeddings file not found at '{EMBEDDINGS_PATH}'. "
        "Run 'python pipeline/build_index.py' first."
    )
    st.stop()


vector_store = initialize_vector_store()

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
answer_language = st.sidebar.selectbox(
    "Response Language:",
    options=["English", "Swedish"],
    index=0,
)
page = st.sidebar.radio("Select a page", ["Chatbot", "Evaluation", "About the app"], index=0)


# ── Chatbot ──────────────────────────────────────────────────────────────────
if page == "Chatbot":
    st.title("The Ableton Live 12 RAG-Bot")
    query = st.text_input("Ask your question:")
    if query:
        with st.spinner("Searching..."):
            query_emb = create_embeddings([query])[0]
            results = vector_store.search(query_emb, k=CHATBOT_TOP_K)
            top_texts = [r["text"] for r in results]
            answer = generate_response(query, top_texts, answer_language=answer_language)
        st.markdown("### Answer:")
        st.write(answer)


# ── About ─────────────────────────────────────────────────────────────────────
elif page == "About the app":
    st.title("About the app")
    st.write("""
Type a question and the bot finds the most relevant parts of the Ableton Live 12 manual,
then asks Gemini to write an answer from what it found. It won't make things up — if the
manual doesn't cover it, it says so.

The search runs fully offline using a local embedding model. Only the final answer generation
hits the Google API (one call per question).

Built by Martin Blomqvist during the Data Scientist program at EC Utbildning 2025.

- [LinkedIn](https://www.linkedin.com/in/martin-blomqvist)
- [GitHub](https://github.com/MartinBlomqvistDev)
""")


# ── Evaluation ────────────────────────────────────────────────────────────────
elif page == "Evaluation":
    st.title("Evaluate Chatbot Responses")
    st.markdown(
        "Pick a question, run the bot, and see how close its answer is to the ideal one."
    )

    predefined_qa_en = [
        {
            "question": "How do I use automation in Ableton Live to change a parameter over time?",
            "ideal_answer": "Automation is drawn directly in tracks using breakpoint envelopes. Select the parameter you want to automate, and then draw its curve using the pen tool or by clicking and dragging breakpoints.",
        },
        {
            "question": "What is the purpose of the Arrangement View in Ableton Live?",
            "ideal_answer": "The Arrangement View is a linear timeline for recording, arranging, and editing MIDI and audio clips in a traditional song structure.",
        },
        {
            "question": "Explain the function of Sends and Returns in Ableton Live.",
            "ideal_answer": "Sends route a portion of a track's signal to a Return track, where effects can be applied. This allows multiple tracks to share the same effect processing, saving CPU and providing a consistent sound.",
        },
        {
            "question": "How can I create a custom drum rack in Ableton Live?",
            "ideal_answer": "Drag individual samples or instruments into the pads of an empty Drum Rack. You can then configure each pad's settings and effects independently.",
        },
        {
            "question": "What is the difference between hot-swapping and replacing a device?",
            "ideal_answer": "Hot-swapping allows you to audition different devices while keeping their current settings intact. Replacing a device permanently swaps it with a new one, discarding the old device's settings.",
        },
        {
            "question": "How do I consolidate tracks or clips in Ableton Live?",
            "ideal_answer": "Select the desired clips or a range of time across multiple tracks, then go to the Edit menu and choose 'Consolidate Time to New Track' or 'Consolidate' (Cmd/Ctrl+J).",
        },
        {
            "question": "What are Scenes in the Session View and how are they used?",
            "ideal_answer": "Scenes in Session View are horizontal rows that contain a collection of clips, typically representing a section of a song. Launching a scene plays all clips within that row simultaneously, useful for live performance and improvisation.",
        },
        {
            "question": "How can I reduce CPU usage in Ableton Live when my project is complex?",
            "ideal_answer": "To reduce CPU usage, you can freeze tracks, flatten tracks, reduce buffer size, disable unused devices, or use fewer CPU-intensive effects.",
        },
        {
            "question": "Describe the function of the Follow Actions feature for MIDI and audio clips.",
            "ideal_answer": "Follow Actions allow you to define what happens after a clip finishes playing, such as playing another clip, stopping, retriggering itself, or launching a different scene. This is useful for creating dynamic arrangements and generative music.",
        },
        {
            "question": "How do I set up an external MIDI controller in Ableton Live 12?",
            "ideal_answer": "Go to Live's Preferences, then 'Link/Tempo/MIDI'. Select your controller from the 'Control Surface' dropdown, enable its 'Track' and 'Remote' switches in the 'MIDI Ports' section, and ensure its MIDI input is active.",
        },
    ]

    predefined_qa_sv = [
        {
            "question": "Hur använder jag automation i Ableton Live för att ändra en parameter över tid?",
            "ideal_answer": "Automation ritas direkt i spåren med hjälp av brytpunktskuvert. Välj den parameter du vill automatisera och rita sedan dess kurva med pennverktyget eller genom att klicka och dra brytpunkter.",
        },
        {
            "question": "Vad är syftet med Arrangement View i Ableton Live?",
            "ideal_answer": "Arrangement View är en linjär tidslinje för inspelning, arrangering och redigering av MIDI- och ljudklipp i en traditionell låtstruktur.",
        },
        {
            "question": "Förklara funktionen av Sends och Returns i Ableton Live.",
            "ideal_answer": "Sends dirigerar en del av ett spårs signal till ett Return-spår, där effekter kan appliceras. Detta gör att flera spår kan dela samma effektprocessering, vilket sparar CPU och ger ett konsekvent ljud.",
        },
        {
            "question": "Hur kan jag skapa ett anpassat Drum Rack i Ableton Live?",
            "ideal_answer": "Dra individuella samplingar eller instrument till padsen i ett tomt Drum Rack. Du kan sedan konfigurera varje pads inställningar och effekter oberoende av varandra.",
        },
        {
            "question": "Vad är skillnaden mellan hot-swapping och att ersätta en enhet?",
            "ideal_answer": "Hot-swapping låter dig provlyssna olika enheter samtidigt som deras nuvarande inställningar behålls. Att ersätta en enhet byter ut den permanent mot en ny, vilket kasserar den gamla enhetens inställningar.",
        },
        {
            "question": "Hur konsoliderar jag spår eller klipp i Ableton Live?",
            "ideal_answer": "Markera önskade klipp eller ett tidsintervall över flera spår, gå sedan till menyn Redigera och välj 'Consolidate Time to New Track' eller 'Consolidate' (Cmd/Ctrl+J).",
        },
        {
            "question": "Vad är Scener i Session View och hur används de?",
            "ideal_answer": "Scener i Session View är horisontella rader som innehåller en samling klipp, typiskt representerande en sektion av en låt. Att starta en scen spelar alla klipp inom den raden samtidigt, vilket är användbart för liveframträdanden och improvisation.",
        },
        {
            "question": "Hur kan jag minska CPU-användningen i Ableton Live när mitt projekt är komplext?",
            "ideal_answer": "För att minska CPU-användningen kan du frysa spår, 'flattena' spår, minska buffertstorleken, inaktivera oanvända enheter eller använda färre CPU-intensiva effekter.",
        },
        {
            "question": "Beskriv funktionen 'Follow Actions' för MIDI- och ljudklipp.",
            "ideal_answer": "Follow Actions låter dig definiera vad som händer efter att ett klipp spelats klart, till exempel att spela ett annat klipp, stoppa, återstarta sig själv, eller starta en annan scen. Detta är användbart för att skapa dynamiska arrangemang och generativ musik.",
        },
        {
            "question": "Hur ställer jag in en extern MIDI-kontroller i Ableton Live 12?",
            "ideal_answer": "Gå till Lives inställningar, sedan 'Link/Tempo/MIDI'. Välj din kontroller från rullgardinsmenyn 'Control Surface', aktivera dess 'Track' och 'Remote' omkopplare i sektionen 'MIDI Ports', och se till att dess MIDI-ingång är aktiv.",
        },
    ]

    predefined_qa = predefined_qa_en if answer_language == "English" else predefined_qa_sv

    st.markdown("### Pick a question:")
    question_idx = st.selectbox(
        "Choose a question:",
        options=list(range(len(predefined_qa))),
        format_func=lambda x: predefined_qa[x]["question"],
    )
    question = predefined_qa[question_idx]["question"]
    ideal_answer = predefined_qa[question_idx]["ideal_answer"]

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            no_answer_phrase = NO_ANSWER_EN if answer_language == "English" else NO_ANSWER_SV

            query_emb = create_embeddings([question])[0]
            results = vector_store.search(query_emb, k=EVALUATION_TOP_K)
            top_texts = [r["text"] for r in results]
            model_answer = generate_response(question, top_texts, answer_language=answer_language)

            if model_answer.strip() == no_answer_phrase.strip():
                score = 0.00
                no_answer = True
            else:
                model_emb = create_embeddings([model_answer])[0]
                ideal_emb = create_embeddings([ideal_answer])[0]
                similarity = dot(model_emb, ideal_emb) / (norm(model_emb) * norm(ideal_emb))
                score = round(float(similarity), 2)
                no_answer = False

            st.session_state.eval_result = {
                "question": question,
                "model_answer": model_answer,
                "ideal_answer": ideal_answer,
                "score": score,
                "no_answer": no_answer,
            }

            if "eval_scores" not in st.session_state:
                st.session_state.eval_scores = []
            if "scored_ids" not in st.session_state:
                st.session_state.scored_ids = set()

            score_id = (question.strip(), model_answer.strip())
            if score_id not in st.session_state.scored_ids:
                st.session_state.eval_scores.append(score)
                st.session_state.scored_ids.add(score_id)

    if "eval_result" in st.session_state:
        result = st.session_state.eval_result
        st.markdown("### RAG-Bot's answer:")
        st.write(result["model_answer"])
        st.markdown("### Ideal answer:")
        st.write(result["ideal_answer"])
        label = " (bot said it didn't know)" if result["no_answer"] else ""
        st.markdown(f"### Similarity Score: `{result['score']}`{label}")

    if "eval_scores" in st.session_state:
        valid = [s for s in st.session_state.eval_scores if s is not None]
        if valid:
            avg = sum(valid) / len(valid)
            st.markdown(
                f"### Session Average: `{avg:.2f}` (from {len(valid)} evaluations)"
            )

    if st.button("Reset Session Scores"):
        st.session_state.eval_scores = []
        st.session_state.scored_ids = set()
        st.session_state.pop("eval_result", None)
        st.rerun()
