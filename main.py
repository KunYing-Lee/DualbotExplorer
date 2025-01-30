import streamlit as st
from streamlit_chat import message
from debate import DualDebateBots
from paper_digest import JournalistBot, AuthorBot
from peer_review import PeerReviewAuthorBot, PeerReviewReviewerBot
import time
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

# Define session settings
SESSION_LENGTHS = ['Short', 'Long']
MAX_EXCHANGE_COUNTS = {
    'Short': {'Debate': 4},
    'Long': {'Debate': 8}
}
AVATAR_SEED = [123, 42]
MODES = {
    "Debate": {"icon": "🗣️", "color": "#FF6B6B"},
    "Paper Digest": {"icon": "📄", "color": "#4ECDC4"},
    "Peer-review Simulation": {"icon": "🔍", "color": "#FF9F43"}
}

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .header-text {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #2d3436 !important;
        margin-bottom: 1rem !important;
    }
    .mode-card {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        background: white;
        margin: 1rem 0;
    }
    .mode-card:hover {
        transform: translateY(-5px);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .disabled-widget {
        opacity: 0.6;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown('<p class="header-text">DualBot Explorer 🤖💬</p>', unsafe_allow_html=True)

# Set the description of the app
with st.expander("🌟 Welcome to DualBot Explorer!", expanded=True):
    cols = st.columns(3)
    for i, (mode_name, config) in enumerate(MODES.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="mode-card" style="border-left: 5px solid {config['color']}">
                <h3>{config['icon']} {mode_name}</h3>
                <hr style="margin: 0.5rem 0; border-color: {config['color']};">
                {"Debate complex topics with AI opponents" if mode_name == "Debate" else 
                 "Analyze research papers through simulated interviews" if mode_name == "Paper Digest" else 
                 "Experience academic peer-review process"}
            </div>
            """, unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input('OpenAI API Key 🔑', type='password')
    mode = st.selectbox('Select Mode 📖', list(MODES.keys()), 
                       format_func=lambda x: f"{MODES[x]['icon']} {x}")

# Initialize session states
if "bot1_mesg" not in st.session_state:
    st.session_state["bot1_mesg"] = []
if "bot2_mesg" not in st.session_state:
    st.session_state["bot2_mesg"] = []
if 'message_counter' not in st.session_state:
    st.session_state["message_counter"] = 0
# Common conversation container
conversation_container = st.container()

# --- Debate Mode ---
if mode == 'Debate':
    scenario = st.sidebar.text_input('Debate Topic 💬')
    role_dict = {
        'role1': {'name': 'Proponent'},
        'role2': {'name': 'Opponent'}
    }
    time_delay = 5
    session_length = st.sidebar.selectbox('Session Length ⏰', SESSION_LENGTHS)

    if st.sidebar.button('Generate Debate'):
        missing = []
        if not api_key: missing.append("OpenAI API Key")
        if not scenario: missing.append("Debate Topic")
        if missing:
            st.sidebar.error(f"Missing required fields: {', '.join(missing)}")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            with conversation_container:
                st.write(f"""#### Debate 💬: {scenario}""")
                with st.spinner("Setting up debate arena..."):
                    DualDebateBots = DualDebateBots('OpenAI', role_dict, scenario, session_length)
                    st.session_state['DualDebateBots'] = DualDebateBots
                
                for _ in range(MAX_EXCHANGE_COUNTS[session_length]['Debate']):
                    with st.spinner("Bots are debating..."):
                        output1, output2 = DualDebateBots.step()
                    
                    mesg_1 = {"role": DualDebateBots.chatbots['role1']['name'], "content": output1}
                    mesg_2 = {"role": DualDebateBots.chatbots['role2']['name'], "content": output2}
                    
                    new_count = message(f"{mesg_1['content']}", is_user=False, 
                                    avatar_style="bottts", seed=AVATAR_SEED[0],
                                    key=st.session_state["message_counter"])
                    st.session_state["message_counter"] += 1
                    time.sleep(time_delay)
                    new_count = message(f"{mesg_2['content']}", is_user=True,
                                    avatar_style="bottts", seed=AVATAR_SEED[1],
                                    key=st.session_state["message_counter"])
                    st.session_state["message_counter"] += 1
                    st.session_state.bot1_mesg.append(mesg_1)
                    st.session_state.bot2_mesg.append(mesg_2)

    if 'DualDebateBots' in st.session_state:
        with st.expander('Debate Summary'):
            scripts = [f"Pro: {m1['content']}\nCon: {m2['content']}" 
                    for m1,m2 in zip(st.session_state.bot1_mesg, st.session_state.bot2_mesg)]
            st.write(st.session_state['DualDebateBots'].summary(scripts))

# --- Paper Digest Mode ---
elif mode == 'Paper Digest':
    topic = st.sidebar.text_input('Paper Topic 🧪')
    abstract = st.sidebar.text_area('Abstract 📝')
    uploaded_file = st.sidebar.file_uploader("Upload PDF paper 📄", type="pdf")

    if st.sidebar.button('Generate Interview'):
        missing = []
        if not api_key: missing.append("OpenAI API Key")
        if not topic: missing.append("Paper Topic")
        if not abstract.strip(): missing.append("Abstract")
        if not uploaded_file: missing.append("PDF upload")
        
        if missing:
            st.sidebar.error(f"Missing required fields: {', '.join(missing)}")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            with conversation_container:
                st.write(f"#### Paper Digest: {topic}")
                with st.spinner("Initializing bots..."):
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader("temp.pdf")
                    pages = loader.load_and_split()
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_documents(pages, embeddings)
                    
                    journalist = JournalistBot('OpenAI')
                    journalist.instruct(topic, abstract)
                    author = AuthorBot('OpenAI', vectorstore)
                    author.instruct(topic)
                    
                    st.session_state.interview_history = []
                    question = journalist.step("")
                
                for _ in range(6):
                    with st.spinner("Journalist is thinking..."):
                        st.session_state.interview_history.append(("Q", question))
                        message(question, key=f"q_{_}", avatar_style="bottts", seed=AVATAR_SEED[0])
                    
                    with st.spinner("Author is responding..."):
                        answer, _ = author.step(question)
                        st.session_state.interview_history.append(("A", answer))
                        message(answer, is_user=True, key=f"a_{_}", avatar_style="bottts", seed=AVATAR_SEED[1])
                    
                    with st.spinner("Generating next question..."):
                        question = journalist.step(answer)
                    
                    time.sleep(1)
                
                with st.spinner("Generating summary..."):
                    st.session_state.interview_summary = author.summary(
                        "\n".join([f"{t}: {c}" for t,c in st.session_state.interview_history])
                    )

    if 'interview_history' in st.session_state:
        with st.expander("Interview Summary"):
            st.markdown("**Key Insights:**")
            st.write(st.session_state.interview_summary)

# --- Peer-review Simulation Mode ---
elif mode == 'Peer-review Simulation':
    st.sidebar.markdown("### Paper Submission")
    paper_title = st.sidebar.text_input('Paper Title 📝')
    paper_abstract = st.sidebar.text_area('Abstract 🔬', height=150)
    uploaded_paper = st.sidebar.file_uploader("Upload Submission PDF 📄", type="pdf")
    review_focus = st.sidebar.multiselect('Review Focus Areas', 
        ['Methodology', 'Results', 'Ethics', 'Originality', 'Reproducibility'])
    rigor_level = st.sidebar.slider('Review Rigor Level', 0, 100, 50)

    if st.sidebar.button('Start Review Process'):
        missing = []
        if not api_key: missing.append("OpenAI API Key")
        if not paper_title: missing.append("Paper Title")
        if not paper_abstract.strip(): missing.append("Abstract")
        if not uploaded_paper: missing.append("PDF upload")
        if not review_focus: missing.append("Review Focus Areas")
        
        if missing:
            st.sidebar.error(f"Missing required fields: {', '.join(missing)}")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            with conversation_container:
                st.write(f"#### Peer Review: {paper_title}")
                with st.spinner("Initializing review process..."):
                    with open("review_temp.pdf", "wb") as f:
                        f.write(uploaded_paper.getbuffer())
                    
                    loader = PyPDFLoader("review_temp.pdf")
                    pages = loader.load_and_split()
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_documents(pages, embeddings)
                    
                    author_bot = PeerReviewAuthorBot('OpenAI')
                    author_bot.instruct(paper_title, paper_abstract, vectorstore)
                    
                    reviewer_bot = PeerReviewReviewerBot('OpenAI')
                    reviewer_bot.instruct(paper_title, paper_abstract, review_focus, rigor_level)
                    
                    st.session_state.review_history = []
                
                for i in range(4):
                    with st.spinner("Reviewer is formulating question..."):
                        question = reviewer_bot.generate_question()
                        st.session_state.review_history.append(("Reviewer", question))
                        message(question, is_user=False, avatar_style="bottts", 
                               seed=AVATAR_SEED[0], key=f"rev_q_{i}")
                    
                    with st.spinner("Author is preparing response..."):
                        answer = author_bot.respond_to_question(question)
                        st.session_state.review_history.append(("Author", answer))
                        message(answer, is_user=True, avatar_style="bottts",
                               seed=AVATAR_SEED[1], key=f"rev_a_{i}")
                    
                    time.sleep(1)
                
                with st.spinner("Generating final verdict..."):
                    verdict = reviewer_bot.generate_verdict()
                    st.session_state.review_history.append(("Verdict", verdict))
                    st.session_state.review_summary = {
                        "critiques": reviewer_bot.summarize_critiques(),
                        "responses": author_bot.summarize_responses()
                    }
                
                st.markdown(f"**Final Verdict:**\n\n{verdict}")

    if 'review_history' in st.session_state:
        with st.expander("Review Summary"):
            st.markdown("**Key Critique Points:**")
            st.write(st.session_state.review_summary['critiques'])
            st.markdown("**Author Responses:**")
            st.write(st.session_state.review_summary['responses'])

