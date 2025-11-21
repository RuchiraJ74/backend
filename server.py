from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
import cloudinary
import cloudinary.uploader
import traceback
import tempfile
import whisper
import shutil
import subprocess
import datetime
import torch
import sqlite3
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from transformers import pipeline
from deep_translator import GoogleTranslator

# ======================= #
# 1️⃣ INITIAL SETUP
# ======================= #
load_dotenv()
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
SUMMARY_FOLDER = "summaries"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)
DB_PATH = os.path.join(os.path.dirname(__file__), "summaries.db")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Device:", device)

# ======================= #
# 2️⃣ LOAD MODELS
# ======================= #
print("⏳ Loading models...")
asr_model = whisper.load_model("medium")   # multilingual Whisper model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("✅ Models loaded successfully!")

# ======================= #
# 3️⃣ DATABASE INIT
# ======================= #
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT NOT NULL,
            transcription TEXT NOT NULL,
            key_points TEXT NOT NULL,
            key_decisions TEXT NOT NULL,
            task_assignments TEXT NOT NULL,
            pdf_filename TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
init_db()

# ======================= #
# 4️⃣ HELPERS
# ======================= #
def extract_audio(video_path):
    audio_path = tempfile.mktemp(suffix=".mp3")
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", audio_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path

def safe_summarize(text, max_length=180, min_length=60):
    words = text.split()
    if len(words) < 20:
        return text
    try:
        return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"⚠️ Summarization failed: {e}"

def chunk_text(text, max_tokens=600):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def structured_summary(text):
    abstract = safe_summarize("Abstract of meeting: " + text, max_length=200, min_length=60)
    key_points = safe_summarize("Key points: " + text, max_length=150, min_length=50)
    key_decisions = safe_summarize("Decisions made: " + text, max_length=100, min_length=30)
    task_assignments = safe_summarize("Tasks assigned: " + text, max_length=100, min_length=30)
    return {
        "Abstract": abstract,
        "Key Points": key_points,
        "Key Decisions": key_decisions,
        "Task Assignments": task_assignments
    }

def translate_summary(summary_dict, lang):
    translated = {}
    for k, v in summary_dict.items():
        translated[k] = GoogleTranslator(source='en', target=lang).translate(v)
    return translated

def create_pdf(transcription, summary_dict):
    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']
    h2_style = styles['Heading2']

    story = []
    story.append(Paragraph("Meeting Transcription", title_style))
    story.append(Spacer(1, 12))
    for line in transcription.split("\n"):
        story.append(Paragraph(line, normal_style))
    story.append(PageBreak())

    story.append(Paragraph("Meeting Summary", title_style))
    story.append(Spacer(1, 12))
    for section, content in summary_dict.items():
        story.append(Paragraph(f"<b>{section}</b>", h2_style))
        story.append(Paragraph(content, normal_style))
        story.append(Spacer(1, 10))

    doc.build(story)
    return pdf_path

# ======================= #
# 5️⃣ MAIN FUNCTION
# ======================= #
def summarize_meeting(file_path, target_lang="en"):
    if file_path.lower().endswith((".mp4", ".mkv", ".mov", ".avi")):
        file_path = extract_audio(file_path)

    print("🔍 Transcribing...")
    result = asr_model.transcribe(file_path)
    transcription = "\n".join([f"Speaker {i%2+1}: {seg['text'].strip()}" for i, seg in enumerate(result["segments"])])

    print("✂️ Summarizing transcript...")
    chunks = chunk_text(transcription)
    summaries = [safe_summarize(chunk) for chunk in chunks]
    final_text = " ".join(summaries)
    summary_dict = structured_summary(final_text)

    # 🌍 Translation step
    supported_langs = {
        "en": "English", "hi": "Hindi", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
        "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
        "pa": "Punjabi", "or": "Odia", "ur": "Urdu",
        "fr": "French", "de": "German", "es": "Spanish", "zh": "Chinese", "ru": "Russian"
    }
    if target_lang in supported_langs and target_lang != "en":
        print(f"🌍 Translating to {supported_langs[target_lang]}...")
        try:
            summary_dict = translate_summary(summary_dict, target_lang)
            transcription = GoogleTranslator(source='en', target=target_lang).translate(transcription)
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            # Fallback: continue with English if translation fails

    pdf_filename = f"meeting_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(SUMMARY_FOLDER, pdf_filename)
    temp_pdf = create_pdf(transcription, summary_dict)
    shutil.copy(temp_pdf, pdf_path)
    return transcription, summary_dict, pdf_filename

# ======================= #
# 6️⃣ FLASK ROUTES
# ======================= #
@app.route('/upload', methods=['POST'])
def upload_recording():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    lang = request.form.get("lang", "en")
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    try:
        transcription, summary_dict, pdf_filename = summarize_meeting(save_path, lang)
        return jsonify({
            "status": "success",
            "transcription": transcription,
            "summary": summary_dict,
            "pdf_url": f"/download/{pdf_filename}"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download_summary(filename):
    path = os.path.join(SUMMARY_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)

# ======================= #
# 7️⃣ CLOUDINARY UPLOAD
# ======================= #
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("CLOUD_API_KEY"),
    api_secret=os.getenv("CLOUD_API_SECRET")
)

@app.route('/upload-to-cloud', methods=['POST'])
def upload_to_cloud():
    filename = request.form.get("filename")
    pdf_path = os.path.join(SUMMARY_FOLDER, filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": "File not found"}), 404
    res = cloudinary.uploader.upload(pdf_path, resource_type="raw")
    return jsonify({"status": "success", "url": res["secure_url"]})

# ======================= #
# 8️⃣ RUN SERVER
# ======================= #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
