import os
import re
import tempfile
import uuid
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
from flask import Flask, jsonify, request, send_file
from num2words import num2words
from pydub import AudioSegment
from faster_whisper import WhisperModel
import torch
import torchaudio
import whisper

# Flask app
app = Flask(__name__)

# Load TTS model
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
# print("XTTS downloaded")

config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
    eval=True,
)
model.cuda()

# TTS file prefix
speech_tts_prefix = "speech-tts-"
wav_suffix = ".wav"
opus_suffix = ".opus"

speaker_wav = "LJ001-0001.wav"
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
language = "hi"

# Load transcription model
model_size = "large-v3"

# Run on GPU with FP16
transcribe_model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Clean temporary files (called every 5 minutes)
def clean_tmp():
    tmp_dir = tempfile.gettempdir()
    for file in os.listdir(tmp_dir):
        if file.startswith(speech_tts_prefix):
            os.remove(os.path.join(tmp_dir, file))
    print("[Speech REST API] Temporary files cleaned!")

# Preprocess text to replace numerals with words
def preprocess_text(text):
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    return text

# Run TTS and save file
# Returns the path to the file
def run_tts_and_save_file(text):
    # Running the TTS
    out = model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                repetition_penalty=5.0,
                temperature=0.75,
            )

    # Get temporary directory
    tmp_dir = tempfile.gettempdir()

    # Save wav to temporary file
    tmp_path_wav = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + wav_suffix)
    torchaudio.save(tmp_path_wav, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    return tmp_path_wav

# TTS endpoint
@app.route('/tts', methods=['POST'])
def generate_tts():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Invalid input: text missing'}), 400

    # Sentences to generate
    text = request.json['text']

    # Remove ' and " and  from text
    text = text.replace("'", "")
    text = text.replace('"', "")

    # Preprocess text to replace numerals with words
    text = preprocess_text(text)

    # Split text by . ? !
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

    # Trim sentences
    sentences = [sentence.strip() for sentence in sentences]

    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence]

    # Logging
    print("[Speech REST API] Got request: length (" + str(len(text)) + "), sentences (" + str(len(sentences)) + ")")

    # Run TTS for each sentence
    output_files = []

    for sentence in sentences:
        print("[Speech REST API] Generating TTS: " + sentence)
        tmp_path_wav = run_tts_and_save_file(sentence)
        output_files.append(tmp_path_wav)

    # Concatenate all files
    audio = AudioSegment.empty()

    for file in output_files:
        audio += AudioSegment.from_wav(file)

    # Save audio to file
    tmp_dir = tempfile.gettempdir()
    tmp_path_opus = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + opus_suffix)
    audio.export(tmp_path_opus, format="opus")

    # Delete tmp files
    for file in output_files:
        os.remove(file)

    # Send file response
    return send_file(tmp_path_opus, mimetype='audio/ogg, codecs=opus')

# Transcribe endpoint
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'Invalid input, form-data: audio'}), 400

    # Audio file
    audio_file = request.files['audio']

    # Save audio file into tmp folder
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, str(uuid.uuid4()))
    audio_file.save(tmp_path)

    # Transcribe
    segments, info = list(model.transcribe(audio_file, beam_size=1))

    # Detect the spoken language
    language = info.language

    # Result
    segments = list(segments)
    transcription = " ".join([segment.text.lstrip() for segment in segments])

    # Delete tmp file
    os.remove(tmp_path)

    return jsonify({
        'language': language,
        'text': transcription
    }), 200

# Health endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/clean', methods=['GET'])
def clean():
    clean_tmp()
    return jsonify({'status': 'ok'}), 200

# Entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))

    # Start server
    print("[Speech REST API] Starting server on port " + str(port))

    app.run(host='0.0.0.0', port=3000)