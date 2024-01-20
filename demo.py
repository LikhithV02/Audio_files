from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import torch
import torchaudio
import os

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
print("XTTS downloaded")

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

prompt = "मैं आपका आभासी सहायक हूं जिसे इस वेबसाइट पर आपके लेनदेन से संबंधित विशिष्ट जानकारी में आपकी सहायता करने के लिए डिज़ाइन किया गया है। यदि आपके कोई प्रश्न हों या विशिष्ट जानकारी की आवश्यकता हो, तो कृपया मुझे बताएं, और मैं तुरंत आपकी सहायता करूंगा"
language = "hi"
speaker_wav = "LJ001-0001.wav"
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)

out = model.inference(
                prompt,
                language,
                gpt_cond_latent,
                speaker_embedding,
                repetition_penalty=5.0,
                temperature=0.75,
            )
torchaudio.save("output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)