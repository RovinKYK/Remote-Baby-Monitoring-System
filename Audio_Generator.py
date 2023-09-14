from diffusers import AudioLDMPipeline
import torch
import scipy

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

for i in range(5, 51, 5):
    prompt = "Clear natural clip of baby crying in hunger no background noise"
    audio = pipe(prompt, num_inference_steps=i, audio_length_in_s=7.0).audios[0]

    scipy.io.wavfile.write("techno"+str(i)+".wav", rate=16000, data=audio)

# best results; num_inference_steps = 35
