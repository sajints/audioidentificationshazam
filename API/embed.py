from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def get_embedding(audio):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.squeeze().numpy()  # Shape: (768,)