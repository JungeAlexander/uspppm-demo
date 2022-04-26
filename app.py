import gradio as gr
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")


def get_sim(anchor: str, target: str) -> float:
    anchor_embed = model.encode([anchor])
    target_embed = model.encode([target])
    return float(1 - cosine(anchor_embed, target_embed))


anchor_input = gr.inputs.Textbox(lines=1, placeholder="Anchor")
target_input = gr.inputs.Textbox(lines=1, placeholder="Target")

sim_output = gr.outputs.Textbox(type="number", label="Similarity")

examples = [
    ["renewable power", "renewable energy"],
    ["previously captured image", "image captured previously"],
    ["labeled ligand", "container labelling"],
    ["gold alloy", "platinum"],
    ["dissolve in glycol", "family gathering"],
]

iface = gr.Interface(
    fn=get_sim,
    inputs=[anchor_input, target_input],
    outputs=sim_output,
    examples=examples,
    theme="grass",
    title="Demo: U.S. Patent Phrase to Phrase Matching",
    description="Scores phrases from U.S. patents according to their similarity. "
    "Similarity scores are between 0 and 1, higher scores mean higher similarrity, and scores "
    "are computed as the cosine similarity of embeddings produced by the AI-Growth-Lab/PatentSBERTa SentenceTransformer model.",
    article="Examples are taken from the *Google Patent Phrase Similarity Dataset* used in the "
    "['U.S. Patent Phrase to Phrase Matching' Kaggle competition](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview). "
    "The code for this app his available on [GitHub](https://github.com/JungeAlexander/uspppm-demo).",
)

if __name__ == "__main__":
    app, local_url, share_url = iface.launch(enable_queue=True)
