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
)

if __name__ == "__main__":
    app, local_url, share_url = iface.launch(enable_queue=True)
