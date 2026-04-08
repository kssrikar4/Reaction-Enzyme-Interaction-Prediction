import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from transformers import EsmTokenizer, EsmModel
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.spatial_proj = nn.Embedding(100, n_heads)
        
    def forward(self, x, spatial_pos, attn_mask=None):
        if attn_mask is not None:
            key_padding_mask = (attn_mask == 0)
        else:
            key_padding_mask = None
        
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class GraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.d_model = d_model
        
    def forward(self, node_features, spatial_pos, attn_mask=None):
        x = self.input_proj(node_features)
        for layer in self.layers:
            x = layer(x, spatial_pos, attn_mask)
        return x

class MultimodalReactionEnzymeModel(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t33_150M_UR50D", graph_input_dim=24, graph_d_model=256,
                 graph_n_layers=4, graph_n_heads=8, cross_attn_heads=8, dropout=0.1, freeze_esm=True):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        esm_hidden_size = self.esm.config.hidden_size
        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False
        self.graph_encoder = GraphTransformerEncoder(input_dim=graph_input_dim, d_model=graph_d_model,
                                                      n_layers=graph_n_layers, n_heads=graph_n_heads, dropout=dropout)
        self.esm_proj = nn.Linear(esm_hidden_size, graph_d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=graph_d_model, num_heads=cross_attn_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(graph_d_model)
        self.rc_head = nn.Sequential(
            nn.Linear(graph_d_model, graph_d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_d_model // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask, node_features, spatial_pos, graph_attention_mask=None):
        esm_outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        enzyme_repr = esm_outputs.last_hidden_state
        enzyme_repr = self.esm_proj(enzyme_repr)
        atom_repr = self.graph_encoder(node_features, spatial_pos, graph_attention_mask)
        enzyme_key_mask = (attention_mask == 0)
        fused_repr, _ = self.cross_attention(query=atom_repr, key=enzyme_repr, value=enzyme_repr, key_padding_mask=enzyme_key_mask)
        fused_repr = self.cross_norm(atom_repr + fused_repr)
        rc_logits = self.rc_head(fused_repr).squeeze(-1)
        if graph_attention_mask is not None:
            rc_logits = rc_logits.masked_fill(graph_attention_mask == 0, float('-inf'))
        return rc_logits, torch.sigmoid(rc_logits)

# --- Data Processing Utils ---

ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'other']
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, 
                  Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED, 
                  Chem.rdchem.HybridizationType.OTHER]

def get_atom_features(atom):
    features = []
    atom_type = atom.GetSymbol()
    type_onehot = [1 if atom_type == t else 0 for t in ATOM_TYPES]
    features.extend(type_onehot)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    hybrid = atom.GetHybridization()
    hybrid_onehot = [1 if hybrid == h else 0 for h in HYBRIDIZATIONS]
    features.extend(hybrid_onehot)
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(atom.GetTotalNumHs())
    features.append(1 if atom.IsInRing() else 0)
    return features

def mol_to_graph_data(mol):
    if mol is None: return None
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0: return None
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    node_features = torch.tensor(node_features, dtype=torch.float)
    spatial_pos = torch.full((num_atoms, num_atoms), 100, dtype=torch.long)
    for i in range(num_atoms): spatial_pos[i, i] = 0
    for start in range(num_atoms):
        visited = {start: 0}; queue = [start]
        while queue:
            node = queue.pop(0)
            atom = mol.GetAtomWithIdx(node)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx not in visited:
                    visited[n_idx] = visited[node] + 1
                    queue.append(n_idx)
        for node, dist in visited.items(): spatial_pos[start, node] = dist
    in_degree = torch.tensor([atom.GetDegree() for atom in mol.GetAtoms()], dtype=torch.long)
    out_degree = in_degree.clone()
    return {'node_features': node_features, 'spatial_pos': spatial_pos, 'in_degree': in_degree, 'out_degree': out_degree, 'num_atoms': num_atoms}

def collate_reaction_enzyme(batch):
    batch_size = len(batch)
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    max_atoms = max(item['graph_data']['num_atoms'] for item in batch)
    node_features = torch.zeros(batch_size, max_atoms, batch[0]['graph_data']['node_features'].shape[1])
    spatial_pos = torch.full((batch_size, max_atoms, max_atoms), 100, dtype=torch.long)
    graph_attention_mask = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    for i, item in enumerate(batch):
        g = item['graph_data']; n = g['num_atoms']
        node_features[i, :n] = g['node_features']
        spatial_pos[i, :n, :n] = g['spatial_pos']
        graph_attention_mask[i, :n] = 1
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'node_features': node_features, 
            'spatial_pos': spatial_pos, 'graph_attention_mask': graph_attention_mask, 'batch_size': batch_size}

st.set_page_config(page_title="Reaction Center Predictor", layout="wide")
st.title("🧪 Multimodal Reaction-Enzyme Predictor")
st.markdown("Predict atom-level reaction centers given a reaction SMILES and an enzyme sequence.")

@st.cache_resource
def load_model():
    repo_id = "kssrikar4/Reaction-Enzyme-Interaction-Prediction-Model"
    model_path = hf_hub_download(repo_id=repo_id, filename="reaction_enzyme_model.pt")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['model_config']
    config['graph_input_dim'] = 24
    config['esm_model_name'] = "facebook/esm2_t30_150M_UR50D"
    model = MultimodalReactionEnzymeModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    tokenizer = EsmTokenizer.from_pretrained(config['esm_model_name'])
    return model, tokenizer

model, tokenizer = load_model()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5)
    if st.button("Load Sample Data"):
        st.session_state.reaction = "C(CC(=O)[O-])C(=O)C(=O)[O-]"
        st.session_state.sequence = "MYRYLAKALLPSRAGPAALGSAANHSAALLGRGRGQPAAASQPGLALAARRHYSELVADREDDPNFFKMVEGFFDRGASIVEDKLVKDLRTQESEEQKRNRVRGILRIIKPCNHVLSLSFPIRRDDGSWEVIEGYRAQHSQHRTPCKGGIRYSTDVSVDEVKALASLMTYKCAVVDVPFGGAKAGVKINPKNYTENELEKITRRFTMELAKKGFIGPGVDVPAPDMNTGEREMSWIADTYASTIGHYDINAHACVTGKPISQGGIHGRISATGRGVFHGIENFINQASYMSILGMTPGFRDKTFVVQGFGNVGLHSMRYLHRFGAKCIAVGESDGSIWNPDGIDPKELEDFKLQHGSILGFPKAKPYEGSILEVDCDILIPAATEKQLTKSNAPRVKAKIIAEGANGPTTPEADKIFLERNILVIPDLYLNAGGVTVSYFEWLKNLNHVSYGRLTFKYERDSNYHLLLSVQESLERKFGKHGGTIPIVPTAEFQDSISGASEKDIVHSALAYTMERSARQIMHTAMKYNLGLDLRTAAYVNAIEKVFKVYSEAGVTFT"

col1, col2 = st.columns(2)

with col1:
    reaction_input = st.text_area("Reaction SMILES", value=st.session_state.get('reaction', ""), height=100)
    sequence_input = st.text_area("Enzyme Sequence", value=st.session_state.get('sequence', ""), height=200)

if st.button("Predict Reaction Centers"):
    if not reaction_input or not sequence_input:
        st.error("Please provide both reaction SMILES and enzyme sequence.")
    else:
        try:
            reactants = reaction_input.split('>>')[0]
            mol = Chem.MolFromSmiles(reactants)
            if mol is None:
                st.error("Invalid reaction SMILES (could not parse reactants).")
            else:
                graph_data = mol_to_graph_data(mol)
                seq_encoding = tokenizer(sequence_input, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
                batch = collate_reaction_enzyme([{
                    'input_ids': seq_encoding['input_ids'].squeeze(0),
                    'attention_mask': seq_encoding['attention_mask'].squeeze(0),
                    'graph_data': graph_data
                }])
                
                with torch.no_grad():
                    logits, probs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        node_features=batch['node_features'],
                        spatial_pos=batch['spatial_pos'],
                        graph_attention_mask=batch['graph_attention_mask']
                    )
                
                probs = probs.squeeze(0).numpy()
                
                with col2:
                    st.subheader("Results")
                    # Highlight atoms in RDKit
                    highlight_atoms = [int(i) for i in np.where(probs > threshold)[0]]
                    
                    # Add atom indices as notes to appear in the image
                    for atom in mol.GetAtoms():
                        atom.SetProp('atomNote', str(atom.GetIdx()))
                    
                    img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, highlightColor=(1, 0.5, 0.5), size=(600, 600))
                    st.image(img, caption="Predicted Reaction Centers (Highlighted with Atom Indices)")
                    
                    # Top predictions table
                    top_indices = np.argsort(probs)[::-1][:10]
                    results_df = pd.DataFrame({
                        'Atom Index': top_indices,
                        'Symbol': [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in top_indices],
                        'Probability': probs[top_indices]
                    })
                    st.table(results_df)
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
