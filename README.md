# Reaction-Enzyme Interaction Prediction Model

A deep learning framework for predicting reaction center atoms in chemical reactions using multimodal learning that combines molecular graph representations with enzyme sequence embeddings.

## Overview

This project implements a multimodal neural network that predicts which atoms in a reactant molecule will participate in a reaction (reaction center prediction) by using:
- **Molecular Graphs**: Graph transformer encoders for atom-level representations
- **Enzyme Sequences**: ESM-2 protein language model embeddings
- **Cross-Attention Mechanism**: Fuses molecular and enzyme information

## Architecture

The model consists of three main components:

1. **ESM-2 Encoder**: Processes enzyme amino acid sequences using META FAIR's ESM-2 protein language model
2. **Graph Transformer Encoder**: Encodes molecular structures using graph attention layers
3. **Cross-Attention Fusion**: Combines enzyme and molecular representations to predict reaction centers

```Map
┌─────────────────┐     ┌─────────────────────┐
│   ESM-2 Model   │     │  Graph Transformer  │
│  (Protein Seq)  │     │   (Molecular Graph) │
└────────┬────────┘     └─────────┬───────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐     ┌──────────────────┐
│  ESM Projection │     │  Atom Features   │
└────────┬────────┘     └─────────┬────────┘
         │                        │
         └──────────┬─────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   Cross-Attention   │
         │      Mechanism      │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Reaction Center    │
         │   Prediction Head   │
         └─────────────────────┘
```

## Features

- **Multimodal Learning**: Combines protein sequence and molecular graph data
- **Reaction Center Prediction**: Identifies atoms involved in enzymatic reactions
- **Pre-trained Models**: Uses ESM-2 for protein embeddings
- **Graph Transformers**: Custom graph attention layers for molecular encoding
- **Interactive GUI**: Streamlit-based web interface for predictions

## Training History
<img width="2234" height="1481" alt="training_history" src="https://github.com/user-attachments/assets/799f3545-1b44-4f89-bc26-3caff28e3f02" />

### Usage

```bash
# Clone the repository
git clone https://github.com/kssrikar4/Reaction-Enzyme-Interaction-Prediction
cd Reaction-Enzyme-Interaction-Prediction

# Install dependencies
pip install torch transformers rdkit streamlit pandas numpy matplotlib
```
Launch the Streamlit application for interactive predictions:

```bash
streamlit run app.py
```

The GUI allows you to:
- Input SMILES strings for molecules
- Input enzyme amino acid sequences
- Visualize predicted reaction centers
- View attention weights and model interpretations

## Data Sources

The model is trained on data from:

- **[Rhea](https://www.rhea-db.org/help/download)**: Expert-curated database of chemical reactions
- **[RetroRules](https://retrorules.org/download)**: Reaction rules for retrosynthesis
- **[UniProt](https://www.uniprot.org/uniprotkb?facets=model_organism%3A9606&groupBy=ec&query=Human)**: Protein sequences and EC numbers

## File Structure

```
.
├── train.ipynb                # Jupyter notebook for training/experimentation
├── app.py                     # Streamlit GUI application
├── model_config.json          # Model hyperparameters
├── training_history.csv       # Training metrics
│(Must be manually downloaded)
├── rhea-reaction-smiles.tsv   # Rhea reaction data
├── rhea-chebi-smiles.tsv      # Chemical compound data
├── rhea-directions.tsv        # Reaction directions
├── rhea-relationships.tsv     # Reaction relationships
├── rhea2ec.tsv               # EC number mappings
└── rhea2uniprot_sprot.tsv    # UniProt mappings
├── Rhea-derived dataset.csv.gz          # Processed Rhea dataset
├── MetaNetX-derived dataset.csv.gz      # MetaNetX metabolic network data
├── USPTO-derived dataset.csv.gz         # USPTO reaction patent data
├── uniprotkb_Human_AND_model_organism_9606_2026_04_07.tsv.gz   # UniProt human enzyme data
├── uniprotkb_Human_AND_model_organism_9606_2026_04_07.fasta.gz # UniProt human enzyme sequences
```

## License

This project is licensed under [Apache License 2.0](LICENSE) - Feel free to use and modify

## Acknowledgments

- [ESM-2](https://github.com/facebookresearch/esm) - Evolutionary Scale Modeling
- [Rhea](https://www.rhea-db.org/) - Reaction database
- [RetroRules](https://retrorules.org/) - Reaction rules database
- [RDKit](https://www.rdkit.org/) - Cheminformatics toolkit
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Model implementations
- [PyTorch](https://pytorch.org) - Deep learning framework
- [Kimi AI with K2.5](https://www.kimi.com) - Assistance in generating code.
