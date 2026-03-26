#!/bin/bash
# Generate professional Mermaid architecture diagrams
OUT="/home/user/D/CIDL_project1/outputs/figures"
cd /tmp

# 1. Simple LSTM Architecture
cat > simple_lstm.mmd << 'EOF'
graph TD
    A["<b>Input Sequence</b><br/>(batch, seq_len, 20 features)"] --> B
    B["<b>LSTM Layer</b><br/>hidden=128, 1 layer"] --> C
    C["<b>Last Hidden State</b><br/>(batch, 128)"] --> D
    D["<b>Fully Connected</b><br/>128 → 1"] --> E
    E["<b>Output</b><br/>Predicted Log-Return"]

    style A fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style B fill:#2980b9,stroke:#1f618d,color:#fff,stroke-width:2px
    style C fill:#7f8c8d,stroke:#566573,color:#fff,stroke-width:2px
    style D fill:#e67e22,stroke:#ca6f1e,color:#fff,stroke-width:2px
    style E fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px
EOF
mmdc -i simple_lstm.mmd -o "$OUT/mermaid_simple_lstm.png" -w 600 -H 800 -b white 2>/dev/null
echo "Done: simple_lstm"

# 2. Stacked LSTM
cat > stacked_lstm.mmd << 'EOF'
graph TD
    A["<b>Input Sequence</b><br/>(batch, seq_len, 20 features)"] --> B1
    B1["<b>LSTM Layer 1</b><br/>hidden=128"] --> B2
    B2["<b>LSTM Layer 2</b><br/>hidden=128, dropout=0.2"] --> B3
    B3["<b>LSTM Layer 3</b><br/>hidden=128, dropout=0.2"] --> C
    C["<b>Last Hidden State</b><br/>(batch, 128)"] --> D1
    D1["<b>FC + ReLU</b><br/>128 → 64"] --> D2
    D2["<b>FC</b><br/>64 → 1"] --> E
    E["<b>Output</b><br/>Predicted Log-Return"]

    style A fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style B1 fill:#2980b9,stroke:#1f618d,color:#fff,stroke-width:2px
    style B2 fill:#2471a3,stroke:#1a5276,color:#fff,stroke-width:2px
    style B3 fill:#1f618d,stroke:#154360,color:#fff,stroke-width:2px
    style C fill:#7f8c8d,stroke:#566573,color:#fff,stroke-width:2px
    style D1 fill:#e67e22,stroke:#ca6f1e,color:#fff,stroke-width:2px
    style D2 fill:#d35400,stroke:#a04000,color:#fff,stroke-width:2px
    style E fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px
EOF
mmdc -i stacked_lstm.mmd -o "$OUT/mermaid_stacked_lstm.png" -w 600 -H 900 -b white 2>/dev/null
echo "Done: stacked_lstm"

# 3. BiLSTM
cat > bilstm.mmd << 'EOF'
graph TD
    A["<b>Input Sequence</b><br/>(batch, seq_len, 20 features)"] --> B1
    B1["<b>BiLSTM Layer 1</b><br/>Forward ↔ Backward, h=128"] --> B2
    B2["<b>BiLSTM Layer 2</b><br/>Forward ↔ Backward, dropout=0.2"] --> C
    C["<b>Concatenate</b><br/>Forward + Backward = 256"] --> D1
    D1["<b>FC + ReLU</b><br/>256 → 128"] --> D2
    D2["<b>FC</b><br/>128 → 1"] --> E
    E["<b>Output</b><br/>Predicted Log-Return"]

    style A fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style B1 fill:#2980b9,stroke:#1f618d,color:#fff,stroke-width:2px
    style B2 fill:#2471a3,stroke:#1a5276,color:#fff,stroke-width:2px
    style C fill:#e74c3c,stroke:#c0392b,color:#fff,stroke-width:2px
    style D1 fill:#e67e22,stroke:#ca6f1e,color:#fff,stroke-width:2px
    style D2 fill:#d35400,stroke:#a04000,color:#fff,stroke-width:2px
    style E fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px
EOF
mmdc -i bilstm.mmd -o "$OUT/mermaid_bilstm.png" -w 600 -H 900 -b white 2>/dev/null
echo "Done: bilstm"

# 4. LSTM + Attention
cat > lstm_attention.mmd << 'EOF'
graph TD
    A["<b>Input Sequence</b><br/>(batch, seq_len, 20 features)"] --> B1
    B1["<b>LSTM Layer 1</b><br/>hidden=128"] --> B2
    B2["<b>LSTM Layer 2</b><br/>hidden=128, dropout=0.2"] --> H
    H["<b>All Hidden States</b><br/>h₁, h₂, ..., hₜ"] --> AT1
    AT1["<b>Bahdanau Attention</b><br/>score = V·tanh(W·hᵢ)"] --> AT2
    AT2["<b>Softmax Weights</b><br/>α₁, α₂, ..., αₜ"] --> CTX
    CTX["<b>Context Vector</b><br/>c = Σ αᵢ·hᵢ"] --> D
    D["<b>FC Layer</b><br/>128 → 1"] --> E
    E["<b>Output</b><br/>Predicted Log-Return"]

    style A fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style B1 fill:#2980b9,stroke:#1f618d,color:#fff,stroke-width:2px
    style B2 fill:#2471a3,stroke:#1a5276,color:#fff,stroke-width:2px
    style H fill:#7f8c8d,stroke:#566573,color:#fff,stroke-width:2px
    style AT1 fill:#c0392b,stroke:#922b21,color:#fff,stroke-width:2px
    style AT2 fill:#e74c3c,stroke:#c0392b,color:#fff,stroke-width:2px
    style CTX fill:#d35400,stroke:#a04000,color:#fff,stroke-width:2px
    style D fill:#e67e22,stroke:#ca6f1e,color:#fff,stroke-width:2px
    style E fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px
EOF
mmdc -i lstm_attention.mmd -o "$OUT/mermaid_lstm_attention.png" -w 600 -H 1000 -b white 2>/dev/null
echo "Done: lstm_attention"

# 5. Training Pipeline
cat > pipeline.mmd << 'EOF'
graph LR
    A["📂 Load<br/>Parquet"] --> B["📊 Compute<br/>20 Features"]
    B --> C["✂️ Walk-Forward<br/>Split (3 folds)"]
    C --> D["⚖️ RobustScaler<br/>(per fold)"]
    D --> E["📦 PyTorch<br/>DataLoader"]
    E --> F["🧠 Train LSTM<br/>on GPU"]
    F --> G["⏱️ Early Stop<br/>Best Model"]
    G --> H["📈 Metrics<br/>& Stat Tests"]
    H --> I["📄 Figures<br/>& Report"]

    style A fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style B fill:#17a589,stroke:#117a65,color:#fff,stroke-width:2px
    style C fill:#2e86c1,stroke:#1a5276,color:#fff,stroke-width:2px
    style D fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px
    style E fill:#2980b9,stroke:#1f618d,color:#fff,stroke-width:2px
    style F fill:#c0392b,stroke:#922b21,color:#fff,stroke-width:2px
    style G fill:#e67e22,stroke:#ca6f1e,color:#fff,stroke-width:2px
    style H fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style I fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px
EOF
mmdc -i pipeline.mmd -o "$OUT/mermaid_training_pipeline.png" -w 1400 -H 300 -b white 2>/dev/null
echo "Done: pipeline"

# 6. Experiment Overview
cat > experiment.mmd << 'EOF'
graph TD
    subgraph Assets
        X["XAUUSD 1H<br/>33,533 bars"]
        BTC["BTCUSD 1D<br/>4,056 bars"]
    end
    subgraph Models
        M1["Simple LSTM<br/>76K params"]
        M2["Stacked LSTM<br/>357K params"]
        M3["BiLSTM<br/>581K params"]
        M4["LSTM+Attention<br/>233K params"]
    end
    subgraph Config
        L["Lookback<br/>30, 60 bars"]
        H["Horizon<br/>1, 5 steps"]
    end
    subgraph Validation
        WF["Walk-Forward CV<br/>3 folds + purge"]
    end

    Assets --> WF
    Models --> WF
    Config --> WF
    WF --> R["<b>32 Experiments</b><br/>on NVIDIA RTX 3050 Ti"]
    R --> OUT["<b>Results</b><br/>148 Figures + Metrics + PDF"]

    style X fill:#f39c12,stroke:#d68910,color:#fff,stroke-width:2px
    style BTC fill:#e67e22,stroke:#ca6f1e,color:#fff,stroke-width:2px
    style M1 fill:#3498db,stroke:#2471a3,color:#fff,stroke-width:2px
    style M2 fill:#2980b9,stroke:#1f618d,color:#fff,stroke-width:2px
    style M3 fill:#2471a3,stroke:#1a5276,color:#fff,stroke-width:2px
    style M4 fill:#1f618d,stroke:#154360,color:#fff,stroke-width:2px
    style L fill:#1abc9c,stroke:#16a085,color:#fff,stroke-width:2px
    style H fill:#16a085,stroke:#117a65,color:#fff,stroke-width:2px
    style WF fill:#9b59b6,stroke:#7d3c98,color:#fff,stroke-width:2px
    style R fill:#c0392b,stroke:#922b21,color:#fff,stroke-width:2px
    style OUT fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
EOF
mmdc -i experiment.mmd -o "$OUT/mermaid_experiment_overview.png" -w 900 -H 800 -b white 2>/dev/null
echo "Done: experiment_overview"

echo ""
echo "All Mermaid diagrams generated!"
