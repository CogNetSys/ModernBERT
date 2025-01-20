# Training the Model

### 📦 Installing Dependencies

Begin by installing the necessary libraries required for the experiment. Execute the following commands in separate code cells within your Colab notebook:

Install Hugging Face Transformers and other dependencies:

```
pip install transformers
pip install torch
pip install scikit-learn
pip install tqdm
pip install seaborn
pip install nltk
pip install spacy
python -m spacy download en_core_web_sm
```

### 📚 Importing Libraries

Import the required libraries in your notebook:

```
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import random
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
```

Download NLTK data and set random seeds for reproducibility:

```
# Download NLTK data
nltk.download('wordnet')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

**Expected Output:**

Using device: cuda

### 🔧 Initializing the Tokenizer and Model

Depending on the model version you wish to train (small or large), initialize the tokenizer and base model accordingly.

#### For Small Model

```
from transformers import AutoTokenizer, AutoModel

# Initialize the tokenizer with ModernBERT Base
model_name_sm = "answerdotai/modernbert-base"  # Small model
tokenizer_sm = AutoTokenizer.from_pretrained(model_name_sm)

# Initialize the base model
try:
    base_model_sm = AutoModel.from_pretrained(model_name_sm)
    print("Successfully loaded ModernBERT Base!")
except ValueError as e:
    print(f"Error loading model '{model_name_sm}': {e}")
```

#### For Large Model

```
from transformers import AutoTokenizer, AutoModel

# Initialize the tokenizer with ModernBERT Large
model_name_lg = "answerdotai/modernbert-large"  # Large model
tokenizer_lg = AutoTokenizer.from_pretrained(model_name_lg)

# Initialize the base model
try:
    base_model_lg = AutoModel.from_pretrained(model_name_lg)
    print("Successfully loaded ModernBERT Large!")
except ValueError as e:
    print(f"Error loading model '{model_name_lg}': {e}")
```

**🔍 Note:** Ensure that `answerdotai/modernbert-base` and `answerdotai/modernbert-large` are available on Hugging Face's model hub. If not, verify the model names or consult the model provider.

### 🛠️ Defining the JointNERAnomalyModel

Implement the `JointNERAnomalyModel` with the corrected forward method to handle the absence of `pooler_output` by using the `[CLS]` token for anomaly classification.

```
import torch.nn as nn

class JointNERAnomalyModel(nn.Module):
    def __init__(self, base_model, num_ner_labels, num_anomaly_labels):
        super(JointNERAnomalyModel, self).__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size

        # NER classifier: processes all token embeddings
        self.ner_classifier = nn.Linear(self.hidden_size, num_ner_labels)

        # Anomaly detection classifier: processes CLS token embedding
        self.anomaly_classifier = nn.Linear(self.hidden_size, num_anomaly_labels)

    def forward(self, input_ids, attention_mask):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Hidden states of all tokens (sequence output)
        sequence_output = outputs.last_hidden_state

        # Use the CLS token (first token) for anomaly classification
        cls_output = sequence_output[:, 0, :]  # Shape: [batch_size, hidden_size]

        # NER logits for each token
        ner_logits = self.ner_classifier(sequence_output)

        # Anomaly logits from the CLS token
        anomaly_logits = self.anomaly_classifier(cls_output)

        return ner_logits, anomaly_logits
```

### ⚙️ Initializing the Joint Model

Choose the model version you intend to train and initialize accordingly.

#### For Small Model

```
# Define label counts
num_ner_labels = len(label_to_id)  # Total unique NER labels
num_anomaly_labels = 2  # Binary classification: Normal or Anomaly

# Initialize the joint model with the small base model
model_sm = JointNERAnomalyModel(base_model_sm, num_ner_labels, num_anomaly_labels)

# Move model to GPU
model_sm.to(device)
```

#### For Large Model

```
# Define label counts
num_ner_labels = len(label_to_id)  # Total unique NER labels
num_anomaly_labels = 2  # Binary classification: Normal or Anomaly

# Initialize the joint model with the large base model
model_lg = JointNERAnomalyModel(base_model_lg, num_ner_labels, num_anomaly_labels)

# Move model to GPU
model_lg.to(device)
```

### 📉 Defining Loss Functions and Optimizer

```
# Define loss functions
ner_loss_fn = nn.CrossEntropyLoss(ignore_index=label_to_id[O_label])  # Ignore 'O' label in loss
anomaly_loss_fn = nn.CrossEntropyLoss()

# Define optimizer
optimizer_sm = torch.optim.AdamW(model_sm.parameters(), lr=2e-5)
optimizer_lg = torch.optim.AdamW(model_lg.parameters(), lr=2e-5)
```

### ⏱️ Setting Up the Learning Rate Scheduler

```
from transformers import get_linear_schedule_with_warmup

epochs = 3  # Adjust based on your requirements and Colab's runtime limits

# For Small Model
total_steps_sm = len(train_loader_sm) * epochs

scheduler_sm = get_linear_schedule_with_warmup(
    optimizer_sm,
    num_warmup_steps=0,
    num_training_steps=total_steps_sm
)

# For Large Model
total_steps_lg = len(train_loader_lg) * epochs

scheduler_lg = get_linear_schedule_with_warmup(
    optimizer_lg,
    num_warmup_steps=0,
    num_training_steps=total_steps_lg
)
```

# 🏋️‍♂️ Training Loop

#### For Small Model

```
for epoch in range(epochs):
    model_sm.train()
    total_loss = 0
    for batch in tqdm(train_loader_sm, desc=f"Training Epoch {epoch+1} - Small Model"):
        optimizer_sm.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ner_labels = batch['ner_labels'].to(device)
        anomaly_labels = batch['anomaly_labels'].to(device)

        ner_logits, anomaly_logits = model_sm(input_ids, attention_mask)

        # Compute NER loss
        ner_loss = ner_loss_fn(ner_logits.view(-1, num_ner_labels), ner_labels.view(-1))

        # Compute Anomaly Detection loss
        anomaly_loss = anomaly_loss_fn(anomaly_logits, anomaly_labels)

        # Total loss
        loss = ner_loss + anomaly_loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model_sm.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer_sm.step()

        # Scheduler step
        scheduler_sm.step()

    avg_loss = total_loss / len(train_loader_sm)
    print(f"Epoch {epoch+1} - Small Model Average Loss: {avg_loss}")
```

#### For Large Model

```
for epoch in range(epochs):
    model_lg.train()
    total_loss = 0
    for batch in tqdm(train_loader_lg, desc=f"Training Epoch {epoch+1} - Large Model"):
        optimizer_lg.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ner_labels = batch['ner_labels'].to(device)
        anomaly_labels = batch['anomaly_labels'].to(device)

        ner_logits, anomaly_logits = model_lg(input_ids, attention_mask)

        # Compute NER loss
        ner_loss = ner_loss_fn(ner_logits.view(-1, num_ner_labels), ner_labels.view(-1))

        # Compute Anomaly Detection loss
        anomaly_loss = anomaly_loss_fn(anomaly_logits, anomaly_labels)

        # Total loss
        loss = ner_loss + anomaly_loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model_lg.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer_lg.step()

        # Scheduler step
        scheduler_lg.step()

    avg_loss = total_loss / len(train_loader_lg)
    print(f"Epoch {epoch+1} - Large Model Average Loss: {avg_loss}")
```

**📈 Monitoring Training:**

- **Loss Tracking:** Observe the average loss per epoch to monitor convergence.
- **TensorBoard Integration:** Optionally, integrate TensorBoard for more detailed monitoring.

Refer to the [Troubleshooting](troubleshooting.md) section for solutions to common training issues.