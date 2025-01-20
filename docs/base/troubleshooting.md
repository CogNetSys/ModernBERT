# Troubleshooting

### 🛠️ Common Issues and Solutions

#### 🔑 Model Loading Errors

**Issue:** KeyError 'modernbert'

**Cause:** The `BaseModelOutput` object returned by ModernBERT does not include a `pooler_output` attribute, leading to compatibility issues.

**Solution:**

1. **Use CLS Token for Anomaly Classification:**
   - Modify the `JointNERAnomalyModel` to use the `[CLS]` token embedding from `last_hidden_state` instead of `pooler_output`.

2. **Updated Forward Method:**

```
def forward(self, input_ids, attention_mask):
    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state
    cls_output = sequence_output[:, 0, :]  # Use CLS token

    ner_logits = self.ner_classifier(sequence_output)
    anomaly_logits = self.anomaly_classifier(cls_output)

    return ner_logits, anomaly_logits
```

3. **Ensure Transformers Library is Updated:**

```
pip install --upgrade transformers
```

#### 🧩 Model Not Found on Hugging Face

**Issue:** ValueError: The checkpoint you are trying to load has model type `modernbert` but Transformers does not recognize this architecture.

**Cause:** The specified ModernBERT model (`answerdotai/modernbert-base` or `answerdotai/modernbert-large`) may not exist on Hugging Face's model hub or the Transformers library is outdated.

**Solution:**

1. **Verify Model Availability:**
   - Visit the [Hugging Face model hub](https://huggingface.co/models) and search for `answerdotai/modernbert-base` or `answerdotai/modernbert-large`.
   - If the model does not exist, confirm the correct model name or consult the model provider.

2. **Update Transformers Library:**

```
pip install --upgrade transformers
```

3. **Fallback to Compatible Model:**
   - If ModernBERT is unavailable, use `bert-base-uncased` or another compatible model.

```
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)
```

#### ⚡ CUDA Availability Issues

**Issue:** Model runs on CPU instead of GPU.

**Cause:** The Colab runtime is not set to use GPU, or GPU resources are not available.

**Solution:**

1. **Enable GPU in Colab:**
   - Click on `Runtime` in the top menu.
   - Select `Change runtime type`.
   - In the popup, set `Hardware accelerator` to `GPU`.
   - Click `Save` and restart the runtime if prompted.

2. **Verify GPU Availability:**

```
import torch
print(torch.cuda.is_available())  # Should return True
```

#### 🧠 Insufficient GPU Memory

**Issue:** Out-of-memory (OOM) errors during training.

**Cause:** The model or batch size is too large for the available GPU memory.

**Solution:**

1. **Reduce Batch Size:**
   - Decrease the `batch_size` parameter in your DataLoader.

```
batch_size = 8  # Example: Reduce from 16 to 8
```

2. **Decrease Maximum Sequence Length:**
   - Lower the `max_length` parameter to reduce memory consumption.

```
max_length = 128  # Example: Reduce from 256 to 128
```

3. **Use Gradient Accumulation:**
   - Accumulate gradients over multiple steps to simulate a larger batch size without increasing memory usage.

```
accumulation_steps = 4
for step, batch in enumerate(train_loader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 🛡️ Hugging Face API Key Issues

**Issue:** Authentication errors when accessing private models.

**Cause:** Incorrect or missing Hugging Face API key, or insufficient permissions.

**Solution:**

1. **Verify API Key:**
   - Ensure that the `HF_API_KEY` secret is correctly set in the Secrets Manager.
   - Double-check for any typos or missing characters.

2. **Check Permissions:**
   - Confirm that the API key has the necessary permissions to access the models you intend to use.

3. **Reload Secrets:**
   - Sometimes, restarting the Colab runtime can help in recognizing newly added secrets.

### 🏷️ Label Mapping Errors

**Issue:** KeyError when assigning labels.

**Cause:** Mismatch between entity types in the dataset and those defined in `entity_label_map`.

**Solution:**

1. **Verify Entity Types:**
   - Ensure that all entity types used in your dataset are correctly mapped in `entity_label_map`.

2. **Update `entity_label_map`:**
   - Add any missing entity types to the mapping.

```
entity_label_map = {
    "person": {"B": "B-PER", "I": "I-PER"},
    "organization": {"B": "B-ORG", "I": "I-ORG"},
    "location": {"B": "B-LOC", "I": "I-LOC"},
    "O": "O",
    "new_entity_type": {"B": "B-NEW", "I": "I-NEW"}  # Example
}
```

3. **Ensure Consistent Label IDs:**
   - Confirm that `label_to_id` includes all necessary labels without duplicates.