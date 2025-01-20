# Inference and Usage

### 🧠 Loading the Trained Models

To utilize the trained models for inference, follow these steps:

#### For Small Model

```
# Initialize the tokenizer and base model
tokenizer_sm = AutoTokenizer.from_pretrained("answerdotai/modernbert-base")
base_model_sm = AutoModel.from_pretrained("answerdotai/modernbert-base")

# Initialize the joint model
model_sm = JointNERAnomalyModel(base_model_sm, num_ner_labels, num_anomaly_labels)

# Load the saved state dictionary
model_path_sm = "/content/drive/MyDrive/JointNERAnomalyModel/joint_ner_anomaly_model_sm.pth"
model_sm.load_state_dict(torch.load(model_path_sm, map_location=device))
model_sm.to(device)
model_sm.eval()
print("Small model loaded successfully!")
```

#### For Large Model

```
# Initialize the tokenizer and base model
tokenizer_lg = AutoTokenizer.from_pretrained("answerdotai/modernbert-large")
base_model_lg = AutoModel.from_pretrained("answerdotai/modernbert-large")

# Initialize the joint model
model_lg = JointNERAnomalyModel(base_model_lg, num_ner_labels, num_anomaly_labels)

# Load the saved state dictionary
model_path_lg = "/content/drive/MyDrive/JointNERAnomalyModel/joint_ner_anomaly_model_lg.pth"
model_lg.load_state_dict(torch.load(model_path_lg, map_location=device))
model_lg.to(device)
model_lg.eval()
print("Large model loaded successfully!")
```

### 🔍 Performing Inference

Create a function to perform NER and Anomaly Detection on new text inputs.

```
def predict(text, model, tokenizer, label_to_id, device, max_length=128):
    """
    Performs NER and Anomaly Detection on the input text.

    Args:
        text (str): The input text.
        model: The trained JointNERAnomalyModel.
        tokenizer: The tokenizer instance.
        label_to_id (dict): Mapping from label strings to IDs.
        device: The computation device.
        max_length (int): Maximum sequence length.

    Returns:
        Tuple[List[Tuple[str, str]], str]: List of (Entity, Label) and Anomaly Prediction.
    """
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        ner_logits, anomaly_logits = model(input_ids, attention_mask)

    # NER Predictions
    ner_pred = torch.argmax(ner_logits, dim=-1).cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [list(label_to_id.keys())[list(label_to_id.values()).index(l)] for l in ner_pred[:len(tokens)]]

    entities = []
    current_entity = []
    current_label = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
                current_entity = []
            current_entity.append(token)
            current_label = label[2:]
        elif label.startswith("I-") and current_label == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
                current_entity = []
            current_label = None

    if current_entity:
        entities.append((" ".join(current_entity), current_label))

    # Anomaly Detection Prediction
    anomaly_pred = torch.argmax(anomaly_logits, dim=-1).cpu().numpy()[0]
    anomaly_label = "Anomaly" if anomaly_pred == 1 else "Normal"

    return entities, anomaly_label
```

### 💡 Example Usage

Perform predictions using both the small and large models.

```
# Example text
text = "Unauthorized access detected for user admin456 from IP 192.168.1.1."

# Perform prediction using the small model
entities_sm, anomaly_sm = predict(text, model_sm, tokenizer_sm, label_to_id, device)
print("Entities (Small Model):", entities_sm)
print("Anomaly Detection (Small Model):", anomaly_sm)

# Perform prediction using the large model
entities_lg, anomaly_lg = predict(text, model_lg, tokenizer_lg, label_to_id, device)
print("Entities (Large Model):", entities_lg)
print("Anomaly Detection (Large Model):", anomaly_lg)
```

**Sample Output:**

Entities (Small Model): [('admin456', 'USERNAME'), ('192.168.1.1', 'IP')]

Anomaly Detection (Small Model): Anomaly

Entities (Large Model): [('admin456', 'USERNAME'), ('192.168.1.1', 'IP')]

Anomaly Detection (Large Model): Anomaly