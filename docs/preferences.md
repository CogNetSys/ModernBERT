# Preferences: Configuring Your ModernBERT Experience 🛠️

This document outlines the various configuration options and preferences available when working with ModernBERT, particularly the `lightonai/modernbert-embed-large` model. These settings allow you to customize the model's behavior, adapt it to your specific needs, and optimize its performance for different tasks and environments.

---

## 1. Model Configuration ⚙️

The model's configuration determines its architecture, including the number of layers, attention heads, and hidden size. While the `lightonai/modernbert-embed-large` model comes with a pre-defined configuration, you can modify certain aspects when loading the model or during fine-tuning.

**`transformers.PretrainedConfig`**

-   This class stores the configuration of a `PreTrainedModel` or a `TFPreTrainedModel`.
-   You can access the configuration of a loaded model through the `model.config` attribute.

**Key Configurable Parameters:**

-   **`hidden_size`**: The dimensionality of the encoder layers and the pooler layer (default: 768 for `lightonai/modernbert-embed-large`).
-   **`intermediate_size`**: The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder (default: 3072 for `lightonai/modernbert-embed-large`).
-   **`num_hidden_layers`**: The number of hidden layers in the Transformer encoder (default: 12 for base, 24 for large).
-   **`num_attention_heads`**: The number of attention heads for each attention layer in the Transformer encoder (default: 12 for base, 16 for large).
-   **`max_position_embeddings`**: The maximum sequence length that this model might ever be used with. For the `lightonai/modernbert-embed-large` model its 8192.
-   **`type_vocab_size`**: The vocabulary size of the `token_type_ids` passed into the model.
-   **`initializer_range`**: The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
-   **`layer_norm_eps`**: The epsilon used by the layer normalization layers.
-   **`gradient_checkpointing`**: Whether to use gradient checkpointing to save memory at the expense of slower backward pass.
-   **`position_embedding_type`**: Type of position embedding. ModernBERT uses `rope` (Rotary Positional Embeddings).

**Example:**

```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("lightonai/modernbert-embed-large")
print(config)

# Modify the configuration (e.g., change the number of layers)
config.num_hidden_layers = 24

# Load the model with the modified configuration
model = AutoModel.from_config(config)
```

**Note:** Modifying the model's architecture can have significant implications for its performance and compatibility with pre-trained weights. It's generally recommended to stick with the default configuration unless you have a specific reason to change it and are prepared to pre-train the model from scratch or carefully fine-tune it with the new architecture.

---

## 2. Tokenizer Configuration 🔤

The tokenizer's configuration affects how the input text is processed before being fed to the model.

**`transformers.PreTrainedTokenizer`**

-   You can access the tokenizer's configuration through the `tokenizer.init_kwargs` attribute.

**Key Configurable Parameters:**

-   **`model_max_length`**: The maximum sequence length that the tokenizer will handle.
-   **`padding_side`**: The side on which the sequences will be padded ('left' or 'right').
-   **`truncation_side`**: The side on which the sequences will be truncated ('left' or 'right').
-   **`special_tokens`**: A dictionary of special tokens used by the tokenizer (e.g., `cls_token`, `sep_token`, `pad_token`, `mask_token`).

**Example:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("lightonai/modernbert-embed-large")

# Access the tokenizer's configuration
print(tokenizer.model_max_length)
print(tokenizer.padding_side)

# Change the maximum length (if needed)
tokenizer.model_max_length = 1024
```

**Note:** It's generally recommended to use the default tokenizer configuration that comes with the pre-trained model unless you have a specific reason to change it.

---

## 3. Inference Preferences 🚀

Several options can be configured during inference to optimize performance and adapt the model's behavior:

-   **Batch Size**: Using a larger batch size can improve throughput, especially on GPUs, but requires more memory.
-   **Sequence Length**: While ModernBERT can handle sequences up to 8192 tokens, using shorter sequences can reduce memory usage and improve speed.
-   **Mixed Precision (FP16)**: Using half-precision floating-point numbers can speed up inference and reduce memory usage on compatible hardware.
-   **Device Selection**: You can choose to run inference on the CPU or GPU, depending on your hardware and performance requirements.

**Example:**

```python
import torch

# Set batch size
batch_size = 16

# Enable mixed precision
with torch.cuda.amp.autocast():
    # Run inference with mixed precision
    outputs = model(**inputs)

# Move the model and inputs to the desired device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
```

---

## 4. Fine-tuning Preferences 🏋️‍♀️

When fine-tuning ModernBERT, you have many options to control the training process:

-   **Learning Rate**: The learning rate is a crucial hyperparameter that determines the step size during optimization.
-   **Number of Epochs**: The number of times the training loop iterates over the entire dataset.
-   **Optimizer**: The optimization algorithm used to update the model's weights (e.g., AdamW, SGD).
-   **Learning Rate Scheduler**: A scheduler that dynamically adjusts the learning rate during training.
-   **Weight Decay**: A regularization technique that penalizes large weights to prevent overfitting.
-   **Loss Function**: The function used to measure the difference between the model's predictions and the true labels.
-   **Evaluation Metric**: The metric used to evaluate the model's performance during training and validation.

**Example (using `transformers.TrainingArguments`):**

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)
```

---

## 5. Hardware and Environment Preferences 💻

-   **GPU Selection**: If you have multiple GPUs, you can choose which one(s) to use for training or inference.
-   **CUDA Version**: Ensure that you have the correct CUDA version installed for your GPU and PyTorch.
-   **Memory Allocation**: You can control how much memory is allocated to the model and the intermediate activations.
-   **Distributed Training**: For large-scale training, you can use distributed training across multiple GPUs or machines.

**Example (setting CUDA device):**

```python
import os

# Set CUDA_VISIBLE_DEVICES environment variable to use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

# Or set the device in PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

---

## 🏁 Conclusion

ModernBERT offers a wide range of configuration options and preferences that allow you to customize its behavior, optimize its performance, and adapt it to your specific needs. By understanding these options and carefully choosing the appropriate settings, you can maximize the effectiveness of `lightonai/modernbert-embed-large` for your particular application, hardware, and performance requirements. Keep in mind that different tasks and domains may require different configurations, and experimentation is often key to finding the optimal settings.