# Saving Models to Google Drive

### 💾 Saving Trained Models to Google Drive

After successfully training your models, it's essential to save them for future use and deployment. Follow these steps to save both the small and large versions to your **Google Drive**.

#### Mounting Google Drive

First, mount your Google Drive to the Colab environment:

```
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')
```

**Steps:**

1. Run the above code cell.
2. Follow the prompted link to authorize Colab to access your Google Drive.
3. Paste the authorization code back into the notebook.

#### Saving the Small Model

```
# Define the directory in Google Drive
model_dir_sm = "/content/drive/MyDrive/JointNERAnomalyModel"
os.makedirs(model_dir_sm, exist_ok=True)

# Save the small model state dictionary
torch.save(model_sm.state_dict(), os.path.join(model_dir_sm, "joint_ner_anomaly_model_sm.pth"))
print(f"Small model saved to {model_dir_sm}/joint_ner_anomaly_model_sm.pth")
```

#### Saving the Large Model

```
# Define the directory in Google Drive
model_dir_lg = "/content/drive/MyDrive/JointNERAnomalyModel"
os.makedirs(model_dir_lg, exist_ok=True)

# Save the large model state dictionary
torch.save(model_lg.state_dict(), os.path.join(model_dir_lg, "joint_ner_anomaly_model_lg.pth"))
print(f"Large model saved to {model_dir_lg}/joint_ner_anomaly_model_lg.pth")
```

**🔍 Note:**

- Ensure that the directory path (`/content/drive/MyDrive/JointNERAnomalyModel`) exists in your Google Drive. You can change it as per your preference.
- Keeping models in Google Drive allows easy access and deployment across different environments.

For more details on managing and accessing your saved models, refer to the [Inference and Usage](inference.md) section.