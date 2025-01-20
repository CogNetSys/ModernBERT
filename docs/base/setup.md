# Setting Up Google Colab

### 🛠️ Creating a New Notebook

To begin your experiment, set up a new **Google Colab** notebook by following these steps:

- **Access Google Colab:**
    - Open your web browser and navigate to [Google Colab](https://colab.research.google.com/).
- **Create a New Notebook:**
    - Click on `File` in the top-left corner.
    - Select `New Notebook` or choose `New notebook in Drive` to save it directly to your Google Drive.
- **Name Your Notebook:**
    - Click on the notebook title (default is `Untitled`) at the top.
    - Rename it appropriately, for example, `Joint_NER_Anomaly_Detection-00`.

### 🔒 Configuring Secrets

To securely access your Hugging Face API key within the notebook, follow these steps:

1. **Open Secrets Manager:**

    - Click on the "key" icon on the left sidebar to open the **Secrets** window.

2. **Add a New Secret:**

    - Click on `+ Add a new secret`.

3. **Configure the Secret:**

    - **Name:** `HF_API_KEY`
    - **Value:** `<your_huggingface_api_key>` *(Replace with your actual Hugging Face API key)*

4. **Set Access Permissions:**

    - Toggle `Notebook access` to `Allow`.

5. **Save the Secret:**

    - Click `Add secret` to save.

**🔑 Important:** Ensure that your Hugging Face API key is kept confidential and not exposed in the notebook. This key grants access to your Hugging Face account and models.

For more details on securely managing secrets, refer to the [Saving Models to Google Drive](saving_models.md) section.