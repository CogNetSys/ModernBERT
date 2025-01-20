# Model Versions

### Model Versions

In this experiment, two versions of the **Joint NER and Anomaly Detection** model are developed to cater to different computational needs and performance requirements. Below are the detailed specifications of each version.

#### Small Model (149MB)

- **Model Name:** `answerdotai/modernbert-base`
- **Output Filename:** `joint_ner_anomaly_model_sm.pth`
- **Description:** This version is optimized for faster training and inference, making it suitable for environments with limited computational resources. It strikes a balance between performance and resource utilization.

**Key Characteristics:**

- **Size:** 149MB
- **Parameters:** Fewer parameters leading to quicker training times.
- **Use Cases:** Ideal for deployment in resource-constrained settings or when rapid iterations are necessary.

For more details, refer to the [Training the Model](training.md) section.

#### Large Model (395MB)

- **Model Name:** `answerdotai/modernbert-large`
- **Output Filename:** `joint_ner_anomaly_model_lg.pth`
- **Description:** The larger model offers enhanced performance with a greater number of parameters, suitable for tasks requiring higher accuracy and deeper contextual understanding.

**Key Characteristics:**

- **Size:** 395MB
- **Parameters:** More parameters allow for capturing complex patterns in data.
- **Use Cases:** Best suited for scenarios where maximum accuracy is paramount and computational resources are ample.

For more details, refer to the [Training the Model](training.md) section.