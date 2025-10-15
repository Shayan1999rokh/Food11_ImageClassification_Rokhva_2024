ChatGPT said:

Food11 Image Classification using Convolutional Neural Networks (MobileNetV2)

This project presents a comprehensive and well-organized deep learning framework for classifying food images from the Food11 dataset. Designed and implemented by Shayan Rokhva, the work explores the capability of Convolutional Neural Networks (CNNs) to tackle the complex visual challenges in food recognition — characterized by high intra-class diversity and strong inter-class resemblance. The primary objective is to train a high-performing yet efficient deep model that accurately categorizes food images into eleven distinct classes, ensuring both robustness and generalization.

Dataset and Problem Description

The project utilizes the Food11 dataset from Kaggle, comprising 16,643 RGB images spanning 11 diverse food categories, including meat-based dishes, rice meals, soups, and desserts. Each category exhibits variations in lighting, background, camera angle, and portion size, making it a demanding benchmark for visual learning. The dataset is divided into training, validation, and test subsets to objectively measure performance. All images are resized to 256×256 pixels for input consistency. The central task involves multiclass food recognition, mapping image pixels to semantically meaningful food labels. This research holds practical significance for dietary monitoring, automated restaurant systems, nutrition analysis, and food waste management.

Data Preprocessing and Augmentation

To ensure stability and generalization, extensive preprocessing and augmentation techniques are employed. Each image is converted to a tensor and normalized using ImageNet’s mean and standard deviation. Augmentation techniques — including random rotations, horizontal flips, and color jittering — are applied to increase data diversity and mitigate overfitting. Class imbalance is addressed through weighted sampling, ensuring fair contribution from minority categories. The entire preprocessing pipeline is implemented with torchvision.transforms, and optimized data handling is achieved using PyTorch DataLoader with mini-batches for parallelized GPU processing.

Model Architecture and Training

The classification model is implemented in PyTorch, following the CNN paradigm of stacked Conv–BatchNorm–ReLU–MaxPool layers. Early layers capture basic visual cues (edges, shapes, and textures), while deeper layers extract complex and semantic representations. The fully connected layers perform final classification using a softmax output across 11 categories. Training is guided by the CrossEntropyLoss function, optimized via the Adam optimizer with an initial learning rate of 0.001, dynamically adjusted through a learning-rate scheduler. Dropout regularization is applied in dense layers to suppress overfitting. The training loop includes detailed performance tracking through tqdm progress bars and visual analysis of loss and accuracy curves using matplotlib. The best-performing model weights are automatically saved for future inference, and GPU acceleration via CUDA significantly boosts computational efficiency.

Evaluation and Results

After training, the model is rigorously evaluated on the test set using standard metrics — accuracy, precision, recall, and F1-score — computed through scikit-learn. A confusion matrix visualization reveals class-specific strengths and weaknesses, highlighting occasional misclassifications among visually similar dishes (e.g., different rice varieties). Overall, the CNN achieves strong and balanced performance across all food categories, effectively learning discriminative visual patterns despite real-world variability in lighting, plating, and orientation. The notebook also provides routines for saving and reloading model weights (model.pth), enabling reproducibility and further fine-tuning.

Discussion and Future Work

The study provides valuable insights into the design of deep models for food recognition. It shows that CNNs can achieve high accuracy with moderate complexity when supported by proper augmentation, balancing, and regularization. Visualizing filters and misclassified samples also enhances interpretability and model debugging. Future improvements may include integrating attention mechanisms like CBAM to refine spatial and channel-level feature learning, leveraging transfer learning with advanced pretrained models such as EfficientNetB7 or ResNet50, or employing ensemble learning for further performance gains. Moreover, optimizing the model for real-time mobile deployment could expand its application to smart kitchens, automated canteens, and diet-tracking platforms.

Conclusion

This notebook delivers a complete, reproducible, and educational deep learning pipeline for food image classification. Covering all major stages — from preprocessing and data augmentation to training, evaluation, and result interpretation — it effectively demonstrates the power of CNNs in solving real-world visual recognition problems. The project not only achieves high accuracy and scalability but also provides a strong foundation for future extensions in food analytics, dietary assessment, and intelligent food management systems.
