Food11 Image Classification with Convolutional Neural Networks (MobileNetV2)

This project presents a complete and well-structured deep learning pipeline for food image classification using the Food11 dataset.
The notebook was designed and implemented by Shayan Rokhva to explore the effectiveness of convolutional neural networks (CNNs) 
in handling image-based classification tasks in the food domain, 
where high intra-class variability and strong inter-class similarity often make learning challenging.
The goal is to train a deep model capable of accurately classifying food images into eleven distinct categories while maintaining efficiency and generalization.

1. Dataset and Problem Description
The dataset used in this project is Food11, publicly available on Kaggle. It contains a total of 16,643 RGB images distributed across 11 food categories,
including diverse classes such as meat dishes, rice-based meals, desserts, and soups. Each class exhibits substantial variation in lighting, background,
angle, and portion presentation, making it a challenging benchmark for visual classification. The dataset is divided into training, validation, and testing
subsets to evaluate model performance objectively. Each image is resized to 256×256 pixels to ensure consistency in input dimensions.
The central problem addressed in this project is multiclass food recognition, where the model learns to map pixel-level information into semantically meaningful class labels.
This task is significant for applications such as food intake monitoring, automated dietary assessment, restaurant automation, and food waste management.

3. Data Preprocessing and Augmentation
Prior to model training, several preprocessing and augmentation steps are applied to improve model robustness.
Each image is converted into a tensor and normalized using the mean and standard deviation values of the ImageNet dataset.
To prevent overfitting and enhance generalization, data augmentation techniques such as random horizontal flipping, random rotation, and color jittering are applied during training.
This increases the effective diversity of the training set without collecting new samples.
Additionally, the dataset is balanced by adjusting the sampling weights for each class to mitigate bias toward majority categories.
The preprocessing pipeline is implemented using the torchvision.transforms library,
and data loading is managed through efficient DataLoader objects with mini-batches for faster computation.

3. Model Architecture and Training
The model architecture is built using PyTorch, leveraging convolutional layers to extract hierarchical features from the input images.
The design follows the classical CNN paradigm with multiple Conv–BatchNorm–ReLU–MaxPool blocks.
The initial layers capture low-level features such as edges and textures, while deeper layers learn more abstract and semantic representations of food types.
Fully connected layers at the end perform classification across 11 categories through a softmax activation function.
For optimization, the model employs the CrossEntropyLoss function, which is standard for multiclass classification problems.
The optimizer used is Adam, selected for its adaptive learning rate and efficient convergence properties.
The learning rate starts at 0.001 and is adjusted dynamically using a scheduler to ensure stable convergence.
Training is conducted for multiple epochs, with both training loss and validation accuracy monitored after each epoch.
The model parameters yielding the highest validation accuracy are saved automatically for future inference.
Regularization techniques such as Dropout are incorporated in fully connected layers to reduce overfitting.
The implementation is GPU-compatible, allowing the notebook to leverage CUDA acceleration if available.
Progress bars (via tqdm) and loss-accuracy plots (via matplotlib) are used to monitor training behavior and diagnose convergence patterns.

5. Evaluation and Results
After training, the model is evaluated on the test set to assess generalization performance.
Evaluation metrics include accuracy, precision, recall, and F1-score, computed using sklearn.metrics.
Additionally, a confusion matrix is generated and visualized to identify misclassified categories and understand which food types are most commonly confused.
The trained CNN achieves strong performance on the Food11 dataset, demonstrating reliable classification accuracy across all classes.
While the model performs exceptionally well on visually distinct food types, slight confusion remains among visually similar categories
(e.g., different rice-based dishes).
The results confirm that the proposed CNN effectively captures discriminative features despite variations in illumination, plating, and viewpoint.
Visual inspection of sample predictions confirms that the model generalizes well to unseen data.
The notebook also includes code for saving and loading trained weights (model.pth), allowing further fine-tuning or transfer learning experiments.

5. Discussion and Potential Improvements
The project highlights several key insights into applying deep learning for food image classification.
First, CNNs can achieve high accuracy with moderate complexity when appropriate preprocessing and augmentation are applied.
Second, balancing data and employing proper regularization are crucial for avoiding bias and overfitting.
Finally, visualization of learned filters and misclassified samples provides interpretability and guides model improvement.
Future work may involve integrating attention mechanisms such as the Convolutional Block Attention Module (CBAM)
to enhance spatial and channel-wise feature extraction, or employing pretrained backbones like EfficientNetB7 or
ResNet50 for transfer learning. Ensemble strategies combining multiple CNN models could also further boost classification performance.
Additionally, extending the framework to real-time inference on mobile devices would make the system suitable
for practical applications in smart kitchens and dietary monitoring.

7. Conclusion

This notebook represents a solid and reproducible implementation of a convolutional neural network for the Food11 image classification task. It systematically covers all stages of the machine learning pipeline — from data preprocessing and augmentation to model training, evaluation, and interpretation. The workflow demonstrates how deep learning can be applied to real-world problems in the food domain with promising accuracy and scalability. The project not only serves as an educational resource for understanding CNN fundamentals but also lays the foundation for future extensions in food recognition, nutrition analysis, and intelligent food management systems.
