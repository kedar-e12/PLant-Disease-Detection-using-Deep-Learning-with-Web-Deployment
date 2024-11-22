# PLant-Disease-Detection-using-Deep-Learning-with-Web-Deployment-using-StreamLit
1. Project Overview -

a.Built an automated plant disease detection system using a Convolutional Neural Network (CNN) model.The CNN model begins with multiple convolutional layers, each with filters of increasing sizes (32, 64, 128, 256, and 512), to capture various features. Same padding was applied to preserve spatial dimensions, and a stride of 2 was used for efficient down-sampling.

b.Achieved an impressive accuracy of 99.4% on the test dataset.

c.Addressed overfitting using techniques like dropout layers and robust preprocessing.

d.Evaluated model performance using metrics such as accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC curves.

e.Developed a Streamlit-based web application for users to upload plant leaf images and receive predictions.

f.Designed the system for high accuracy and usability, combining deep learning techniques with an intuitive interface.

2. Dataset Description-

Dataset Link - https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

The dataset comprises 87,000 RGB images of plant leaves across 38 classes, representing various diseases and healthy plants. These images underwent preprocessing, including resizing to 128x128 pixels and normalization to scale pixel values between 0 and 1. The data is split into 80% for training, 20% for validation, and an unseen test set of 33 images, ensuring balanced and representative evaluation during model training and testing. Additionally, offline augmentation techniques were applied to enrich the dataset and improve model generalization.

3.Steps to run the project-

a.Download Dataset and Clone the Repository.

b.Install all requirements in the "requirements.txt" file.

c.Run "train_plant_disease.ipynb" for training the model.

d.Use "test_plant_disease.ipynb" for testing the model with the images in the "test" dataset.

e.Run the command "streamlit run main.py" in the terminal for the web application and input image from the "test" set on the web application and get output.
