# Potato Disease Classification using CNN

This project implements a Convolutional Neural Network (CNN) model to classify potato leaf diseases from images using deep learning. It identifies three primary diseases — Early Blight, Late Blight, and Black Scurf — and helps in the early detection and prevention of crop damage.

## 🚀 Project Features

- Classifies images of potato leaves into:
  - Early Blight
  - Late Blight
  - Black Scurf
  - Healthy
- Trained on the PlantVillage dataset (Kaggle)
- Achieves high accuracy with CNN-based architecture
- Deployed using Google Colab and exported to .h5 format
- Integrated with a web application using FastAPI (backend) and React (frontend)

## 🌐 Web Application

After training and evaluating the model, a lightweight web interface was developed using:

- **FastAPI**: To serve the machine learning model as an API
- **React.js**: To provide a responsive and modern user interface
- **Upload Feature**: Allows users to upload a leaf image and get instant prediction
- **Deployment**: Ready to be deployed on platforms like Vercel (frontend) and Render/Heroku (backend)


## 📂 Project Structure

project/
│
├── data/                     # Dataset (training/test images)
├── model/                    # Saved models and ONNX format
├── backend/                  # FastAPI backend application
├── frontend/                 # React frontend app
├── Potato_Classifier.ipynb   # Training and testing notebook
├── utils/                    # Helper functions and preprocessing
├── README.md
└── LICENSE
```

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- FastAPI
- React.js
- Google Colab
- (Optional) Raspberry Pi for deployment

## 📊 Dataset

- Source: [PlantVillage Dataset - Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Includes labeled images of potato leaves for multiple disease categories

## 💡 How to Use

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/potato-disease-classification-cnn.git
   ```
2. Train the model using `Potato_Classifier.ipynb` or use the pre-trained model in `model/`
3. Navigate to `backend/` and run FastAPI:
   ```bash
   uvicorn main:app --reload
   ```
4. Navigate to `frontend/` and start React app:
   ```bash
   npm install
   npm start
   ```
5. Open browser and test predictions through the web UI

## 🤖 Future Work

- Real-time detection using mobile or Raspberry Pi camera
- Integration with alert system for farmers
- AIoT dashboard for field monitoring

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For queries, feedback, or collaboration:

