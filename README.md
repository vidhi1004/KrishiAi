# KrishiAi: Crop Recommendation System 🌿

KrishiAi is an intelligent crop recommendation system designed to help farmers make informed decisions about which crops to plant based on environmental factors. By leveraging machine learning, this tool provides predictions to maximize agricultural yield and efficiency.

## ✨ Features

- **Intelligent Recommendations:** Utilizes a machine learning model to suggest the most suitable crop for given soil and weather conditions.
- **Data-Driven Insights:** Considers key factors like nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall.
- **Easy-to-Use Web Interface:** A simple and intuitive web application built with Flask for easy interaction.

## 🛠️ How It Works

The application uses a pre-trained machine learning model (`crop_recommendation.pkl`) to predict the best crop to grow. 
The model was trained on a dataset of various crop types and their corresponding environmental conditions. 
The Flask backend receives the input data from the user, processes it, and returns the predicted crop.

## ▶️ Usage

1.  **Run the Flask application:**
    ```bash
    python predict.py
    ```

2.  **Open your web browser and navigate to the local URL provided (usually `http://127.0.0.1:5000`).**

3.  **Enter the environmental data into the input fields and click "Predict" to see the recommended crop.**

## 💻 Technologies Used

- **Python:** The core programming language used for the application.
- **Flask:** A lightweight web framework for building the user interface.
- **Scikit-learn:** For building and using the machine learning model.
- **Pandas & NumPy:** For data manipulation and numerical operations.

## 📜 License

This project is licensed under the MIT License.
