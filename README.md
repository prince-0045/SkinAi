# SkinAi – Skin Disease Detection & Healing Tracker

SkinAi is an advanced AI-powered application designed to help users detect skin diseases and track their healing progress over time. Built with a modern tech stack, it combines deep learning for image analysis with a secure and user-friendly web interface.

## 🚀 Features

-   **AI Skin Disease Detection**: Upload an image of a skin condition to get an instant analysis and potential diagnosis.
-   **Healing Progress Tracker**: Compare images over time to visualize and track healing progress with side-by-side comparisons and percentage improvements.
-   **Secure Authentication**:
    -   Google Sign-In integration.
    -   Email/Password authentication with OTP verification.
-   **User Dashboard**: Personalized dashboard to manage scans, track history, and view reports.
-   **Modern UI/UX**: A responsive and aesthetic interface built with React and Tailwind CSS.

## 🛠️ Tech Stack

### Frontend
-   **React** (Vite)
-   **Tailwind CSS** (Styling)
-   **Lucide React** (Icons)
-   **Framer Motion** (Animations)

### Backend
-   **FastAPI** (Python)
-   **MongoDB** (Database)
-   **TensorFlow/Keras** (AI Model - *implied*)
-   **Pydantic** (Data Validation)

## 📦 Installation & Setup

Follow these steps to get the project running locally.

### Prerequisites
-   Node.js & npm installed.
-   Python 3.8+ installed.
-   MongoDB installed and running (or a MongoDB Atlas connection string).

### 1. Clone the Repository

```bash
git clone https://github.com/prince-0045/SkinAi.git
cd SkinAi
```

### 2. Backend Setup

Navigate to the backend directory and set up the Python environment.

```bash
cd backend
python -m venv venv
```

Activate the virtual environment:
-   **Windows**: `venv\Scripts\activate`
-   **Mac/Linux**: `source venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

**Environment Configuration (.env)**
Create a `.env` file in the `backend` directory with the following variables:
```env
MONGODB_URL=mongodb://localhost:27017/skinai  # Or your MongoDB Atlas URL
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email Configuration (for OTP)
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
MAIL_FROM=your_email@gmail.com
MAIL_PORT=587
MAIL_SERVER=smtp.gmail.com

# Google Auth
GOOGLE_CLIENT_ID=your_google_client_id
```

Run the backend server:
```bash
uvicorn app.main:app --reload
```
The backend will run at `http://localhost:8000`.

### 3. Frontend Setup

Open a new terminal, navigate to the frontend directory, and install dependencies.

```bash
cd frontend
npm install
```

**Environment Configuration (.env)**
Create a `.env` file in the `frontend` directory:
```env
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your_google_client_id
```

Run the frontend development server:
```bash
npm run dev
```
The application will usually run at `http://localhost:5173`.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## 📄 License

This project is licensed under the MIT License.
