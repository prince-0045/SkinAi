import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/common/Navbar';
import Footer from './components/common/Footer';

// Pages
import Landing from './pages/Landing';
import Login from './pages/Login';
import Signup from './pages/Signup';
import OTP from './pages/OTP';
import Success from './pages/Success';
import Detect from './pages/Detect';
import Tracker from './pages/Tracker';
import Profile from './pages/Profile';
import { useAuth } from './context/AuthContext';

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) return <div>Loading...</div>; // Replace with proper loader

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
};

function App() {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/otp" element={<OTP />} />
          <Route path="/success" element={<Success />} />

          {/* Protected Routes */}
          <Route
            path="/detect"
            element={
              //   <ProtectedRoute>
              <Detect />
              //   </ProtectedRoute>
            }
          />
          <Route
            path="/track"
            element={
              //   <ProtectedRoute>
              <Tracker />
              //   </ProtectedRoute>
            }
          />
          <Route
            path="/profile"
            element={
              //   <ProtectedRoute>
              <Profile />
              //   </ProtectedRoute>
            }
          />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;
