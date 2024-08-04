//ProtectedRoute.js
import React from 'react';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ element: Element }) => {
  const token = localStorage.getItem('token');

  return token ? <Element /> : <Navigate to="/" />;
};

export default ProtectedRoute;
