import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from '../components/Layout';
import ScoutingForm from '../pages/ScoutingForm';
import Dashboard from '../pages/Dashboard';
import Login from '../pages/Login';
import Register from '../pages/Register';
import TeamDetails from '../pages/TeamDetails';
import PrivateRoute from '../components/PrivateRoute';

const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/scout" />} />
        <Route path="scout" element={<ScoutingForm />} />
        <Route path="login" element={<Login />} />
        <Route path="register" element={<Register />} />
        
        <Route element={<PrivateRoute />}>
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="team/:teamNumber" element={<TeamDetails />} />
        </Route>
      </Route>
    </Routes>
  );
};

export default AppRoutes; 