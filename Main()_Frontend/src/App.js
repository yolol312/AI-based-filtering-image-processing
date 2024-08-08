import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Login from './Login';
import Signup from './Signup';
import Home from './Home';
import { DataProvider } from './Data/DataContext';
import { ClipDataProvider } from './Data/ClipDataContext';
import { PersonDataProvider } from './Data/PersonDataContext';
import { MapDataProvider } from './Data/MapDataContext';
import { ProgressBarDataProvider } from './Data/ProgressBarDataContext';
import ProtectedRoute from './ProtectedRoute';

function App() {
  return (
    <DataProvider>
      <ClipDataProvider>
        <PersonDataProvider>
          <MapDataProvider>
            <ProgressBarDataProvider>
              <Router>
                <Routes>
                  <Route path="/" element={<Login />} />
                  <Route path="/signup" element={<Signup />} />
                  <Route path="/home" element={<ProtectedRoute element={Home} />} />
                </Routes>
              </Router>
            </ProgressBarDataProvider>
          </MapDataProvider>
        </PersonDataProvider>
      </ClipDataProvider>
    </DataProvider>
  );
}

export default App;
