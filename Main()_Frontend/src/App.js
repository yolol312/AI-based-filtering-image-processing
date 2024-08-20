import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Login from "./Login";
import Signup from "./Signup";
import Home from "./Home";
import { DataProvider } from "./Data/DataContext";
import { ClipDataProvider } from "./Data/ClipDataContext";
import { PersonDataProvider } from "./Data/PersonDataContext";
import { MapDataProvider } from "./Data/MapDataContext";
import { ProgressBarDataProvider } from "./Data/ProgressBarDataContext";
import { LogDataProvider } from "./Data/LogDataContext";
import { ImageLogDataProvider } from "./Data/ImageLogDataContext";
import { RealTimeDataProvider } from "./Data/RealTimeDataContext";
import { LogRealTimeDataProvider } from "./Data/LogRealTimeDataContext";
import ProtectedRoute from "./ProtectedRoute";

function App() {
  return (
    <DataProvider>
      <RealTimeDataProvider>
        <LogRealTimeDataProvider>
          <ClipDataProvider>
            <PersonDataProvider>
              <MapDataProvider>
                <ProgressBarDataProvider>
                  <LogDataProvider>
                    <ImageLogDataProvider>
                      <Router>
                        <Routes>
                          <Route path="/" element={<Login />} />
                          <Route path="/signup" element={<Signup />} />
                          <Route
                            path="/home"
                            element={<ProtectedRoute element={Home} />}
                          />
                        </Routes>
                      </Router>
                    </ImageLogDataProvider>
                  </LogDataProvider>
                </ProgressBarDataProvider>
              </MapDataProvider>
            </PersonDataProvider>
          </ClipDataProvider>
        </LogRealTimeDataProvider>
      </RealTimeDataProvider>
    </DataProvider>
  );
}

export default App;
