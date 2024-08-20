// index.js

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css"; // index.css 파일 임포트

const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
