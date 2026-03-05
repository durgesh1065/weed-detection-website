import React from "react";
import { createRoot } from "react-dom/client";
import CApp from "./CApp";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <CApp />
  </React.StrictMode>
);

