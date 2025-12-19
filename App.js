import { useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./components/Home";
import TripPlanner from "./components/TripPlanner";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/plan" element={<TripPlanner />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
