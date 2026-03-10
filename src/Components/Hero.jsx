// Hero.jsx
import React, { useState, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import CameraController from "./CameraController";
import ModelViewer from "./ModelViewer";
import ViewControls from "./ViewControls";
import FileUpload from "./FileUpload";
import PartsPricingTable from "./PartsPricingTable"; // Import the table component
// import StraightButton from "./StraightButton";
// import DimensionsDisplay from "./DimensionsDisplay";
import logo from "../assets/logo.png";
import "../App.css";
import { GizmoHelper, GizmoViewport } from "@react-three/drei";

const Hero = () => {
  const [modelData, setModelData] = useState(null);
  const [currentView, setCurrentView] = useState("front");
  const [initialDimensions, setInitialDimensions] = useState(null);  // Store initial dimensions
  const [reorientedDimensions, setReorientedDimensions] = useState(null);  // Store reoriented dimensions
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fileStates, setFileStates] = useState([]); // Add state for fileStates

  const modelViewerRef = useRef();

  const handleViewChange = (view) => {
    setCurrentView(view);
  };

  // Handle the Straighten Model button click to reset orientation and update dimensions
  const handleStraighten = () => {
    if (modelViewerRef.current) {
      modelViewerRef.current.resetModelOrientation(); // Reset model orientation
      setReorientedDimensions(initialDimensions);  // Update to the reoriented dimensions
    }
  };

  const handleRotate = (axis, amount) => {
    if (modelViewerRef.current) {
      modelViewerRef.current.rotateModel(axis, amount);
    }
  };

  // Callback to receive fileStates from FileUpload
  const handleFileStatesChange = (states) => {
    setFileStates(states);
  };

  return (
    <main className="bg-gray-900 w-screen h-screen overflow-x-hidden overflow-y-auto">
      <div className="flex items-center justify-between bg-gray-700 p-2 pb-3 rounded-t-xl">
        <div className="flex items-center">
          <img src={logo} alt="logo" className="w-16 h-16 sm:w-20 sm:h-20 mx-3 sm:mx-5" />
          <h1 className="text-white text-xl sm:text-3xl font-bold">3D Model Properties</h1>
        </div>
      </div>

      <div className="viewer-container flex flex-col lg:flex-row gap-4 w-full h-full p-4 pt-2">
        {error && (
          <div className="flex items-center justify-center text-2xl font-bold text-red-700 w-full">
            {error}
          </div>
        )}

        <div className="flex flex-col w-full lg:w-4/5">
          {/* 3D Model Viewer */}
          <div className="model border-2 rounded-2xl w-full h-[60vh] lg:h-[65vh] bg-[#FFFEFE] m-2 p-2">
            <Canvas
              camera={{ position: [0, 0, 5], fov: 45, near: 0.1, far: 1000 }}
            >
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} />
              <CameraController
                view={currentView}
                distance={initialDimensions ? initialDimensions.maxDimension * 2 : 50}
              />
              <ModelViewer
                ref={modelViewerRef}
                geometry={modelData}
                setDimensions={(dimensions) => {
                  if (!initialDimensions) {
                    setInitialDimensions(dimensions);
                  }
                }}
                setReorientedDimensions={setReorientedDimensions}
              />
              <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
                <GizmoViewport axisColors={["red", "green", "blue"]} labelColor="white" />
              </GizmoHelper>
            </Canvas>
          </div>

          {/* Parts & Pricing Table - Now appears below the 3D model viewer */}
          {modelData && fileStates.length > 0 && (
            <PartsPricingTable 
              fileStates={fileStates}
            />
          )}
        </div>

        <div className="sidebar flex flex-col items-center gap-4 w-full lg:w-1/5">
          <FileUpload
            onFileLoad={setModelData}
            onError={setError}
            onLoading={setLoading}
            onFileStatesChange={handleFileStatesChange} // Add this prop
          />
          {/* <StraightButton onStraighten={handleStraighten} /> */}
          <ViewControls
            currentView={currentView}
            onViewChange={handleViewChange}
            onRotate={handleRotate}
          />
          {/* <DimensionsDisplay
            dimensions={reorientedDimensions || initialDimensions}
            currentView={currentView}
          /> */}
        </div>
      </div>
    </main>
  );
};

export default Hero;