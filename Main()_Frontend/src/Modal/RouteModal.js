// MapModal.js

import React, { useEffect, useCallback } from "react";
import "./RouteModal.css";
import RouteDraw from "./RouteDraw";

function RouteModal({ isOpen, onClose }) {
  //모달 창 닫기 이벤트
  const handleSimpleClose = useCallback(() => {
    onClose();
  }, [onClose]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        handleSimpleClose();
      }
    };

    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, handleSimpleClose]);

  if (!isOpen) return null;
  return (
    <div className="RouteModal-overlay">
      <div className="RouteModal-content">
        <div className="RouteModal-top">
          <button
            className="RouteModal-button-simple-close"
            onClick={handleSimpleClose}
          >
            X
          </button>
        </div>
        <div className="RouteModal-body">
          <RouteDraw onClose={handleSimpleClose} />
        </div>
      </div>
    </div>
  );
}

export default RouteModal;
