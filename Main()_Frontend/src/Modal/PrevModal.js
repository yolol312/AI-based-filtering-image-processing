import React, { useEffect, useCallback } from "react";
import "./PrevModal.css";
import PrevVideoPlay from "./PrevVideoPlay";

function PrevModal({ isOpen, onClose }) {
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
    <div className="Prev-modal-overlay">
      <div className="Prev-modal-content">
        <div className="Prev-modal-top">
          <h2>이전 분석 기록</h2>
          <button
            className="Prev-modal-button-simple-close"
            onClick={handleSimpleClose}
          >
            X
          </button>
        </div>
        <div className="Prev-modal-body">
          <PrevVideoPlay onClose={handleSimpleClose} />
        </div>
      </div>
    </div>
  );
}

export default PrevModal;
