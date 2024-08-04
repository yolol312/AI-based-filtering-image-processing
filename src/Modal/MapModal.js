// MapModal.js

import React, { useEffect, useCallback } from 'react';
import './MapModal.css';
import AddressMapMarker from './AddressMapMarker';

function MapModal({ isOpen, onClose }) {


  //모달 창 닫기 이벤트
  const handleSimpleClose = useCallback(() => {
    onClose(); 
  }, [onClose]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        handleSimpleClose();
      }
    };

    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, handleSimpleClose]);

  if (!isOpen) return null;

  return (
    <div className="Map-modal-overlay">
      <div className="Map-modal-content">
        <div className="Map-modal-top">
          <button className="Map-modal-button-simple-close" onClick={handleSimpleClose}>
            닫기
          </button>
        </div>
        <div className="Map-modal-body">
          <AddressMapMarker onClose={handleSimpleClose} />
        </div>
      </div>
    </div>
  );
}

export default MapModal;
