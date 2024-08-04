// FileModal.js

import React, { useContext, useEffect, useCallback } from 'react';
import './FileModal.css';
import FileUploadFiltering from './FileUploadFiltering';
import { DataContext } from '../Data/DataContext';

// 환경 변수에서 서버 IP를 가져옵니다.
const Server_IP = process.env.REACT_APP_Server_IP;

function FileModal({ isOpen, onClose }) {
  const { data, setData } = useContext(DataContext);
  const userId = data.userId; // data에서 userId 추출

  const fetchProPersonInfo = async () => {
    if (!userId) {
      console.error('User ID is not available');
      return;
    }

    try {
      // fetch를 사용하여 서버에서 데이터를 가져옵니다.
      const response = await fetch(`${Server_IP}/update_pro_person?user_id=${userId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      console.log('받아온 provideopersoninfo:', data.provideo_person_info); // 콘솔에 데이터 출력
      // 가져온 데이터를 DataContext에 업데이트합니다.
      setData({ provideopersoninfo: data.provideo_person_info });
    } catch (error) {
      console.error('Error fetching map camera info:', error);
    }
  };

  const handleClose = () => {
    fetchProPersonInfo();
    onClose(); // 모달 닫기
  };

  const handleSimpleClose = useCallback(() => {
    onClose(); // 모달 단순히 닫기
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
    <div className="File-modal-overlay">
      <div className="File-modal-content">
        <div className="File-modal-top">
          <button className="File-modal-button-simple-close" onClick={handleSimpleClose}>
            닫기
          </button>
        </div>
        <div className="File-modal-body">
          <FileUploadFiltering />
        </div>
        <div className="File-modal-footer">
          <button className="File-modal-button-close" onClick={handleClose}>
            완료
          </button>
        </div>
      </div>
    </div>
  );
}

export default FileModal;
